//! safetensors read/write for maps of [`HostTensor`]s, with symmetric,
//! allocate-last limit enforcement.
//!
//! Writing goes through the crate's [atomic](crate::persist) save path, and
//! the writer checks the **same** [`Limits`] the reader enforces, so the
//! library can never emit a file its own default reader would reject. Reading
//! inspects the file length and the parsed header *before* it slices out any
//! tensor data, so a file that lies about its sizes is rejected with an
//! [`Error::Persistence`](crate::Error::Persistence) rather than a runaway
//! allocation.

use crate::error::{Error, Result};
use crate::persist::atomic::write_atomic;
use crate::persist::host_tensor::{HostTensor, byte_len, from_st_dtype, to_st_dtype};
use crate::persist::options::Limits;
use safetensors::tensor::{Metadata, SafeTensors, TensorView};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Turn a [`safetensors`] error into a persistence error without leaking the
/// dependency's type into our public surface.
pub(crate) fn st_err(context: &str, e: safetensors::SafeTensorError) -> Error {
    Error::Persistence {
        msg: format!("{context}: {e}"),
    }
}

/// Serialize a name → [`HostTensor`] map plus an optional string-keyed
/// metadata map into safetensors bytes, validating writer limits first.
pub(crate) fn serialize_tensors(
    tensors: &BTreeMap<String, HostTensor>,
    metadata: Option<HashMap<String, String>>,
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_writer(tensors, limits)?;
    if let Some(metadata) = &metadata {
        check_metadata(metadata, limits)?;
    }

    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, t)| {
            let view = TensorView::new(to_st_dtype(t.dtype()), t.dims().to_vec(), t.bytes())
                .map_err(|e| st_err("build tensor view", e))?;
            Ok((name.as_str(), view))
        })
        .collect::<Result<_>>()?;

    // safetensors sorts tensors, but serializes free-form metadata directly
    // from a randomized HashMap. Add that object ourselves in key order so
    // equal checkpoints have identical bytes without changing the format.
    let mut bytes =
        safetensors::tensor::serialize(views, None).map_err(|e| st_err("serialize", e))?;
    if let Some(metadata) = metadata {
        bytes = add_canonical_metadata(bytes, metadata)?;
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("serialized header"));
    Limits::check("metadata bytes", header_len, limits.max_metadata_bytes)?;
    Ok(bytes)
}

fn check_metadata(metadata: &HashMap<String, String>, limits: &Limits) -> Result<()> {
    for (key, value) in metadata {
        Limits::check(
            "metadata key length",
            key.len() as u64,
            limits.max_string_bytes,
        )?;
        Limits::check(
            "metadata value length",
            value.len() as u64,
            limits.max_string_bytes,
        )?;
    }
    Ok(())
}

fn add_canonical_metadata(bytes: Vec<u8>, metadata: HashMap<String, String>) -> Result<Vec<u8>> {
    let old_header_len =
        u64::from_le_bytes(bytes[..8].try_into().expect("serialized header")) as usize;
    let old_data_start = 8 + old_header_len;
    let mut header = bytes[8..old_data_start].to_vec();
    while header.last() == Some(&b' ') {
        header.pop();
    }
    if header.pop() != Some(b'}') {
        return Err(Error::Persistence {
            msg: "serialized safetensors header is not a JSON object".to_string(),
        });
    }
    if header.len() > 1 {
        header.push(b',');
    }
    header.extend_from_slice(b"\"__metadata__\":{");
    for (index, (key, value)) in metadata
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .iter()
        .enumerate()
    {
        if index != 0 {
            header.push(b',');
        }
        push_json_string(&mut header, key);
        header.push(b':');
        push_json_string(&mut header, value);
    }
    header.extend_from_slice(b"}}");
    header.resize(header.len().next_multiple_of(8), b' ');

    let mut out = Vec::with_capacity(8 + header.len() + bytes.len() - old_data_start);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&bytes[old_data_start..]);
    Ok(out)
}

fn push_json_string(out: &mut Vec<u8>, value: &str) {
    out.push(b'"');
    for ch in value.chars() {
        match ch {
            '"' => out.extend_from_slice(br#"\""#),
            '\\' => out.extend_from_slice(br"\\"),
            '\u{08}' => out.extend_from_slice(br"\b"),
            '\u{0c}' => out.extend_from_slice(br"\f"),
            '\n' => out.extend_from_slice(br"\n"),
            '\r' => out.extend_from_slice(br"\r"),
            '\t' => out.extend_from_slice(br"\t"),
            ch if ch <= '\u{1f}' => {
                out.extend_from_slice(format!("\\u{:04x}", ch as u32).as_bytes());
            }
            ch => {
                let mut encoded = [0; 4];
                out.extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
            }
        }
    }
    out.push(b'"');
}

/// Atomically write a tensor map (with optional metadata) to `path`.
pub(crate) fn save_tensors(
    path: &Path,
    tensors: &BTreeMap<String, HostTensor>,
    metadata: Option<HashMap<String, String>>,
    limits: &Limits,
) -> Result<()> {
    let bytes = serialize_tensors(tensors, metadata, limits)?;
    write_atomic(path, &bytes)
}

/// Writer-side limit check: mirrors every reader bound so the writer can never
/// produce a file the default reader would reject.
pub(crate) fn check_writer(tensors: &BTreeMap<String, HostTensor>, limits: &Limits) -> Result<()> {
    Limits::check("record count", tensors.len() as u64, limits.max_records)?;
    let mut total: u64 = 0;
    for (name, t) in tensors {
        Limits::check("name length", name.len() as u64, limits.max_string_bytes)?;
        Limits::check("rank", t.dims().len() as u64, limits.max_rank)?;
        let n = t.bytes().len() as u64;
        Limits::check("tensor bytes", n, limits.max_tensor_bytes)?;
        total = total.checked_add(n).ok_or_else(|| Error::Persistence {
            msg: "total byte length overflow".to_string(),
        })?;
    }
    Limits::check("total bytes", total, limits.max_total_bytes)?;
    Ok(())
}

/// Read the whole file into memory with an up-front size cap, then parse and
/// return `(header_size, Metadata, full_buffer)` after enforcing every limit
/// on the *declared* sizes — before any tensor data is materialized.
///
/// The buffer is returned so the caller can hand out tensor views without a
/// second read; every declared offset in the metadata has already been
/// validated by safetensors against this buffer's length.
pub(crate) fn read_and_validate(path: &Path, limits: &Limits) -> Result<(Metadata, Vec<u8>)> {
    // 1. File length cap before we allocate a buffer for it. The whole file is
    //    header + all tensor data, so bound it by metadata + total budget.
    //
    //    The handle is opened once and everything below is derived from it: a
    //    cap taken from a separate `fs::metadata` call and then handed to
    //    `fs::read` would be advisory only, because the read runs to EOF no
    //    matter what the earlier stat said. Two ways that bites: a stat of a
    //    non-regular file (a FIFO reports length 0) passes any cap and then
    //    streams unboundedly, and a regular file can be grown or swapped
    //    between the stat and the read. So: reject anything that is not a
    //    regular file, and bound the read itself.
    use std::io::Read as _;
    let file = std::fs::File::open(path)?;
    let file_meta = file.metadata()?;
    if !file_meta.is_file() {
        return Err(Error::Persistence {
            msg: format!("{} is not a regular file", path.display()),
        });
    }
    let file_cap = limits
        .max_metadata_bytes
        .saturating_add(limits.max_total_bytes)
        .saturating_add(8);
    Limits::check("file bytes", file_meta.len(), file_cap)?;

    // Read one byte past the cap so an over-cap file is detected rather than
    // silently truncated to a prefix that might still parse.
    let mut buffer = Vec::new();
    file.take(file_cap.saturating_add(1))
        .read_to_end(&mut buffer)?;
    Limits::check("file bytes", buffer.len() as u64, file_cap)?;

    // 2. Header (metadata) size cap: the first 8 bytes are the little-endian
    //    header length. Check it before trusting the rest of the header.
    if buffer.len() < 8 {
        return Err(Error::Persistence {
            msg: format!("file too short: {} bytes", buffer.len()),
        });
    }
    let header_len = u64::from_le_bytes(buffer[..8].try_into().expect("8-byte slice"));
    Limits::check("metadata bytes", header_len, limits.max_metadata_bytes)?;

    let header_end = 8usize
        .checked_add(header_len.try_into().map_err(|_| Error::Persistence {
            msg: "metadata byte length does not fit this platform".to_string(),
        })?)
        .ok_or_else(|| Error::Persistence {
            msg: "metadata byte length overflow".to_string(),
        })?;
    let raw_header = buffer
        .get(8..header_end)
        .ok_or_else(|| Error::Persistence {
            msg: "declared metadata extends past end of file".to_string(),
        })?;
    reject_duplicate_json_keys(raw_header)?;

    // 3. Parse the header only (no tensor allocation) and validate structure.
    let (_n, metadata) =
        SafeTensors::read_metadata(&buffer).map_err(|e| st_err("read metadata", e))?;

    // 4. Enforce structural limits against the *declared* metadata before the
    //    caller slices out any tensor data.
    let infos = metadata.tensors();
    Limits::check("record count", infos.len() as u64, limits.max_records)?;
    let mut total: u64 = 0;
    for (name, info) in &infos {
        Limits::check("name length", name.len() as u64, limits.max_string_bytes)?;
        Limits::check("rank", info.shape.len() as u64, limits.max_rank)?;
        let (start, end) = info.data_offsets;
        let n = end.saturating_sub(start) as u64;
        Limits::check("tensor bytes", n, limits.max_tensor_bytes)?;
        total = total.checked_add(n).ok_or_else(|| Error::Persistence {
            msg: "total byte length overflow".to_string(),
        })?;
    }
    Limits::check("total bytes", total, limits.max_total_bytes)?;
    if let Some(values) = metadata.metadata() {
        check_metadata(values, limits)?;
    }

    Ok((metadata, buffer))
}

/// Reject a header whose objects contain duplicate keys, before
/// `serde_json`'s map representation can collapse them.
///
/// A header that two JSON parsers read differently is a header we refuse: the
/// reader that decides shapes and the reader that decides bytes must agree.
/// `serde_json` does the lexing (escapes, surrogate pairs, malformed
/// brackets, trailing data); the visitor below adds the two rules it does not
/// enforce for us — no repeated key in any object, and a nesting bound.
fn reject_duplicate_json_keys(header: &[u8]) -> Result<()> {
    serde_json::from_slice::<Checked>(header)
        .map(|_| ())
        .map_err(|e| Error::Persistence {
            msg: format!("invalid safetensors JSON header: {e}"),
        })
}

/// Ordinary safetensors headers use only a few levels. The visitor recurses
/// with `serde_json`, so bounding nesting as it descends keeps both away from
/// the call-stack limit even for a header close to the byte limit.
const MAX_JSON_NESTING: usize = 32;

/// A JSON value deserialized only for its side effects: it keeps no data, and
/// exists so that every object in the header is visited and checked.
struct Checked;

impl<'de> serde::Deserialize<'de> for Checked {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(Descent { depth: 0 })
    }
}

/// The visitor proper, carrying how many `{`/`[` levels it is already inside.
#[derive(Clone, Copy)]
struct Descent {
    depth: usize,
}

impl<'de> serde::de::Visitor<'de> for Descent {
    type Value = Checked;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Checked, A::Error> {
        let inner = self.descend::<A::Error>()?;
        let mut seen = std::collections::HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate safetensors header field {key:?}"
                )));
            }
            map.next_value_seed(inner)?;
        }
        Ok(Checked)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Checked, A::Error> {
        let inner = self.descend::<A::Error>()?;
        while seq.next_element_seed(inner)?.is_some() {}
        Ok(Checked)
    }

    // Scalars carry nothing to check.
    fn visit_bool<E>(self, _: bool) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }
    fn visit_i64<E>(self, _: i64) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }
    fn visit_u64<E>(self, _: u64) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }
    fn visit_f64<E>(self, _: f64) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }
    fn visit_str<E>(self, _: &str) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }
    fn visit_unit<E>(self) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }
    fn visit_none<E>(self) -> std::result::Result<Checked, E> {
        Ok(Checked)
    }

    fn visit_some<D: serde::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Checked, D::Error> {
        d.deserialize_any(self)
    }
}

impl Descent {
    /// The visitor for one level further in, or the nesting error.
    fn descend<E: serde::de::Error>(self) -> std::result::Result<Descent, E> {
        if self.depth == MAX_JSON_NESTING {
            return Err(E::custom(format!(
                "safetensors JSON nesting exceeds limit {MAX_JSON_NESTING}"
            )));
        }
        Ok(Descent {
            depth: self.depth + 1,
        })
    }
}

/// `Descent` is its own seed, so a nested value inherits the running depth.
impl<'de> serde::de::DeserializeSeed<'de> for Descent {
    type Value = Checked;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Checked, D::Error> {
        deserializer.deserialize_any(self)
    }
}

/// Load a name → [`HostTensor`] map plus the file's string metadata map,
/// enforcing `limits` on every declared size first.
pub(crate) fn load_tensors(
    path: &Path,
    limits: &Limits,
) -> Result<(BTreeMap<String, HostTensor>, HashMap<String, String>)> {
    let (metadata, buffer) = read_and_validate(path, limits)?;
    let meta_map = metadata.metadata().clone().unwrap_or_default();

    // Now that the declared sizes are within budget, materialize the views.
    let st = SafeTensors::deserialize(&buffer).map_err(|e| st_err("deserialize", e))?;
    let mut out = BTreeMap::new();
    for (name, view) in st.tensors() {
        let dtype = from_st_dtype(view.dtype())?;
        let dims = view.shape().to_vec();
        // Defence in depth: safetensors already checked the view length, but
        // re-validate against our own dtype/dims arithmetic.
        let expected = byte_len(dtype, &dims)?;
        let data = view.data();
        if data.len() != expected {
            return Err(Error::Persistence {
                msg: format!(
                    "tensor `{name}` byte length {} does not match dtype {dtype} dims {dims:?} ({expected})",
                    data.len()
                ),
            });
        }
        out.insert(name, HostTensor::from_bytes(dtype, dims, data.to_vec())?);
    }
    Ok((out, meta_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rstorch-persist-st-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample() -> BTreeMap<String, HostTensor> {
        let mut m = BTreeMap::new();
        m.insert(
            "w".to_string(),
            HostTensor::from_bytes(DType::F32, vec![2, 2], 1.0f32.to_le_bytes().repeat(4)).unwrap(),
        );
        m.insert(
            "b".to_string(),
            HostTensor::from_bytes(DType::I64, vec![3], 7i64.to_le_bytes().repeat(3)).unwrap(),
        );
        m
    }

    fn raw_file(header: &str) -> Vec<u8> {
        let mut header = header.as_bytes().to_vec();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes
    }

    /// The size cap must bound the *read*, not merely a preceding `stat`.
    /// A FIFO reports length 0, so a stat-only cap admits it and then reads
    /// until the writer stops — unbounded memory from a path the caller was
    /// told had been size-checked. A non-regular file must be refused outright.
    #[cfg(unix)]
    #[test]
    fn non_regular_files_are_refused_rather_than_streamed() {
        use std::os::unix::ffi::OsStrExt as _;

        let dir = tmpdir("fifo");
        let path = dir.join("pipe.safetensors");

        // `mkfifo(2)` via libc is unavailable here, so shell out to `mkfifo(1)`.
        let made = std::process::Command::new("mkfifo")
            .arg(path.as_os_str())
            .status();
        let Ok(status) = made else {
            return; // no `mkfifo` binary: nothing to assert on this platform
        };
        assert!(status.success(), "mkfifo failed");
        assert!(
            !path.as_os_str().as_bytes().is_empty(),
            "fifo path must be non-empty"
        );

        // A writer that would otherwise feed the reader forever. Opening a FIFO
        // for reading blocks until a writer appears, so spawn one that emits a
        // little data and exits; the assertion is that we reject on file *type*,
        // before length ever matters.
        let mut writer = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!(
                "yes 0123456789 > {} 2>/dev/null || true",
                path.display()
            ))
            .spawn()
            .expect("spawn writer");

        let err = load_tensors(&path, &Limits::defaults())
            .expect_err("a FIFO must not be accepted as a checkpoint");
        assert!(
            matches!(&err, Error::Persistence { msg } if msg.contains("not a regular file")),
            "expected a regular-file rejection, got {err:?}"
        );

        let _ = writer.kill();
        let _ = writer.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A file larger than the cap is rejected by the bounded read even when the
    /// declared header is small, and the error names the file-length budget.
    #[test]
    fn over_cap_file_is_rejected_by_the_bounded_read() {
        let dir = tmpdir("cap");
        let path = dir.join("big.safetensors");
        let limits = Limits {
            max_metadata_bytes: 64,
            max_total_bytes: 64,
            ..Limits::defaults()
        };
        // Well past `64 + 64 + 8`.
        std::fs::write(&path, vec![0u8; 4096]).unwrap();

        let err = load_tensors(&path, &limits).expect_err("over-cap file must be rejected");
        assert!(
            matches!(&err, Error::Persistence { msg } if msg.contains("file bytes")),
            "expected a file-bytes cap error, got {err:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_tensors_and_metadata() {
        let dir = tmpdir("rt");
        let path = dir.join("m.safetensors");
        let tensors = sample();
        let mut meta = HashMap::new();
        meta.insert("k".to_string(), "v".to_string());

        save_tensors(&path, &tensors, Some(meta.clone()), &Limits::defaults()).unwrap();
        let (back, back_meta) = load_tensors(&path, &Limits::defaults()).unwrap();
        assert_eq!(back, tensors);
        assert_eq!(back_meta.get("k").map(String::as_str), Some("v"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn metadata_serialization_is_canonical_and_ecosystem_compatible() {
        let tensors = sample();
        let mut first = HashMap::new();
        first.insert("z".to_string(), "last".to_string());
        first.insert("a".to_string(), "first".to_string());
        let mut second = HashMap::new();
        second.insert("a".to_string(), "first".to_string());
        second.insert("z".to_string(), "last".to_string());

        let a = serialize_tensors(&tensors, Some(first), &Limits::defaults()).unwrap();
        let b = serialize_tensors(&tensors, Some(second), &Limits::defaults()).unwrap();
        assert_eq!(a, b);

        let compatible = SafeTensors::deserialize(&a).unwrap();
        assert_eq!(compatible.len(), tensors.len());
        let (_, parsed) = SafeTensors::read_metadata(&a).unwrap();
        assert_eq!(parsed.metadata().as_ref().unwrap()["a"], "first");
    }

    #[test]
    fn writer_and_reader_apply_the_same_metadata_limits() {
        let dir = tmpdir("metadata-limits");
        let path = dir.join("m.safetensors");
        let tensors = sample();
        let mut metadata = HashMap::new();
        metadata.insert("section".to_string(), "value".to_string());

        let bytes =
            serialize_tensors(&tensors, Some(metadata.clone()), &Limits::defaults()).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let mut exact = Limits::defaults();
        exact.max_metadata_bytes = header_len;
        exact.max_string_bytes = "section".len() as u64;
        save_tensors(&path, &tensors, Some(metadata.clone()), &exact).unwrap();
        load_tensors(&path, &exact).unwrap();

        exact.max_metadata_bytes = header_len - 1;
        assert!(save_tensors(&path, &tensors, Some(metadata.clone()), &exact).is_err());
        exact.max_metadata_bytes = header_len;
        exact.max_string_bytes = "value".len() as u64 - 1;
        assert!(save_tensors(&path, &tensors, Some(metadata), &exact).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn raw_duplicate_metadata_and_tensor_names_are_rejected() {
        let dir = tmpdir("duplicates");
        let cases = [
            r#"{"__metadata__":{"rstorch.magic":"bad","rstorch.magic":"good"}}"#,
            r#"{"__metadata__":{"rstorch.magic":"bad","rstorch.\u006dagic":"good"}}"#,
            r#"{"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
        ];
        for (index, header) in cases.iter().enumerate() {
            let path = dir.join(format!("duplicate-{index}.safetensors"));
            std::fs::write(&path, raw_file(header)).unwrap();
            let message = load_tensors(&path, &Limits::defaults())
                .unwrap_err()
                .to_string();
            assert!(
                message.contains("duplicate safetensors header field"),
                "{message}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn deeply_nested_near_limit_headers_reject_on_small_stacks() {
        let max = Limits::defaults().max_metadata_bytes as usize;
        let depth = 100_000;
        let mut arrays = "[".repeat(depth);
        arrays.push('0');
        arrays.push_str(&"]".repeat(depth));
        arrays.push_str(&" ".repeat(max - arrays.len()));

        let mut objects = "{\"a\":".repeat(depth);
        objects.push('0');
        objects.push_str(&"}".repeat(depth));
        objects.push_str(&" ".repeat(max - objects.len()));

        for (name, header) in [("arrays", arrays), ("objects", objects)] {
            std::thread::Builder::new()
                .name(format!("nested-{name}"))
                .stack_size(64 * 1024)
                .spawn(move || {
                    let error = reject_duplicate_json_keys(header.as_bytes()).unwrap_err();
                    assert!(
                        error.to_string().contains("nesting exceeds limit"),
                        "{error}"
                    );
                })
                .unwrap()
                .join()
                .unwrap();
        }
    }

    #[test]
    fn reasonable_json_nesting_is_accepted() {
        let depth = MAX_JSON_NESTING / 2;
        let mut header = "[".repeat(depth);
        header.push_str(r#"{"decoded\u002dkey":{"inner":true}}"#);
        header.push_str(&"]".repeat(depth));
        reject_duplicate_json_keys(header.as_bytes()).unwrap();
    }

    #[test]
    fn malformed_json_brackets_and_escapes_are_structured_errors() {
        let malformed = [
            r#"{"a":[}"#,
            r#"{"a":[]]"#,
            r#"{"a":{"b":0}"#,
            r#"{"a":"unterminated}"#,
            r#"{"a":"\q"}"#,
            r#"{"a":"\u12xz"}"#,
            r#"{"a":"\ud800x"}"#,
        ];
        for header in malformed {
            assert!(
                matches!(
                    reject_duplicate_json_keys(header.as_bytes()),
                    Err(Error::Persistence { .. })
                ),
                "accepted malformed header {header:?}"
            );
        }
    }

    #[test]
    fn writer_rejects_over_record_limit() {
        let tensors = sample();
        let mut limits = Limits::defaults();
        limits.max_records = 1;
        assert!(matches!(
            check_writer(&tensors, &limits),
            Err(Error::Persistence { .. })
        ));
    }

    #[test]
    fn reader_rejects_over_tensor_bytes_limit() {
        let dir = tmpdir("limit");
        let path = dir.join("m.safetensors");
        save_tensors(&path, &sample(), None, &Limits::defaults()).unwrap();

        let mut limits = Limits::defaults();
        // The f32 2x2 tensor is 16 bytes; cap per-tensor at 8.
        limits.max_tensor_bytes = 8;
        assert!(matches!(
            load_tensors(&path, &limits),
            Err(Error::Persistence { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reader_rejects_over_total_bytes_limit() {
        let dir = tmpdir("total");
        let path = dir.join("m.safetensors");
        save_tensors(&path, &sample(), None, &Limits::defaults()).unwrap();

        let mut limits = Limits::defaults();
        limits.max_total_bytes = 10; // both tensors together exceed this.
        assert!(matches!(
            load_tensors(&path, &limits),
            Err(Error::Persistence { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reader_rejects_corrupt_file() {
        let dir = tmpdir("corrupt");
        let path = dir.join("m.safetensors");
        std::fs::write(&path, b"not a safetensors file at all").unwrap();
        assert!(matches!(
            load_tensors(&path, &Limits::defaults()),
            Err(Error::Persistence { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reader_rejects_truncated_header() {
        let dir = tmpdir("trunc");
        let path = dir.join("m.safetensors");
        std::fs::write(&path, [1u8, 2, 3]).unwrap(); // < 8 bytes.
        assert!(matches!(
            load_tensors(&path, &Limits::defaults()),
            Err(Error::Persistence { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reader_rejects_giant_declared_header() {
        let dir = tmpdir("bighdr");
        let path = dir.join("m.safetensors");
        // Declare an 8-byte header length of u64::MAX with no body.
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            load_tensors(&path, &Limits::defaults()),
            Err(Error::Persistence { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
