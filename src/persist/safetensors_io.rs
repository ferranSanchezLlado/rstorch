//! safetensors read/write for maps of [`HostTensor`]s, with symmetric,
//! allocate-last limit enforcement (16.14 WS2/WS3).
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

    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, t)| {
            let view = TensorView::new(to_st_dtype(t.dtype()), t.dims().to_vec(), t.bytes())
                .map_err(|e| st_err("build tensor view", e))?;
            Ok((name.as_str(), view))
        })
        .collect::<Result<_>>()?;

    safetensors::tensor::serialize(views, metadata).map_err(|e| st_err("serialize", e))
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
    let file_len = std::fs::metadata(path)?.len();
    let file_cap = limits
        .max_metadata_bytes
        .saturating_add(limits.max_total_bytes)
        .saturating_add(8);
    Limits::check("file bytes", file_len, file_cap)?;

    let buffer = std::fs::read(path)?;

    // 2. Header (metadata) size cap: the first 8 bytes are the little-endian
    //    header length. Check it before trusting the rest of the header.
    if buffer.len() < 8 {
        return Err(Error::Persistence {
            msg: format!("file too short: {} bytes", buffer.len()),
        });
    }
    let header_len = u64::from_le_bytes(buffer[..8].try_into().expect("8-byte slice"));
    Limits::check("metadata bytes", header_len, limits.max_metadata_bytes)?;

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

    Ok((metadata, buffer))
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
