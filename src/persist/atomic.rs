//! Atomic temp-and-rename saves.
//!
//! A naive `File::create(path)` truncates an existing valid artifact the
//! instant it opens — a crash mid-write then leaves a corrupt file where a
//! good one used to be. Every save in this crate instead:
//!
//! 1. creates a **unique** temporary sibling in the destination's directory
//!    (never a fixed `.tmp`, so concurrent saves never collide);
//! 2. writes and flushes all bytes to it;
//! 3. `sync_all`s the file (durability of the data before the rename);
//! 4. atomically renames it over the destination;
//! 5. best-effort syncs the parent directory (durability of the rename);
//! 6. removes the temporary file on any earlier failure.
//!
//! # Platform guarantees
//!
//! [`std::fs::rename`] is atomic with respect to concurrent readers on Unix
//! and on Windows (`ReplaceFile`/`MoveFileEx`) for same-directory renames,
//! which is why the temp file is a sibling. The `sync_all` + parent-directory
//! sync give crash consistency on POSIX filesystems that honor `fsync`. This
//! is documented rather than claimed as universal: exotic or networked
//! filesystems may weaken these guarantees.

use crate::error::{Error, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Write `bytes` to `path` atomically (see module docs).
///
/// On success `path` is replaced in one step; on any failure `path` is left
/// exactly as it was and no temporary file remains.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic_with(path, |f| f.write_all(bytes))
}

/// Atomic-save core: `fill` writes the full contents into the temp file.
///
/// Factored out so a streaming writer (many sections) shares the exact same
/// durability and cleanup path as the single-buffer [`write_atomic`].
pub(crate) fn write_atomic_with<F>(path: &Path, fill: F) -> Result<()>
where
    F: FnOnce(&mut std::fs::File) -> std::io::Result<()>,
{
    write_atomic_with_candidates(path, fill, || unique_temp_path(path))
}

fn write_atomic_with_candidates<F, N>(path: &Path, fill: F, mut next: N) -> Result<()>
where
    F: FnOnce(&mut std::fs::File) -> std::io::Result<()>,
    N: FnMut() -> PathBuf,
{
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = parent.unwrap_or_else(|| Path::new("."));

    let (temp, mut file) = (0..128)
        .find_map(|_| {
            let temp = next();
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
            {
                Ok(file) => Some(Ok((temp, file))),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => Some(Err(e)),
            }
        })
        .transpose()?
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not exclusively create an atomic-save temp file after 128 attempts",
            )
        })?;
    // A guard that removes the temp file unless we explicitly disarm it after
    // a successful rename. This covers every early-return below.
    let mut guard = TempGuard::new(temp.clone());

    let result = (|| -> std::io::Result<()> {
        fill(&mut file)?;
        file.flush()?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            guard.disarm();
            // Best-effort: the data + rename are already durable per file;
            // syncing the directory makes the rename itself durable. A
            // filesystem that cannot open a directory as a file is not a
            // failure of the save.
            sync_dir_best_effort(dir);
            Ok(())
        }
        Err(e) => Err(Error::Io(e)),
    }
}

/// A unique sibling temp path: `.<name>.tmp-<pid>-<counter>`.
///
/// The process id plus a monotonic counter make the name unique across
/// concurrent saves within and across processes without needing a random
/// source dependency.
fn unique_temp_path(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let file_name = path
        .file_name()
        .map_or_else(|| "artifact".to_string(), |s| s.to_string_lossy().into_owned());
    let temp_name = format!(".{file_name}.tmp-{pid}-{n}");
    match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(parent) => parent.join(temp_name),
        None => PathBuf::from(temp_name),
    }
}

/// fsync the directory so the rename entry is durable. Not all platforms let
/// a directory be opened as a file; failures here are ignored on purpose.
fn sync_dir_best_effort(dir: &Path) {
    if let Ok(handle) = std::fs::File::open(dir) {
        let _ = handle.sync_all();
    }
}

/// Removes its path on drop unless disarmed.
struct TempGuard {
    path: Option<PathBuf>,
}

impl TempGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for TempGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rstorch-persist-atomic-{}-{}-{}",
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

    #[test]
    fn writes_bytes_atomically() {
        let dir = tmpdir("write");
        let path = dir.join("out.bin");
        write_atomic(&path, b"hello world").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello world");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failure_leaves_original_intact_and_no_temp() {
        let dir = tmpdir("fail");
        let path = dir.join("out.bin");
        write_atomic(&path, b"original").unwrap();

        // A fill closure that errors partway through must not clobber `path`.
        let err = write_atomic_with(&path, |f| {
            f.write_all(b"partial")?;
            Err(io::Error::other("boom"))
        });
        assert!(matches!(err, Err(Error::Io(_))));

        // Original is untouched.
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        // No temp file lingers.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "temp files leaked: {leftovers:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn overwrite_replaces_contents() {
        let dir = tmpdir("overwrite");
        let path = dir.join("out.bin");
        write_atomic(&path, b"v1").unwrap();
        write_atomic(&path, b"v2-longer").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"v2-longer");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn concurrent_temp_paths_are_unique() {
        let dir = tmpdir("unique");
        let path = dir.join("out.bin");
        let a = unique_temp_path(&path);
        let b = unique_temp_path(&path);
        assert_ne!(a, b);
        assert_eq!(a.parent(), Some(dir.as_path()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn preexisting_temp_symlink_is_not_followed() {
        use std::os::unix::fs::symlink;

        let dir = tmpdir("symlink");
        let path = dir.join("out.bin");
        let victim = dir.join("victim.bin");
        let collision = dir.join("predictable.tmp");
        let safe = dir.join("exclusive.tmp");
        std::fs::write(&victim, b"do not touch").unwrap();
        symlink(&victim, &collision).unwrap();
        let mut candidates = vec![safe.clone(), collision.clone()].into_iter().rev();

        write_atomic_with_candidates(
            &path,
            |file| file.write_all(b"checkpoint"),
            || candidates.next().unwrap(),
        )
        .unwrap();

        assert_eq!(std::fs::read(&victim).unwrap(), b"do not touch");
        assert_eq!(std::fs::read(&path).unwrap(), b"checkpoint");
        assert!(
            std::fs::symlink_metadata(&collision)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(!safe.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
