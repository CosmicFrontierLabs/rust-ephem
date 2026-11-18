//! Shared EOP2 download and caching utilities
//!
//! Provides a single place to download the JPL EOP2 "short" file,
//! store it in a local cache, and read it back with a TTL.
//! Both UT1 and polar motion providers can consume the returned text.

use crate::utils::config::DEFAULT_EOP_PATH;
use crate::utils::config::DEFAULT_EOP_TTL;
use crate::utils::config::EOP2_URL;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eop2Provenance {
    pub source_url: &'static str,
    pub sha256: String,
    pub loaded_from: &'static str,
    pub stale: bool,
}

#[derive(Debug, Clone)]
pub struct Eop2Data {
    pub text: String,
    pub provenance: Eop2Provenance,
}

fn eop2_data(text: String, loaded_from: &'static str, stale: bool) -> Eop2Data {
    let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    Eop2Data {
        text,
        provenance: Eop2Provenance {
            source_url: EOP2_URL,
            sha256,
            loaded_from,
            stale,
        },
    }
}

fn fetch_eop2_text() -> Result<String, Box<dyn Error>> {
    let mut response = ureq::get(EOP2_URL).call()?;
    Ok(response.body_mut().read_to_string()?)
}

fn try_read_fresh_cache(path: &Path, ttl: Duration) -> Option<Eop2Data> {
    let meta = fs::metadata(path).ok()?;
    if let Ok(modified) = meta.modified() {
        // Clock skew or preserved copy timestamps can put mtime in the future.
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();
        if age <= ttl {
            if let Ok(mut f) = fs::File::open(path) {
                let mut buf = String::new();
                if f.read_to_string(&mut buf).is_ok() {
                    eprintln!(
                        "EOP2 text loaded from cache: {} (age: {}s)",
                        path.display(),
                        age.as_secs()
                    );
                    return Some(eop2_data(buf, "fresh_cache", false));
                }
            }
        }
    }
    None
}

fn save_cache(path: &Path, body: &str) -> std::io::Result<()> {
    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    // Use the same directory so rename atomically replaces the cache, and
    // create_new so concurrent writers never share a temporary file.
    let (temp_path, mut file) = loop {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temp_path = parent.join(format!(".eop2-{}-{id}.tmp", std::process::id()));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => break (temp_path, file),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    };
    let result = (|| {
        file.write_all(body.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp_path, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn try_read_stale_cache(path: &Path) -> Option<Eop2Data> {
    if path.exists() {
        if let Ok(mut f) = fs::File::open(path) {
            let mut buf = String::new();
            if f.read_to_string(&mut buf).is_ok() {
                eprintln!("EOP2 text loaded from STALE cache: {}", path.display());
                return Some(eop2_data(buf, "stale_cache", true));
            }
        }
    }
    None
}

/// Load EOP2 data and provenance: use cache if fresh, or download and update cache.
/// Falls back to stale cache if network fails.
pub fn load_or_download_eop2_data() -> Result<Eop2Data, Box<dyn Error>> {
    let path = DEFAULT_EOP_PATH.clone();
    let ttl = Duration::from_secs(*DEFAULT_EOP_TTL);
    if let Some(text) = try_read_fresh_cache(&path, ttl) {
        return Ok(text);
    }
    match fetch_eop2_text() {
        Ok(text) => {
            if let Err(err) = save_cache(&path, &text) {
                eprintln!(
                    "Warning: failed to save EOP2 cache at {} ({err}); using downloaded data \
                     for this run, but it will not be cached.",
                    path.display()
                );
            }
            Ok(eop2_data(text, "download", false))
        }
        Err(e) => {
            eprintln!("Warning: failed to download EOP2 after cache miss ({e}).");
            if let Some(text) = try_read_stale_cache(&path) {
                return Ok(text);
            }
            Err("Unable to load EOP2 from cache or network".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{eop2_data, save_cache, try_read_fresh_cache, try_read_stale_cache};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, SystemTime};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "rust-ephem-eop-test-{}-{}",
                    std::process::id(),
                    NEXT_ID.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(err) => panic!("Cannot create test directory: {err}"),
                }
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn future_mtime_is_fresh_even_with_zero_ttl() {
        let dir = TestDir::new();
        let path = dir.0.join("latest_eop2.short");
        fs::write(&path, "pinned data").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(7200))
            .unwrap();

        for ttl in [0, 3_153_600_000] {
            let data = try_read_fresh_cache(&path, Duration::from_secs(ttl)).unwrap();
            assert_eq!(data.text, "pinned data");
            assert_eq!(data.provenance.loaded_from, "fresh_cache");
            assert!(!data.provenance.stale);
        }
    }

    #[test]
    fn expired_cache_remains_available_as_stale_and_never_ttl_keeps_it_fresh() {
        let dir = TestDir::new();
        let path = dir.0.join("latest_eop2.short");
        fs::write(&path, "pinned data").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(3 * 86_400))
            .unwrap();

        assert!(try_read_fresh_cache(&path, Duration::from_secs(86_400)).is_none());
        let data = try_read_stale_cache(&path).unwrap();
        assert_eq!(data.text, "pinned data");
        assert!(data.provenance.stale);
        assert!(try_read_fresh_cache(&path, Duration::from_secs(u64::MAX)).is_some());
    }

    #[test]
    fn concurrent_cache_replacements_leave_complete_files() {
        let dir = TestDir::new();
        let path = dir.0.join("nested/latest_eop2.short");
        let original = "old data".repeat(1024);
        save_cache(&path, &original).unwrap();
        let bodies: Vec<_> = (0..4)
            .map(|i| format!("writer {i}\n").repeat(1024))
            .collect();
        std::thread::scope(|scope| {
            let writers: Vec<_> = bodies
                .iter()
                .map(|body| {
                    let path = &path;
                    scope.spawn(move || save_cache(path, body).unwrap())
                })
                .collect();
            for _ in 0..100 {
                let text = fs::read_to_string(&path).unwrap();
                assert!(text == original || bodies.contains(&text));
            }
            for writer in writers {
                writer.join().unwrap();
            }
        });
        assert!(bodies.contains(&fs::read_to_string(&path).unwrap()));
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn failed_cache_writes_return_errors_and_clean_up_temporary_files() {
        let dir = TestDir::new();
        // A directory at the target forces rename to fail after writing.
        let path = dir.0.join("latest_eop2.short");
        fs::create_dir(&path).unwrap();
        fs::write(path.join("sentinel"), "keep me").unwrap();
        assert!(save_cache(&path, "new data").is_err());
        assert_eq!(
            fs::read_to_string(path.join("sentinel")).unwrap(),
            "keep me"
        );
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        // A file in place of the parent forces directory creation to fail.
        assert!(save_cache(&path.join("sentinel/cache"), "new data").is_err());
    }

    #[test]
    fn provenance_hashes_the_exact_loaded_text() {
        let data = eop2_data("abc".to_string(), "download", false);

        assert_eq!(
            data.provenance.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(data.provenance.loaded_from, "download");
        assert!(!data.provenance.stale);
    }
}
