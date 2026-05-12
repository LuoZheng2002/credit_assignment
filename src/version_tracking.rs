use base64::{Engine, engine::general_purpose};
use serde::{Deserialize, Serialize};

pub trait AssetFile {
    type FileModel;
    // this function regenerates the file if it is missing or outdated, and returns the hash of the file after synchronization
    // the criteria for regenerating the file can be:
    // 1. The tracking file is missing (we assert that if the tracking file exists, then the target file must also exist)
    // 2. The hash of the dependencies (if any) has changed compared to the hash stored in the tracking file, which indicates that the content of the file is stale
    // We call .synchronize() for dependencies to ensure they are up-to-date, and compare the returned hash with the hash stored in self's tracking file to determine whether the target file is stale.
    // If the file needs regeneration, then we call .fetch() on the dependencies to get their contents.
    // fetch() should call synchronize() at the beginning.
    fn synchronize(&self) -> Base64Hash;
    fn fetch(&self) -> Self::FileModel;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Base64Hash(String);

pub fn hash_file(file_path: impl AsRef<std::path::Path>) -> Result<Base64Hash, String> {
    let file = std::fs::File::open(file_path.as_ref())
        .map_err(|e| format!("Cannot open file {}: {}", file_path.as_ref().display(), e))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut reader, &mut hasher).map_err(|e| {
        format!(
            "Failed to read file {}: {}",
            file_path.as_ref().display(),
            e
        )
    })?;
    let hash = hasher.finalize();
    Ok(Base64Hash(
        general_purpose::STANDARD.encode(hash.as_bytes()),
    ))
}
