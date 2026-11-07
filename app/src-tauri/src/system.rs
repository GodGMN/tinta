//! The Chrome extension folder. The registration of the native host lives in the platform
//! files; on Linux it writes the manifest into each Chromium browser that the user runs.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// The folder that Chrome loads the unpacked extension from.
pub fn extension_dir() -> PathBuf {
    tinta_core::paths::base_dir().join("Chrome extension")
}

/// Replaces the extension folder with the bundled extension.
pub fn copy_extension(source: &Path) -> Result<PathBuf> {
    let target = extension_dir();
    if target.exists() {
        std::fs::remove_dir_all(&target)?;
    }
    crate::copy_dir(source, &target)?;
    Ok(target)
}

/// Updates the extension folder when the bundled extension has a different version.
/// Chrome loads the new files the next time that it starts.
pub fn update_extension(source: &Path) {
    let version = |dir: &Path| {
        std::fs::read(dir.join("manifest.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|manifest| manifest["version"].as_str().map(str::to_string))
    };
    let target = extension_dir();
    if target.exists() && version(source).is_some() && version(source) != version(&target) {
        let _ = copy_extension(source);
    }
}
