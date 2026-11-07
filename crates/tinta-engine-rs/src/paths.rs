//! Where the engine keeps its models.
//!
//! The models live under the platform's app data folder, next to the app's own
//! `Tinta` folder (`crates/tinta-core/src/paths.rs` uses the same base). The app
//! sums this folder for Settings, so it must only contain installed models.
//!
//! `TINTA_ENGINE_MODELS_DIR` overrides the folder. Tests and manual runs use it
//! to stay out of the real app data.

use std::path::PathBuf;

pub fn models_root() -> PathBuf {
    if let Ok(dir) = std::env::var("TINTA_ENGINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Tinta")
        .join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_wins_when_set() {
        std::env::set_var("TINTA_ENGINE_MODELS_DIR", "/tmp/tinta-models-test");
        assert_eq!(models_root(), PathBuf::from("/tmp/tinta-models-test"));
        std::env::remove_var("TINTA_ENGINE_MODELS_DIR");
    }
}
