//! Where the engine keeps its models.
//!
//! The models live under the platform's local app data folder, next to the
//! app's own `Tinta` folder (`crates/tinta-core/src/paths.rs` uses the same
//! base). `data_local_dir` avoids syncing roughly 700 MB through a Windows
//! roaming profile.
//!
//! `TINTA_DATA_DIR` is the app's own override for the `Tinta` folder, so the
//! engine honours it too: the models then live in `<TINTA_DATA_DIR>/models`.
//! `TINTA_ENGINE_MODELS_DIR` overrides the whole path and is what tests and
//! manual runs use to stay out of the real app data.

use std::path::PathBuf;

pub fn models_root() -> PathBuf {
    if let Ok(dir) = std::env::var("TINTA_ENGINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("TINTA_DATA_DIR") {
        return PathBuf::from(dir).join("models");
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Tinta")
        .join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_win_when_set() {
        let previous_data = std::env::var_os("TINTA_DATA_DIR");
        let previous_models = std::env::var_os("TINTA_ENGINE_MODELS_DIR");

        std::env::remove_var("TINTA_ENGINE_MODELS_DIR");
        std::env::set_var("TINTA_DATA_DIR", "tinta-data-test");
        assert_eq!(
            models_root(),
            PathBuf::from("tinta-data-test").join("models")
        );

        std::env::set_var("TINTA_ENGINE_MODELS_DIR", "tinta-models-test");
        assert_eq!(models_root(), PathBuf::from("tinta-models-test"));

        match previous_models {
            Some(value) => std::env::set_var("TINTA_ENGINE_MODELS_DIR", value),
            None => std::env::remove_var("TINTA_ENGINE_MODELS_DIR"),
        }
        match previous_data {
            Some(value) => std::env::set_var("TINTA_DATA_DIR", value),
            None => std::env::remove_var("TINTA_DATA_DIR"),
        }
    }
}
