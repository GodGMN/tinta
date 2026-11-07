//! The Windows and Linux engine for Tinta.
//!
//! This crate is the second implementation of the engine protocol in
//! `docs/engine-protocol.md`. It runs as `tinta-engine`, talks JSON lines on
//! stdin/stdout and logs to stderr with the `[tinta-engine]` prefix.
//!
//! The engine implements the process lifecycle, `ping`, `models_status`,
//! `install_models`, `permissions`, `list_sources`, `sample`, `import_audio`,
//! `finalize` and the documented degraded replies for the macOS-only commands.
//! Recording commands are a later task and fail with an `unsupported` error.

pub mod audio;
pub mod clustering;
pub mod echo;
pub mod engine;
pub mod finalize;
pub mod import;
pub mod install;
pub mod language;
pub mod manifest;
pub mod paths;
pub mod protocol;
pub mod sample;
pub mod segment;
pub mod sherpa;
pub mod speech;
pub mod threads;
pub mod types;
