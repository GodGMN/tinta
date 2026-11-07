//! The Windows and Linux engine for Tinta.
//!
//! This crate is the second implementation of the engine protocol in
//! `docs/engine-protocol.md`. It runs as `tinta-engine`, talks JSON lines on
//! stdin/stdout and logs to stderr with the `[tinta-engine]` prefix.
//!
//! The skeleton implements the process lifecycle, `ping`, `models_status`,
//! `install_models`, `permissions`, `list_sources` and the documented degraded
//! replies for the macOS-only commands. Recording, transcription and
//! diarization are later tasks; their commands fail with an `unsupported` error
//! and every capability is false.

pub mod engine;
pub mod install;
pub mod manifest;
pub mod paths;
pub mod protocol;
pub mod sherpa;
pub mod threads;
