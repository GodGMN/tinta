//! The parts of the app that differ between macOS and the other platforms.
//!
//! `macos` holds the macOS code, moved as it was. `other` has a stand-in for each item: it
//! refuses what needs macOS, or does nothing. `linux` holds the Linux-specific parts, which
//! `other` re-exports. Keep the files in step, because the rest of the app calls all of them
//! through this module.

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(not(target_os = "macos"))]
mod other;

#[cfg(target_os = "macos")]
pub use macos::*;
#[cfg(not(target_os = "macos"))]
pub use other::*;
