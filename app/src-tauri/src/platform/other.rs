//! Stand-ins for the macOS integration on Windows and Linux.
//!
//! Each item either does nothing, because the macOS feature has no counterpart here, or
//! refuses with a clear error, so a caller never believes that it worked. Keep the items in
//! step with `platform/macos.rs`.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The value of the `call_browser` setting that selects the default browser.
pub const DEFAULT_BROWSER: &str = "default";

/// A browser that can open call links.
#[derive(serde::Serialize)]
pub struct Browser {
    pub id: String,
    pub name: String,
}

/// FileVault is a macOS feature. Nothing to check here, so the setup does not warn about it.
pub fn filevault_on() -> bool {
    true
}

/// Time Machine is a macOS feature. Other platforms back up the data folder in their own way.
pub fn exclude_from_backup(_dir: &Path) {}

/// The user name of the current session. The app reads it once.
pub fn full_name() -> String {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Me".into())
    })
    .clone()
}

/// A helper binary next to the app executable, or in the Cargo target directory in development.
pub fn helper_path(name: &str) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(&file)))
        .unwrap_or_else(|| PathBuf::from(file))
}

/// Chrome starts the Meet extension host from a manifest that differs on each platform.
/// Windows and Linux registration comes with the rest of the port, so the app runs without
/// the extension until then.
pub fn install_native_host() -> Result<PathBuf> {
    Ok(PathBuf::new())
}

pub fn remove_old_native_host() {}

/// The platform peer check is not implemented yet. The Windows build has no app socket, and
/// the Linux one rejects every connection until the check exists: fail closed.
#[allow(dead_code)]
pub fn peer_path(_fd: i32) -> Option<PathBuf> {
    None
}

/// Rejects every peer: helper authentication is not implemented on this platform yet.
#[allow(dead_code)]
pub fn trusted_peer(_path: &Path) -> bool {
    false
}

/// The system default browser, when the platform names one.
pub fn default_browser() -> Option<Browser> {
    None
}

/// The browsers that the user can select are a macOS list for now.
pub fn browsers() -> Vec<Browser> {
    Vec::new()
}

/// Opens a call link in the default handler and records all system audio: the app cannot pick
/// a browser without the macOS browser list, and the browser processes play the call audio.
pub fn open_call(link: &tinta_core::calendar::Link, _browser: &str) -> String {
    let _ = open_url(&link.url);
    "all".to_string()
}

/// Opens a link with its default handler.
pub fn open_url(url: &str) -> Result<()> {
    let status = opener().arg(url).status()?;
    if !status.success() {
        bail!("cannot open {url}");
    }
    Ok(())
}

/// Opens a folder or a file with its default handler.
pub fn open_path(path: &Path) -> Result<()> {
    opener().arg(path).spawn()?;
    Ok(())
}

/// Shows a file or a folder in the file manager.
pub fn reveal(path: &Path) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        // Explorer selects the item with `/select,`.
        let argument = format!("/select,{}", path.display());
        Command::new("explorer").arg(argument).spawn()?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        // The file managers of Linux do not select the item, so show the folder that holds it.
        let folder = path.parent().unwrap_or(path);
        Command::new("xdg-open").arg(folder).spawn()?;
    }
    Ok(())
}

/// System settings panes are a macOS feature.
pub fn open_privacy_settings(_pane: &str) -> Result<()> {
    bail!("Tinta cannot open system privacy settings on this platform")
}

/// Copying a file to the clipboard is a macOS feature for now.
pub fn copy_file(_path: &Path) -> Result<()> {
    bail!("Tinta cannot copy a file to the clipboard on this platform")
}

fn opener() -> Command {
    // Explorer opens a folder or a link with the default handler on Windows.
    #[cfg(target_os = "windows")]
    let command = Command::new("explorer");
    #[cfg(not(target_os = "windows"))]
    let command = Command::new("xdg-open");
    command
}

pub mod bubble {
    use tauri::{AppHandle, Manager};

    /// The recording bubble is a macOS window for now.
    pub fn sync(_app: &AppHandle, _recording: bool) {}

    /// Shows the main window, for a click in a reminder.
    pub fn show_main(app: &AppHandle) {
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
        }
    }
}

pub mod migrate {
    /// The data folders that this move looks for exist only on macOS.
    pub fn run(_identifier: &str) {}
}

pub mod notify {
    /// Meeting reminders use the macOS Notification Center. Linux and Windows reminders come
    /// with the rest of the port.
    pub fn init(_on_join: impl Fn(String) + Send + Sync + 'static) {}

    pub fn request_permission() {}

    pub fn remind(_event_id: &str, _title: &str, _body: &str) {}

    pub fn withdraw(_event_id: &str) {}
}

pub mod overview {
    /// Mission Control is a macOS feature.
    pub fn watch() {}
}
