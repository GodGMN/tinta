//! Linux integration: the Chrome native host manifest, the peer check for the app socket, and
//! the installed browsers.
//!
//! The native host manifest is the same manifest that macOS writes, in the folder of each
//! Chromium browser that this user runs. Chrome starts the helper from the manifest, so the
//! app must write the absolute path of its own `tinta-native-host` binary.

use anyhow::{anyhow, Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};

pub const NATIVE_HOST_NAME: &str = "com.usetinta.tinta";
/// The native host name of versions before 0.4.0, kept in step with `platform/macos.rs`.
const OLD_NATIVE_HOST_NAME: &str = "app.tinta";
pub const EXTENSION_ID: &str = "ajncjfpbmkmiheokjfhfdlhnmfbaofij";

/// A Chromium browser: the executable for call links and the profile folder below the config
/// folder, which is also where the browser looks for native messaging hosts.
struct Chromium {
    id: &'static str,
    name: &'static str,
    profile: &'static str,
}

const CHROMIUM: [Chromium; 3] = [
    Chromium { id: "google-chrome", name: "Google Chrome", profile: "google-chrome" },
    Chromium { id: "chromium", name: "Chromium", profile: "chromium" },
    Chromium { id: "brave-browser", name: "Brave", profile: "BraveSoftware/Brave-Browser" },
];

/// The helper binaries that the app trusts: the ones next to the app executable. The Cargo
/// target directory is the same folder in development, so it needs no separate entry.
pub fn allowed_helpers() -> Vec<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(tinta_core::peer::helpers_in))
        .unwrap_or_default()
}

/// The executable of the process at the other end of a socket, when it runs as this user.
/// `None` closes the connection: the check fails closed on every error.
pub fn peer_path(fd: i32) -> Option<PathBuf> {
    let (uid, exe) = tinta_core::peer::peer_executable(fd)?;
    (uid == tinta_core::peer::current_uid()).then_some(exe)
}

/// Accepts a peer only if its executable is one of the helper binaries of this app.
pub fn trusted_peer(path: &Path) -> bool {
    tinta_core::peer::allowed_exe(path, &allowed_helpers())
}

/// The folder of the user configuration files: `$XDG_CONFIG_HOME`, or `~/.config`.
fn config_home() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"))
}

/// The Chromium browsers that have a profile folder here, sorted by name.
fn browsers_in(config: &Path) -> Vec<super::other::Browser> {
    let mut browsers: Vec<super::other::Browser> = CHROMIUM
        .iter()
        .filter(|browser| config.join(browser.profile).is_dir())
        .map(|browser| super::other::Browser { id: browser.id.to_string(), name: browser.name.to_string() })
        .collect();
    browsers.sort_by_key(|browser| browser.name.to_lowercase());
    browsers
}

/// The browsers that the user can pick for call links.
pub fn browsers() -> Vec<super::other::Browser> {
    browsers_in(&config_home())
}

/// The default browser is the one behind `xdg-open`, which the app does not read, so the list
/// shows no default and `open_call` falls back to the handler of the system.
pub fn default_browser() -> Option<super::other::Browser> {
    None
}

/// The executable that opens a call link in `browser`, when the browser is one of the
/// Chromium browsers of the list. `None` for the default browser.
pub fn browser_command(browser: &str) -> Option<&'static str> {
    CHROMIUM.iter().find(|entry| entry.id == browser).map(|entry| entry.id)
}

/// The folders in which the browsers look for native messaging hosts. A browser writes its
/// profile folder when it first starts, so a fresh install registers with Chrome instead when
/// no browser has run yet.
fn native_host_dirs_in(config: &Path) -> Vec<PathBuf> {
    let mut profiles: Vec<PathBuf> = CHROMIUM
        .iter()
        .map(|browser| config.join(browser.profile))
        .filter(|profile| profile.is_dir())
        .collect();
    if profiles.is_empty() {
        profiles.push(config.join(CHROMIUM[0].profile));
    }
    profiles.into_iter().map(|profile| profile.join("NativeMessagingHosts")).collect()
}

fn native_host_dirs() -> Vec<PathBuf> {
    native_host_dirs_in(&config_home())
}

/// The manifest that Chrome reads. Chrome starts the host only for the published extension ID.
fn host_manifest(name: &str, helper: &Path) -> serde_json::Value {
    json!({
        "name": name,
        "description": "Tinta: Google Meet participant names",
        "path": helper.to_string_lossy(),
        "type": "stdio",
        "allowed_origins": [format!("chrome-extension://{EXTENSION_ID}/")],
    })
}

fn write_host_manifest(dir: &Path, name: &str, helper: &Path) -> Result<PathBuf> {
    let manifest = host_manifest(name, helper);
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?).context("cannot write the native host manifest")?;
    Ok(path)
}

/// Writes the native host manifest in every browser that this user runs. Called on each start,
/// so the absolute helper path follows an update of the app.
pub fn install_native_host() -> Result<PathBuf> {
    let helper = super::other::helper_path("tinta-native-host");
    // Chrome needs an absolute path, and the canonical one keeps working when the install
    // folder is a symbolic link.
    let helper = helper.canonicalize().unwrap_or(helper);
    let mut first: Option<PathBuf> = None;
    let mut last_error = None;
    for dir in native_host_dirs() {
        match install_in(&dir, &helper) {
            Ok(path) => {
                first.get_or_insert(path);
            }
            Err(error) => last_error = Some(error),
        }
    }
    match first {
        Some(path) => Ok(path),
        None => Err(last_error.unwrap_or_else(|| anyhow!("cannot find a browser to register the extension host"))),
    }
}

fn install_in(dir: &Path, helper: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = write_host_manifest(dir, NATIVE_HOST_NAME, helper)?;
    // Without the extension folder, no installed extension uses the old name.
    let old = dir.join(format!("{OLD_NATIVE_HOST_NAME}.json"));
    if old.exists() {
        if crate::system::extension_dir().exists() {
            write_host_manifest(dir, OLD_NATIVE_HOST_NAME, helper)?;
        } else {
            let _ = std::fs::remove_file(old);
        }
    }
    Ok(path)
}

/// Removes the manifest of the old native host name. The extension calls it when it connects
/// with the current name.
pub fn remove_old_native_host() {
    for dir in native_host_dirs() {
        let _ = std::fs::remove_file(dir.join(format!("{OLD_NATIVE_HOST_NAME}.json")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> serde_json::Value {
        host_manifest(NATIVE_HOST_NAME, Path::new("/usr/bin/tinta-native-host"))
    }

    #[test]
    fn manifest_names_the_helper_and_the_extension() {
        let manifest = manifest();
        assert_eq!(manifest["name"], NATIVE_HOST_NAME);
        assert_eq!(manifest["type"], "stdio");
        assert_eq!(manifest["path"], "/usr/bin/tinta-native-host");
        assert_eq!(manifest["allowed_origins"], json!([format!("chrome-extension://{EXTENSION_ID}/")]));
    }

    #[test]
    fn registers_with_the_browsers_that_have_a_profile() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("chromium")).unwrap();
        std::fs::create_dir_all(dir.path().join("BraveSoftware/Brave-Browser")).unwrap();
        let dirs = native_host_dirs_in(dir.path());
        assert_eq!(
            dirs,
            vec![
                dir.path().join("chromium/NativeMessagingHosts"),
                dir.path().join("BraveSoftware/Brave-Browser/NativeMessagingHosts"),
            ]
        );
    }

    #[test]
    fn registers_with_chrome_when_no_browser_ran_yet() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(native_host_dirs_in(dir.path()), vec![dir.path().join("google-chrome/NativeMessagingHosts")]);
    }

    #[test]
    fn lists_only_the_browsers_that_have_a_profile() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("google-chrome")).unwrap();
        let browsers = browsers_in(dir.path());
        assert_eq!(browsers.len(), 1);
        assert_eq!(browsers[0].id, "google-chrome");
        assert_eq!(browsers[0].name, "Google Chrome");
    }

    #[test]
    fn registers_the_same_origin_as_macos() {
        // The extension ID is the one in `extension/EXTENSION_ID.txt`.
        assert_eq!(EXTENSION_ID, "ajncjfpbmkmiheokjfhfdlhnmfbaofij");
        assert_eq!(NATIVE_HOST_NAME, "com.usetinta.tinta");
    }

    #[test]
    fn a_selected_browser_opens_the_call() {
        assert_eq!(browser_command("google-chrome"), Some("google-chrome"));
        assert_eq!(browser_command("brave-browser"), Some("brave-browser"));
        assert_eq!(browser_command("default"), None);
    }

    #[test]
    fn the_helper_paths_are_next_to_the_app() {
        let dir = tempfile::tempdir().unwrap();
        let paths = tinta_core::peer::helpers_in(dir.path());
        assert_eq!(paths, vec![dir.path().join("tinta-native-host"), dir.path().join("tinta-mcp")]);
    }
}
