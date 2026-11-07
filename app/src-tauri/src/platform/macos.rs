//! macOS integration: FileVault state, Time Machine exclusion, the Chrome native host
//! manifest, peer checks for the app socket, browsers, the clipboard, notifications, the
//! recording bubble, Mission Control, and the data move from versions before 0.4.0.
//!
//! `platform/other.rs` has a stand-in for each item on the other platforms.

pub mod bubble;
pub mod migrate;
pub mod notify;
pub mod overview;

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

pub const NATIVE_HOST_NAME: &str = "com.usetinta.tinta";
/// The native host name of versions before 0.4.0. An extension of such a version connects with
/// this name until Chrome starts again and loads the new extension files.
const OLD_NATIVE_HOST_NAME: &str = "app.tinta";
pub const EXTENSION_ID: &str = "ajncjfpbmkmiheokjfhfdlhnmfbaofij";

pub fn filevault_on() -> bool {
    Command::new("/usr/bin/fdesetup")
        .arg("isactive")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Excludes the data directory from Time Machine. The exclusion stays with the folder.
pub fn exclude_from_backup(dir: &Path) {
    let _ = Command::new("/usr/bin/tmutil").arg("addexclusion").arg(dir).output();
}

/// The full name of the macOS user. The app reads it once.
pub fn full_name() -> String {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        Command::new("/usr/bin/id")
            .arg("-F")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Me".into())
    })
    .clone()
}

/// A helper binary next to the app executable, or in the Cargo target directory in development.
pub fn helper_path(name: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(name)))
        .unwrap_or_else(|| PathBuf::from(name))
}

/// Writes the Native Messaging host manifest for Chrome. Chrome starts the host only
/// for the published extension ID.
pub fn install_native_host() -> Result<()> {
    let dir = native_hosts_dir();
    std::fs::create_dir_all(&dir)?;
    write_host_manifest(&dir, NATIVE_HOST_NAME)?;
    // Without the extension folder, no installed extension uses the old name.
    if dir.join(format!("{OLD_NATIVE_HOST_NAME}.json")).exists() {
        if crate::system::extension_dir().exists() {
            write_host_manifest(&dir, OLD_NATIVE_HOST_NAME)?;
        } else {
            remove_old_native_host();
        }
    }
    Ok(())
}

/// Removes the manifest of the old native host name. The extension calls it when it
/// connects with the current name.
pub fn remove_old_native_host() {
    let _ = std::fs::remove_file(native_hosts_dir().join(format!("{OLD_NATIVE_HOST_NAME}.json")));
}

fn native_hosts_dir() -> PathBuf {
    dirs_home().join("Library/Application Support/Google/Chrome/NativeMessagingHosts")
}

fn write_host_manifest(dir: &Path, name: &str) -> Result<PathBuf> {
    let manifest = json!({
        "name": name,
        "description": "Tinta: Google Meet participant names",
        "path": helper_path("tinta-native-host"),
        "type": "stdio",
        "allowed_origins": [format!("chrome-extension://{EXTENSION_ID}/")],
    });
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?).context("cannot write the native host manifest")?;
    Ok(path)
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// The executable path of the process at the other end of a Unix socket.
pub fn peer_path(fd: i32) -> Option<PathBuf> {
    let mut pid: libc::pid_t = 0;
    let mut len = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
    // SOL_LOCAL = 0, LOCAL_PEERPID = 2 on macOS.
    let status = unsafe { libc::getsockopt(fd, 0, 2, &mut pid as *mut _ as *mut libc::c_void, &mut len) };
    if status != 0 || pid <= 0 {
        return None;
    }
    let mut buffer = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let size = unsafe { libc::proc_pidpath(pid, buffer.as_mut_ptr() as *mut libc::c_void, buffer.len() as u32) };
    if size <= 0 {
        return None;
    }
    buffer.truncate(size as usize);
    String::from_utf8(buffer).ok().map(PathBuf::from)
}

static SIGNATURES: Mutex<Option<HashMap<PathBuf, bool>>> = Mutex::new(None);

/// Accepts a peer only if it is one of the helper binaries next to this app and its code
/// signature is valid.
pub fn trusted_peer(path: &Path) -> bool {
    let allowed = ["tinta-mcp", "tinta-native-host"].iter().any(|name| {
        let expected = helper_path(name);
        match (expected.canonicalize(), path.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    });
    if !allowed {
        return false;
    }
    let mut cache = SIGNATURES.lock().unwrap();
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(valid) = cache.get(path) {
        return *valid;
    }
    let valid = Command::new("/usr/bin/codesign")
        .args(["--verify", "--strict"])
        .arg(path)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    cache.insert(path.to_path_buf(), valid);
    valid
}

/// The value of the `call_browser` setting that selects the default browser of macOS.
pub const DEFAULT_BROWSER: &str = "default";

/// A browser that can open call links.
#[derive(serde::Serialize)]
pub struct Browser {
    /// The bundle ID.
    pub id: String,
    pub name: String,
}

fn browser_at(url: &objc2_foundation::NSURL) -> Option<Browser> {
    let id = objc2_foundation::NSBundle::bundleWithURL(url)?.bundleIdentifier()?.to_string();
    let path = PathBuf::from(url.path()?.to_string());
    let name = path.file_stem()?.to_string_lossy().into_owned();
    Some(Browser { id, name })
}

fn https_url() -> objc2::rc::Retained<objc2_foundation::NSURL> {
    objc2_foundation::NSURL::URLWithString(&objc2_foundation::NSString::from_str("https://")).expect("a valid URL")
}

/// The default browser of macOS.
pub fn default_browser() -> Option<Browser> {
    let workspace = objc2_app_kit::NSWorkspace::sharedWorkspace();
    let url = workspace.URLForApplicationToOpenURL(&https_url())?;
    browser_at(&url)
}

/// The apps on this Mac that open web links, sorted by name.
pub fn browsers() -> Vec<Browser> {
    let workspace = objc2_app_kit::NSWorkspace::sharedWorkspace();
    let mut browsers: Vec<Browser> = Vec::new();
    for url in workspace.URLsForApplicationsToOpenURL(&https_url()).iter() {
        if let Some(browser) = browser_at(&url).filter(|b| !browsers.iter().any(|o| o.id == b.id)) {
            browsers.push(browser);
        }
    }
    browsers.sort_by_key(|b| b.name.to_lowercase());
    browsers
}

/// Opens a call link and returns the recording source for the call. Zoom and Microsoft Teams calls open in their
/// desktop apps when the apps are installed. Other calls open in `browser`, a bundle ID or [`DEFAULT_BROWSER`].
/// A browser that is not installed now gives the default browser. A browser call records all system audio,
/// because some browsers play the call audio in other processes.
pub fn open_call(link: &tinta_core::calendar::Link, browser: &str) -> String {
    let open = |args: &[&str]| Command::new("/usr/bin/open").args(args).status().map(|s| s.success()).unwrap_or(false);
    if let (Some(url), Some(bundle)) = (link.app_url(), link.app_bundle()) {
        if open(&["-b", bundle, &url]) {
            return bundle.to_string();
        }
    }
    if browser == DEFAULT_BROWSER || !open(&["-b", browser, &link.url]) {
        let _ = open(&[&link.url]);
    }
    "all".to_string()
}

/// Opens a link with its default handler.
pub fn open_url(url: &str) -> Result<()> {
    let status = Command::new("/usr/bin/open").arg(url).status()?;
    if !status.success() {
        bail!("cannot open {url}");
    }
    Ok(())
}

/// Opens a folder in Finder.
pub fn open_path(path: &Path) -> Result<()> {
    Command::new("/usr/bin/open").arg(path).spawn()?;
    Ok(())
}

/// Shows a file or a folder in Finder.
pub fn reveal(path: &Path) -> Result<()> {
    Command::new("/usr/bin/open").arg("-R").arg(path).spawn()?;
    Ok(())
}

/// Opens a pane of System Settings.
pub fn open_privacy_settings(pane: &str) -> Result<()> {
    let url = match pane {
        "accessibility" => "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        "microphone" => "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
        "calendars" => "x-apple.systempreferences:com.apple.preference.security?Privacy_Calendars",
        "internet_accounts" => "x-apple.systempreferences:com.apple.Internet-Accounts-Settings.extension",
        other => bail!("unknown settings pane {other}"),
    };
    open_url(url)
}

/// Puts a file on the clipboard, so the user can paste the file in Finder, Mail, or a chat app.
pub fn copy_file(path: &Path) -> Result<()> {
    use objc2::runtime::ProtocolObject;
    use objc2_app_kit::{NSPasteboard, NSPasteboardWriting};
    use objc2_foundation::{NSArray, NSString, NSURL};
    if !path.is_file() {
        anyhow::bail!("{} does not exist", path.display());
    }
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let objects: [objc2::rc::Retained<ProtocolObject<dyn NSPasteboardWriting>>; 1] = [ProtocolObject::from_retained(url)];
    let board = NSPasteboard::generalPasteboard();
    board.clearContents();
    if !board.writeObjects(&NSArray::from_retained_slice(&objects)) {
        anyhow::bail!("Tinta cannot put the file on the clipboard");
    }
    Ok(())
}
