//! Runs `tinta-engine` as a child process and exchanges JSON lines with it.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The calls that wait for a reply from one engine process. The map is `None` after the process stops.
type Pending = Arc<Mutex<Option<HashMap<u64, Sender<Result<Value, String>>>>>>;
pub type EventHandler = Arc<dyn Fn(Value) + Send + Sync>;

struct Process {
    child: Child,
    stdin: ChildStdin,
    pending: Pending,
}

pub struct Engine {
    path: PathBuf,
    process: Mutex<Option<Process>>,
    next_id: AtomicU64,
    on_event: EventHandler,
    stopped: AtomicBool,
}

/// The engine binary: next to the app executable in the bundle, or the engine's build in development.
/// Windows bundles the engine as `tinta-engine.exe`, so the name follows the platform suffix.
pub fn engine_path() -> PathBuf {
    let file = format!("tinta-engine{}", std::env::consts::EXE_SUFFIX);
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(&file)))
        .filter(|p| p.exists());
    if let Some(path) = bundled {
        return path;
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if cfg!(target_os = "macos") {
        return manifest.join("../../engine/.build/release").join(file);
    }
    // The Rust engine, from `cargo build` on the workspace or on the engine crate alone.
    // The app still opens when no build exists: the engine shows as unavailable.
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let other = if cfg!(debug_assertions) { "release" } else { "debug" };
    let dirs = [
        manifest.join("../../target").join(profile),
        manifest.join("../../target").join(other),
        manifest.join("../../crates/tinta-engine-rs/target").join(profile),
        manifest.join("../../crates/tinta-engine-rs/target").join(other),
    ];
    dirs.iter().map(|d| d.join(&file)).find(|p| p.exists()).unwrap_or_else(|| dirs[0].join(&file))
}

impl Engine {
    pub fn new(on_event: EventHandler) -> Arc<Self> {
        Arc::new(Self {
            path: engine_path(),
            process: Mutex::new(None),
            next_id: AtomicU64::new(1),
            on_event,
            stopped: AtomicBool::new(false),
        })
    }

    fn ensure_running(&self, slot: &mut Option<Process>) -> Result<()> {
        if self.stopped.load(Ordering::SeqCst) {
            bail!("the engine is shut down");
        }
        if let Some(process) = slot.as_mut() {
            if process.child.try_wait()?.is_none() {
                return Ok(());
            }
        }
        let mut command = Command::new(&self.path);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        // A console engine must not open a console window in front of the app on Windows.
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("cannot start {}", self.path.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no engine stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no engine stdout"))?;
        let pending: Pending = Arc::new(Mutex::new(Some(HashMap::new())));
        let reader_pending = pending.clone();
        let on_event = self.on_event.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let Ok(message) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(id) = message.get("id").and_then(Value::as_u64) {
                    let sender = reader_pending.lock().unwrap().as_mut().and_then(|p| p.remove(&id));
                    if let Some(sender) = sender {
                        let result = if message["ok"].as_bool() == Some(true) {
                            Ok(message["result"].clone())
                        } else {
                            Err(message["error"].as_str().unwrap_or("engine error").to_string())
                        };
                        let _ = sender.send(result);
                    }
                } else {
                    on_event(message);
                }
            }
            let waiting = reader_pending.lock().unwrap().take().unwrap_or_default();
            for (_, sender) in waiting {
                let _ = sender.send(Err("the engine stopped".into()));
            }
            on_event(json!({"event": "engine_exit"}));
        });
        *slot = Some(Process { child, stdin, pending });
        Ok(())
    }

    /// Sends one command and waits for its reply.
    pub fn call(&self, cmd: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = channel();
        let mut line = serde_json::to_string(&json!({"id": id, "cmd": cmd, "params": params}))?;
        line.push('\n');
        let pending = {
            let mut slot = self.process.lock().unwrap();
            self.ensure_running(&mut slot)?;
            let process = slot.as_mut().expect("running");
            // The reply can arrive before the write returns, so the call waits in the map before the write.
            match process.pending.lock().unwrap().as_mut() {
                Some(waiting) => waiting.insert(id, sender),
                None => bail!("the engine stopped"),
            };
            let written = process.stdin.write_all(line.as_bytes()).and_then(|()| process.stdin.flush());
            if let Err(error) = written {
                if let Some(waiting) = process.pending.lock().unwrap().as_mut() {
                    waiting.remove(&id);
                }
                return Err(error).context("cannot write to the engine");
            }
            process.pending.clone()
        };
        match receiver.recv_timeout(timeout) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) => bail!(error),
            Err(_) => {
                if let Some(waiting) = pending.lock().unwrap().as_mut() {
                    waiting.remove(&id);
                }
                bail!("the engine did not answer {cmd} in time")
            }
        }
    }

    /// True after `shutdown`.
    pub fn is_shut_down(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Stops the engine. Later calls fail and do not start it again.
    pub fn shutdown(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        let mut slot = self.process.lock().unwrap();
        if let Some(mut process) = slot.take() {
            let _ = process.stdin.write_all(b"{\"id\":0,\"cmd\":\"shutdown\"}\n");
            let _ = process.stdin.flush();
            std::thread::sleep(Duration::from_millis(300));
            let _ = process.child.kill();
        }
    }
}
