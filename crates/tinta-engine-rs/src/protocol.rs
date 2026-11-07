//! The JSON-lines wire format shared with the Swift engine.
//!
//! See `docs/engine-protocol.md` in the repository. Requests carry a numeric `id`
//! and a `cmd`; replies echo the `id` with `ok` plus `result` or `error`; events
//! carry `event` and no `id`. Every line is written under one lock and flushed
//! before the next line can be written, so a slow reader never sees a partial
//! object.

use serde_json::{Map, Value};
use std::io::{self, Write};
use std::sync::Mutex;

/// A parsed request line.
#[derive(Debug, Clone)]
pub struct Request {
    /// Echoed unchanged in the reply, like the Swift engine. Usually a number.
    pub id: Value,
    pub cmd: String,
    pub params: Map<String, Value>,
}

impl Request {
    /// Parses one line. A line that is not a JSON object with a string `cmd` is
    /// an invalid request: the engine emits an `error` event and sends no reply.
    pub fn parse(line: &str) -> Option<Self> {
        let value: Value = serde_json::from_str(line).ok()?;
        let object = value.as_object()?;
        let cmd = object.get("cmd")?.as_str()?.to_string();
        let id = object.get("id").cloned().unwrap_or(Value::Null);
        let params = object
            .get("params")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Some(Self { id, cmd, params })
    }
}

/// Thread-safe stdout writer. Each object is serialized into one buffer and
/// written together, then flushed.
pub struct Output {
    inner: Mutex<Box<dyn Write + Send>>,
}

impl Output {
    pub fn new<W: Write + Send + 'static>(writer: W) -> Self {
        Self {
            inner: Mutex::new(Box::new(writer)),
        }
    }

    pub fn write_value(&self, value: &Value) -> io::Result<()> {
        let mut line = serde_json::to_vec(value)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        line.push(b'\n');
        let mut writer = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("the output lock is poisoned"))?;
        writer.write_all(&line)?;
        writer.flush()
    }

    pub fn reply(&self, id: &Value, result: Value) -> io::Result<()> {
        self.write_value(&serde_json::json!({
            "id": id,
            "ok": true,
            "result": result,
        }))
    }

    pub fn fail(&self, id: &Value, message: &str) -> io::Result<()> {
        self.write_value(&serde_json::json!({
            "id": id,
            "ok": false,
            "error": message,
        }))
    }

    /// Writes an event. `fields` must be an object; `event` is added to it.
    pub fn event(&self, name: &str, fields: Value) -> io::Result<()> {
        let mut object = match fields {
            Value::Object(object) => object,
            _ => Map::new(),
        };
        object.insert("event".to_string(), Value::String(name.to_string()));
        self.write_value(&Value::Object(object))
    }
}

/// Writes one log line to stderr with the prefix the Swift engine uses.
pub fn log(message: &str) {
    eprintln!("[tinta-engine] {message}");
}
