//! The optional features that the engine reports in `ready` and in the `ping` reply.
//!
//! An engine that does not report the object is the macOS engine from before the field
//! existed, and it has every one of these features. The app therefore reads a missing
//! object, a missing field, or a value that is not a boolean as "supported", and turns a
//! feature off only when the engine says `false`.

use serde::Serialize;
use serde_json::Value;

/// The engine features that the app can turn off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Capabilities {
    /// Record the audio of the whole system.
    pub capture_system_audio: bool,
    /// Record the audio of one app.
    pub capture_app_audio: bool,
    /// Detect calls in desktop meeting apps.
    pub calls_detection: bool,
    /// Read the names, the active speaker, and the mute state from a call window.
    pub call_reader: bool,
    /// Read the calendars, for the next meetings and the reminders.
    pub calendar: bool,
    /// Write a summary of a meeting.
    pub summaries: bool,
    /// Remove the echo of the meeting audio from the microphone.
    pub echo_cancellation: bool,
    /// Import an audio file as a meeting.
    pub import_audio: bool,
}

impl Default for Capabilities {
    /// Every feature on: what an engine that predates the field supports.
    fn default() -> Self {
        Self {
            capture_system_audio: true,
            capture_app_audio: true,
            calls_detection: true,
            call_reader: true,
            calendar: true,
            summaries: true,
            echo_cancellation: true,
            import_audio: true,
        }
    }
}

impl Capabilities {
    /// True when the message names the `capabilities` object at all.
    pub fn reported(message: &Value) -> bool {
        message.get("capabilities").is_some_and(Value::is_object)
    }

    /// Reads the `capabilities` object of a `ready` event or a `ping` result.
    /// A message without the object, or without a field, reports the feature as supported.
    pub fn from_json(message: &Value) -> Self {
        let Some(object) = message.get("capabilities").and_then(Value::as_object) else {
            return Self::default();
        };
        let supports = |key: &str| object.get(key).and_then(Value::as_bool).unwrap_or(true);
        Self {
            capture_system_audio: supports("capture_system_audio"),
            capture_app_audio: supports("capture_app_audio"),
            calls_detection: supports("calls_detection"),
            call_reader: supports("call_reader"),
            calendar: supports("calendar"),
            summaries: supports("summaries"),
            echo_cancellation: supports("echo_cancellation"),
            import_audio: supports("import_audio"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_supports_every_feature() {
        let capabilities = Capabilities::default();
        assert!(capabilities.capture_system_audio);
        assert!(capabilities.capture_app_audio);
        assert!(capabilities.calls_detection);
        assert!(capabilities.call_reader);
        assert!(capabilities.calendar);
        assert!(capabilities.summaries);
        assert!(capabilities.echo_cancellation);
        assert!(capabilities.import_audio);
        assert_eq!(Capabilities::from_json(&json!({"event": "ready"})), capabilities);
        assert_eq!(Capabilities::from_json(&json!({"pong": true})), capabilities);
    }

    #[test]
    fn reads_the_object_from_ready_and_ping() {
        let all_false = json!({"capture_system_audio": false, "capture_app_audio": false, "calls_detection": false,
            "call_reader": false, "calendar": false, "summaries": false, "echo_cancellation": false, "import_audio": false});
        let ready = json!({"event": "ready", "version": "0.9.3", "capabilities": all_false});
        let ping = json!({"pong": true, "capabilities": all_false});
        let none = Capabilities {
            capture_system_audio: false,
            capture_app_audio: false,
            calls_detection: false,
            call_reader: false,
            calendar: false,
            summaries: false,
            echo_cancellation: false,
            import_audio: false,
        };
        assert_eq!(Capabilities::from_json(&ready), none);
        assert_eq!(Capabilities::from_json(&ping), none);
        assert!(Capabilities::reported(&ready));
        assert!(Capabilities::reported(&ping));
        assert!(!Capabilities::reported(&json!({"pong": true})));
    }

    #[test]
    fn missing_fields_and_other_values_are_supported() {
        let partial = json!({"capabilities": {"calendar": false, "summaries": "no"}});
        let capabilities = Capabilities::from_json(&partial);
        assert!(!capabilities.calendar);
        // A value that is not a boolean does not turn the feature off.
        assert!(capabilities.summaries);
        assert!(capabilities.calls_detection);
        assert!(capabilities.import_audio);
    }

    #[test]
    fn serializes_every_field_for_the_window() {
        let text = serde_json::to_string(&Capabilities::default()).unwrap();
        for field in [
            "capture_system_audio",
            "capture_app_audio",
            "calls_detection",
            "call_reader",
            "calendar",
            "summaries",
            "echo_cancellation",
            "import_audio",
        ] {
            assert!(text.contains(&format!("\"{field}\":true")), "{field} missing in {text}");
        }
    }
}
