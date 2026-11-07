//! The final-pass value types, with the JSON shapes the app parses.
//!
//! `crates/tinta-core/src/transcript.rs` deserializes a word as
//! `{"w": string, "s": number, "e": number}` and a diarization segment as
//! `{"speaker": string, "s": number, "e": number}`.

use serde_json::{json, Value};

/// One transcribed word, in seconds on the meeting clock.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub w: String,
    pub s: f64,
    pub e: f64,
}

impl Word {
    pub fn json(&self) -> Value {
        json!({ "w": self.w, "s": self.s, "e": self.e })
    }
}

/// One diarized speaker region. `speaker` is an opaque label such as `"S1"`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
    pub speaker: String,
    pub s: f64,
    pub e: f64,
}

impl SpeakerSegment {
    pub fn json(&self) -> Value {
        json!({ "speaker": self.speaker, "s": self.s, "e": self.e })
    }
}

/// Serializes words for a `TrackResult`.
pub fn words_json(words: &[Word]) -> Value {
    Value::Array(words.iter().map(Word::json).collect())
}

/// Serializes diarization segments for a `TrackResult`.
pub fn segments_json(segments: &[SpeakerSegment]) -> Value {
    Value::Array(segments.iter().map(SpeakerSegment::json).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_json_has_the_consumed_fields() {
        let word = Word {
            w: "hello".to_string(),
            s: 1.5,
            e: 1.9,
        };
        assert_eq!(word.json(), json!({ "w": "hello", "s": 1.5, "e": 1.9 }));
    }

    #[test]
    fn segment_json_has_the_consumed_fields() {
        let segment = SpeakerSegment {
            speaker: "S2".to_string(),
            s: 3.0,
            e: 4.25,
        };
        assert_eq!(
            segment.json(),
            json!({ "speaker": "S2", "s": 3.0, "e": 4.25 })
        );
    }
}
