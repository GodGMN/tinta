//! Language detection for the `language` field of `finalize`.
//!
//! The Swift engine uses Apple's `NLLanguageRecognizer` (`Speech.swift:219-224`),
//! which is not available off macOS. This uses `whatlang` on the same concatenated
//! transcript and returns the same kind of BCP-47 tag. See the note in
//! `docs/engine-protocol.md` §3.19.

use whatlang::Lang;

/// Detects the dominant language of `text`. Returns `None` when the text is at
/// most 20 characters, matching `Speech.detectLanguage`.
pub fn detect(text: &str) -> Option<String> {
    if text.chars().count() <= 20 {
        return None;
    }
    whatlang::detect(text).map(|info| iso639_1(info.lang()).to_string())
}

/// Maps whatlang's ISO 639-3 language to the ISO 639-1 code Apple returns.
pub fn iso639_1(lang: Lang) -> &'static str {
    match lang {
        Lang::Epo => "eo",
        Lang::Eng => "en",
        Lang::Rus => "ru",
        Lang::Cmn => "zh",
        Lang::Spa => "es",
        Lang::Por => "pt",
        Lang::Ita => "it",
        Lang::Ben => "bn",
        Lang::Fra => "fr",
        Lang::Deu => "de",
        Lang::Ukr => "uk",
        Lang::Kat => "ka",
        Lang::Ara => "ar",
        Lang::Hin => "hi",
        Lang::Jpn => "ja",
        Lang::Heb => "he",
        Lang::Yid => "yi",
        Lang::Pol => "pl",
        Lang::Amh => "am",
        Lang::Jav => "jv",
        Lang::Kor => "ko",
        Lang::Nob => "nb",
        Lang::Dan => "da",
        Lang::Swe => "sv",
        Lang::Fin => "fi",
        Lang::Tur => "tr",
        Lang::Nld => "nl",
        Lang::Hun => "hu",
        Lang::Ces => "cs",
        Lang::Ell => "el",
        Lang::Bul => "bg",
        Lang::Bel => "be",
        Lang::Mar => "mr",
        Lang::Kan => "kn",
        Lang::Ron => "ro",
        Lang::Slv => "sl",
        Lang::Hrv => "hr",
        Lang::Srp => "sr",
        Lang::Mkd => "mk",
        Lang::Lit => "lt",
        Lang::Lav => "lv",
        Lang::Est => "et",
        Lang::Tam => "ta",
        Lang::Vie => "vi",
        Lang::Urd => "ur",
        Lang::Tha => "th",
        Lang::Guj => "gu",
        Lang::Uzb => "uz",
        Lang::Pan => "pa",
        Lang::Aze => "az",
        Lang::Ind => "id",
        Lang::Tel => "te",
        Lang::Pes => "fa",
        Lang::Mal => "ml",
        Lang::Ori => "or",
        Lang::Mya => "my",
        Lang::Nep => "ne",
        Lang::Sin => "si",
        Lang::Khm => "km",
        Lang::Tuk => "tk",
        Lang::Aka => "ak",
        Lang::Zul => "zu",
        Lang::Sna => "sn",
        Lang::Afr => "af",
        Lang::Lat => "la",
        Lang::Slk => "sk",
        Lang::Cat => "ca",
        Lang::Tgl => "tl",
        Lang::Hye => "hy",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_has_no_language() {
        assert_eq!(detect("hello"), None);
        assert_eq!(detect("12345678901234567890"), None);
    }

    #[test]
    fn english_text_returns_en() {
        let text = "This is a meeting about the quarterly planning and the budget.";
        assert_eq!(detect(text).as_deref(), Some("en"));
    }

    #[test]
    fn spanish_text_returns_es() {
        let text =
            "Esta es una reunión sobre la planificación del trimestre y el presupuesto anual.";
        assert_eq!(detect(text).as_deref(), Some("es"));
    }

    #[test]
    fn every_language_maps_to_a_two_letter_code() {
        for lang in Lang::all() {
            let code = iso639_1(*lang);
            assert_eq!(code.len(), 2, "{} -> {code}", lang.code());
        }
    }
}
