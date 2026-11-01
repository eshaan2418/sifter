//! Text analysis matching Lucene's `EnglishAnalyzer`, the analyzer behind
//! the standard BM25 baselines: Unicode word tokenization, lowercasing,
//! English possessive removal, the Lucene English stop list, and the
//! original Porter (1980) stemmer.

/// Lucene's `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Splits text into analyzed terms.
pub fn analyze(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    analyze_into(text, &mut out);
    out
}

pub fn analyze_into(text: &str, out: &mut Vec<String>) {
    for raw in tokens(text) {
        let mut t = raw.to_lowercase();
        // EnglishPossessiveFilter: strip trailing 's (and the curly-quote forms).
        for suffix in ["'s", "\u{2019}s", "\u{ff07}s"] {
            if t.ends_with(suffix) {
                t.truncate(t.len() - suffix.len());
                break;
            }
        }
        if t.is_empty() || STOP_WORDS.contains(&t.as_str()) {
            continue;
        }
        out.push(stem(&t));
    }
}

/// Word tokens: maximal runs of alphanumerics, keeping internal apostrophes
/// (so possessives can be removed) and splitting on everything else.
fn tokens(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let start = rest.find(|c: char| c.is_alphanumeric())?;
        rest = &rest[start..];
        let mut end = 0;
        let mut chars = rest.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let keep = c.is_alphanumeric()
                || (matches!(c, '\'' | '\u{2019}')
                    && chars.peek().is_some_and(|&(_, n)| n.is_alphanumeric()));
            if !keep {
                break;
            }
            end = i + c.len_utf8();
        }
        let tok = &rest[..end];
        rest = &rest[end..];
        Some(tok)
    })
}

// ---------------------------------------------------------------------------
// Porter stemmer (M. F. Porter, "An algorithm for suffix stripping", 1980),
// following the reference implementation. Non-ASCII words are left as-is.

pub fn stem(word: &str) -> String {
    if word.len() <= 2 || !word.is_ascii() {
        return word.to_string();
    }
    let mut s = Stemmer {
        b: word.as_bytes().to_vec(),
        k: word.len() - 1,
        j: 0,
    };
    s.step1ab();
    if s.k > 0 {
        s.step1c();
        s.step2();
        s.step3();
        s.step4();
        s.step5();
    }
    s.b.truncate(s.k + 1);
    String::from_utf8(s.b).expect("ascii")
}

struct Stemmer {
    b: Vec<u8>,
    k: usize, // index of the last character of the current word
    j: usize, // general offset into the word
}

impl Stemmer {
    fn cons(&self, i: usize) -> bool {
        match self.b[i] {
            b'a' | b'e' | b'i' | b'o' | b'u' => false,
            b'y' => i == 0 || !self.cons(i - 1),
            _ => true,
        }
    }

    /// m(): the number of consonant sequences between 0 and j.
    fn m(&self) -> usize {
        let mut n = 0;
        let mut i = 0;
        loop {
            if i > self.j {
                return n;
            }
            if !self.cons(i) {
                break;
            }
            i += 1;
        }
        i += 1;
        loop {
            loop {
                if i > self.j {
                    return n;
                }
                if self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            n += 1;
            loop {
                if i > self.j {
                    return n;
                }
                if !self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
        }
    }

    fn vowel_in_stem(&self) -> bool {
        (0..=self.j).any(|i| !self.cons(i))
    }

    fn doublec(&self, j: usize) -> bool {
        j >= 1 && self.b[j] == self.b[j - 1] && self.cons(j)
    }

    /// cvc(i): i-2, i-1, i is consonant-vowel-consonant and the last is not w, x or y.
    fn cvc(&self, i: usize) -> bool {
        if i < 2 || !self.cons(i) || self.cons(i - 1) || !self.cons(i - 2) {
            return false;
        }
        !matches!(self.b[i], b'w' | b'x' | b'y')
    }

    fn ends(&mut self, s: &str) -> bool {
        let s = s.as_bytes();
        let len = s.len();
        if len > self.k + 1 || &self.b[self.k + 1 - len..=self.k] != s {
            return false;
        }
        self.j = self.k + 1 - len;
        // j points at the character before the suffix (may underflow for a
        // whole-word match; callers check m() which handles j via wrapping).
        self.j = self.j.wrapping_sub(1);
        true
    }

    fn setto(&mut self, s: &str) {
        let start = self.j.wrapping_add(1);
        self.b.truncate(start);
        self.b.extend_from_slice(s.as_bytes());
        self.k = self.b.len() - 1;
    }

    fn r(&mut self, s: &str) {
        if self.j != usize::MAX && self.m() > 0 {
            self.setto(s);
        }
    }

    fn step1ab(&mut self) {
        if self.b[self.k] == b's' {
            if self.ends("sses") {
                self.k -= 2;
            } else if self.ends("ies") {
                self.setto("i");
            } else if self.k >= 1 && self.b[self.k - 1] != b's' {
                self.k -= 1;
            }
            self.b.truncate(self.k + 1);
        }
        if self.ends("eed") {
            if self.j != usize::MAX && self.m() > 0 {
                self.k -= 1;
                self.b.truncate(self.k + 1);
            }
        } else if (self.ends("ed") || self.ends("ing"))
            && self.j != usize::MAX
            && self.vowel_in_stem()
        {
            self.k = self.j;
            self.b.truncate(self.k + 1);
            if self.ends("at") {
                self.setto("ate");
            } else if self.ends("bl") {
                self.setto("ble");
            } else if self.ends("iz") {
                self.setto("ize");
            } else if self.doublec(self.k) {
                self.k -= 1;
                let ch = self.b[self.k];
                if matches!(ch, b'l' | b's' | b'z') {
                    self.k += 1;
                }
                self.b.truncate(self.k + 1);
            } else {
                self.j = self.k;
                if self.m() == 1 && self.cvc(self.k) {
                    self.j = self.k;
                    self.setto("e");
                }
            }
        }
    }

    fn step1c(&mut self) {
        if self.ends("y") && self.j != usize::MAX && self.vowel_in_stem() {
            self.b[self.k] = b'i';
        }
    }

    fn step2(&mut self) {
        if self.k == 0 {
            return;
        }
        let rules: &[(&str, &str)] = match self.b[self.k - 1] {
            b'a' => &[("ational", "ate"), ("tional", "tion")],
            b'c' => &[("enci", "ence"), ("anci", "ance")],
            b'e' => &[("izer", "ize")],
            b'l' => &[
                ("bli", "ble"),
                ("alli", "al"),
                ("entli", "ent"),
                ("eli", "e"),
                ("ousli", "ous"),
            ],
            b'o' => &[("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            b's' => &[
                ("alism", "al"),
                ("iveness", "ive"),
                ("fulness", "ful"),
                ("ousness", "ous"),
            ],
            b't' => &[("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            b'g' => &[("logi", "log")],
            _ => &[],
        };
        for (suffix, repl) in rules {
            if self.ends(suffix) {
                self.r(repl);
                return;
            }
        }
    }

    fn step3(&mut self) {
        let rules: &[(&str, &str)] = match self.b[self.k] {
            b'e' => &[("icate", "ic"), ("ative", ""), ("alize", "al")],
            b'i' => &[("iciti", "ic")],
            b'l' => &[("ical", "ic"), ("ful", "")],
            b's' => &[("ness", "")],
            _ => &[],
        };
        for (suffix, repl) in rules {
            if self.ends(suffix) {
                self.r(repl);
                return;
            }
        }
    }

    fn step4(&mut self) {
        if self.k == 0 {
            return;
        }
        let suffixes: &[&str] = match self.b[self.k - 1] {
            b'a' => &["al"],
            b'c' => &["ance", "ence"],
            b'e' => &["er"],
            b'i' => &["ic"],
            b'l' => &["able", "ible"],
            b'n' => &["ant", "ement", "ment", "ent"],
            b'o' => &["ion", "ou"],
            b's' => &["ism"],
            b't' => &["ate", "iti"],
            b'u' => &["ous"],
            b'v' => &["ive"],
            b'z' => &["ize"],
            _ => &[],
        };
        for suffix in suffixes {
            if self.ends(suffix) {
                if *suffix == "ion"
                    && !(self.j != usize::MAX && matches!(self.b[self.j], b's' | b't'))
                {
                    continue;
                }
                if self.j != usize::MAX && self.m() > 1 {
                    self.k = self.j;
                    self.b.truncate(self.k + 1);
                }
                return;
            }
        }
    }

    fn step5(&mut self) {
        self.j = self.k;
        if self.b[self.k] == b'e' {
            let a = self.m();
            if a > 1 || (a == 1 && !self.cvc(self.k - 1)) {
                self.k -= 1;
                self.b.truncate(self.k + 1);
            }
        }
        if self.b[self.k] == b'l' && self.doublec(self.k) {
            self.j = self.k;
            if self.m() > 1 {
                self.k -= 1;
                self.b.truncate(self.k + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn porter_reference_examples() {
        // From Porter's paper and the reference vocabulary.
        let cases = [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("bled", "bled"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("tanned", "tan"),
            ("falling", "fall"),
            ("hissing", "hiss"),
            ("fizzed", "fizz"),
            ("failing", "fail"),
            ("filing", "file"),
            ("happy", "happi"),
            ("sky", "sky"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("valenci", "valenc"),
            ("digitizer", "digit"),
            ("conformabli", "conform"),
            ("radicalli", "radic"),
            ("differentli", "differ"),
            ("vileli", "vile"),
            ("analogousli", "analog"),
            ("vietnamization", "vietnam"),
            ("predication", "predic"),
            ("operator", "oper"),
            ("feudalism", "feudal"),
            ("decisiveness", "decis"),
            ("hopefulness", "hope"),
            ("callousness", "callous"),
            ("formaliti", "formal"),
            ("sensitiviti", "sensit"),
            ("sensibiliti", "sensibl"),
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electriciti", "electr"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("irritant", "irrit"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("homologou", "homolog"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("angulariti", "angular"),
            ("homologous", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
            ("probate", "probat"),
            ("rate", "rate"),
            ("cease", "ceas"),
            ("controll", "control"),
            ("roll", "roll"),
            ("generalizations", "gener"),
            ("oscillators", "oscil"),
        ];
        for (w, want) in cases {
            assert_eq!(stem(w), want, "stem({w})");
        }
    }

    #[test]
    fn analyzer_pipeline() {
        assert_eq!(
            analyze("The CAT's toys were running, and it's 3.5x faster!"),
            // "it's" -> "it" (possessive) -> dropped (stop word), as in Lucene.
            // Unlike Lucene's UAX#29 tokenizer, "3.5x" splits at the dot.
            vec!["cat", "toi", "were", "run", "3", "5x", "faster"]
        );
        assert!(analyze("the and of").is_empty());
        assert_eq!(analyze("naïve café"), vec!["naïve", "café"]);
    }
}
