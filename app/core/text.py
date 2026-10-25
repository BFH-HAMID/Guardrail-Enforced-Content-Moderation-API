"""Text hygiene: normalization, de-obfuscation and payload sniffing.

Two distinct products are produced from one input, and keeping them separate is what
makes span-level masking trustworthy:

``sanitize_text()``
    Canonical text that is what the rest of the pipeline *moderates and returns*.
    It removes control/zero-width/bidi characters and applies NFKC normalisation.
    Because NFKC can change length, we never try to map offsets back to the raw
    input - all detector spans are expressed in **sanitized** coordinates.

``build_detection_view()``
    A *same-length* view of the sanitized text with leetspeak/homoglyph
    substitutions folded onto ASCII. Same length means every span found here is
    directly usable on the sanitized text, so Stage 2 can mask or rewrite the real
    characters without any offset bookkeeping.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------- control chars

#: characters that survive sanitisation (they are meaningful structure, not noise)
_ALLOWED_CONTROL = {"\n", "\r", "\t"}

_INVISIBLE_CHARS = (
    "\u200b"  # zero width space
    "\u200c\u200d"  # ZWNJ / ZWJ
    "\u2060"  # word joiner
    "\ufeff"  # BOM
    "\u00ad"  # soft hyphen
    "\u2062"  # invisible times
)

_BIDI_CONTROLS = (
    "\u202a\u202b\u202c\u202d\u202e"  # embedding / override controls
    "\u2066\u2067\u2068\u2069"  # isolates
)

#: C0/C1 control characters (excluding the allowed set) - a classic tokeniser smuggle.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# ------------------------------------------------------------------- leet maps

_LET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
    "(": "c",
    "/": "v",
}

#: Unicode look-alikes commonly used to dodge keyword filters (1:1 on purpose).
_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0443": "y",  # у
    "\u0445": "x",  # х
    "\u043a": "k",  # к
    "\u043c": "m",  # м
    "\u0442": "t",  # т
    "\u0456": "i",  # І
    "\u04bb": "h",  # һ
    "\u0501": "q",  # ԁ
    "\u0391": "a",  # Greek Α
    "\u0392": "b",  # Β
    "\u0395": "e",  # Ε
    "\u0397": "h",  # Η
    "\u0399": "i",  # Ι
    "\u039a": "k",  # Κ
    "\u039c": "m",  # Μ
    "\u039d": "n",  # Ν
    "\u039f": "o",  # Ο
    "\u03a1": "p",  # Ρ
    "\u03a4": "t",  # Τ
    "\u03a5": "u",  # Υ
    "\u03a7": "x",  # Χ
    "\u03b1": "a",  # α
    "\u03bf": "o",  # ο
    "\u03c5": "u",  # υ
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\uff01": "!",  # fullwidth !
    "\u2024": ".",
}

# Upper-case variants of the look-alikes above (detection view is case-folded, but
# authors who type "IGNORE PREVIOUS" in caps often also use caps homoglyphs).
for _cy, _lat in list(_HOMOGLYPH_MAP.items()):
    if _cy.isalpha() and _lat.isalpha():
        _HOMOGLYPH_MAP.setdefault(_cy.upper(), _lat)

_LEET_MAP = {**_LET_MAP, **{k.upper(): v for k, v in _LET_MAP.items() if k.isalpha()}}

_COMBINING_MARK_RE = re.compile(r"[\u0300-\u036f\u1ab0-\u1aff\u20d0-\u20ff]")
_EMOJI_RE = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "]+"
)
_REPEAT_RE = re.compile(r"(.)\1{3,}")
# 3+ spaces collapse to a *double* space: the double space is our marker for
# "these letters were spaced apart deliberately", and dropping it would make
# "i g n o r e  p r e v i o u s" indistinguishable from a normal sentence.
_WS_RE = re.compile(r"[ \t]{3,}")
_LETTER_SPACING_RE = re.compile(r"(?:(?:[A-Za-z0-9])[\s._\-]){2,}[A-Za-z0-9]")

# --------------------------------------------------------------------- results


@dataclass(slots=True)
class SanitizedText:
    """Sanitized text plus the audit trail of what was removed and why."""

    raw: str
    text: str
    removed_control_chars: int = 0
    removed_invisible_chars: int = 0
    removed_bidi_controls: int = 0
    normalized: bool = False
    collapsed_repeats: bool = False
    truncated: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.text)

    @property
    def was_modified(self) -> bool:
        return self.text != self.raw

    @property
    def obfuscation_score(self) -> float:
        """0..1 heuristic: how much of the input looks deliberately mangled."""
        removed = self.removed_control_chars + self.removed_invisible_chars + self.removed_bidi_controls
        base = max(len(self.raw), 1)
        score = min(1.0, (removed * 6) / base)
        if "bidi_control" in self.flags:
            score = max(score, 0.85)
        if "repeat_collapse" in self.flags:
            score = max(score, 0.45)
        if "letter_spacing" in self.flags:
            score = max(score, 0.5)
        if "zero_width" in self.flags:
            score = max(score, 0.6)
        return round(score, 4)


@dataclass(slots=True)
class PayloadSignals:
    """Encoded / structural payloads that a naive filter would wave through."""

    base64_spans: list[tuple[int, int, str]] = field(default_factory=list)
    #: Decoded base64/hex that reads as natural language but is *not* an instruction
    #: override. Kept separately because moderation has to look inside the envelope: a
    #: base64-wrapped CSAM solicitation or doxxing dump is the same request once decoded,
    #: and "we only inspect payloads that look like prompt overrides" is an evasion
    #: invitation. The stage that appends these to the detection view owns that decision.
    decoded_language_spans: list[tuple[int, int, str]] = field(default_factory=list)
    hex_spans: list[tuple[int, int]] = field(default_factory=list)
    url_encoded: bool = False
    rot13_suspect: bool = False
    unicode_escape: bool = False
    prompt_like_json: bool = False
    max_entropy: float = 0.0
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "base64_spans": len(self.base64_spans),
            "base64_decoded_prompts": [span[2][:120] for span in self.base64_spans],
        "decoded_language_spans": len(self.decoded_language_spans),
        "decoded_previews": [s[2][:120] for s in self.decoded_language_spans],
            "hex_spans": len(self.hex_spans),
            "url_encoded": self.url_encoded,
            "rot13_suspect": self.rot13_suspect,
            "unicode_escape": self.unicode_escape,
            "prompt_like_json": self.prompt_like_json,
            "max_entropy": round(self.max_entropy, 3),
            "score": round(self.score, 4),
        }


# ------------------------------------------------------------------ primitives


#: --- Indic-aware regex boundaries ---------------------------------------------
#: Python's ``\w`` is Unicode-aware, but it classifies a Bengali/Assamese vowel sign
#: (``ো`` U+09CB, ``া`` U+09BE, ``ি`` ...) as *non-word*. A pattern like ``\\bবোমা\\b`` therefore
#: asserts a boundary at a position where both neighbours are "non-word" and never matches - the
#: word is intact, the assertion is simply meaningless. Every bilingual policy author hits this
#: and concludes the detector is broken, so the boundary is stripped where it cannot be trusted
#: instead of relying on everyone to remember it.
_UNTRUSTABLE_PRECEDERS = (")", "]", "*", "+", "?")


def indic_safe_pattern(pattern: str) -> str:
    """Return *pattern* with ``\\b`` assertions that Bengali orthography defeats removed.

    Only patterns containing non-ASCII are touched, so English rules keep their exact
    word-boundary semantics. A boundary is dropped when it follows a group, a class or a
    quantifier (whose last character may be a combining mark) or a non-ASCII character itself.
    """
    if all(ord(ch) < 128 for ch in pattern):
        return pattern
    out: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        if (
            pattern[index : index + 2] == "\\b"
            and index > 0
            and (pattern[index - 1] in _UNTRUSTABLE_PRECEDERS or ord(pattern[index - 1]) > 127)
        ):
            index += 2
            continue
        out.append(pattern[index])
        index += 1
    return "".join(out)


def compile_pattern(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """``re.compile`` with :func:`indic_safe_pattern` applied."""
    return re.compile(indic_safe_pattern(pattern), flags)


def _char_is_allowed(ch: str) -> bool:
    if ch in _ALLOWED_CONTROL:
        return True
    category = unicodedata.category(ch)
    # Cc (control), Cf (format: zero width, bidi), Co/Cn (private/unassigned), Cs (surrogate)
    return category not in {"Cc", "Cf", "Co", "Cs", "Cn"}


#: A word written one letter at a time - "s h o u l d". Four letters minimum, so ordinary
#: acronym/initial usage ("F B I", "J R") is left alone.
#: A single letter with at most one trailing piece of punctuation - one link of "s h o u l d".
_LETTER_TOKEN_RE = re.compile(r"[A-Za-z][,.!?;:\"\)\]]{0,1}")
#: How many consecutive isolated letters make it look deliberate rather than a list.
_MIN_SPACED_LETTERS = 3


def _spaced_letter_runs(text: str) -> list[tuple[int, int]]:
    """Character offsets ``[start, end)`` of runs of isolated letters.

    Runs are read off the *text*, not off ``split()``: the gap between tokens is the signal that
    separated one smuggled word from the next ("s h o u l d   k i l l"), and a tokenizer that
    throws whitespace away fuses the two into "shouldkill". A run continues across exactly one
    space between bare letters; it ends at a wider gap, at punctuation, or at the end of text,
    because "Q: a b c" is a label plus a list rather than a hidden word.
    """
    runs: list[tuple[int, int]] = []
    length = len(text)
    position = 0
    while position < length:
        if text[position].isspace() or (position and not text[position - 1].isspace()):
            position += 1
            continue
        head = _LETTER_TOKEN_RE.match(text, position)
        if head is None or (position + 1 < length and text[position + 1].isalnum()):
            position += 1
            continue
        cursor = head.end()
        count = 1
        if cursor - position == 1:  # the head token is a bare letter
            while True:
                if cursor >= length or text[cursor] not in " \t":
                    break
                nxt = cursor + 1
                if nxt >= length or text[nxt] in " \t":
                    break  # wider gap: the evader's word separator
                token = _LETTER_TOKEN_RE.match(text, nxt)
                if token is None or (nxt + 1 < length and text[nxt + 1].isalnum()):
                    break
                cursor = token.end()
                count += 1
                if cursor - nxt != 1:
                    break  # this token carries punctuation: last one of the run
        if count >= _MIN_SPACED_LETTERS:
            runs.append((position, cursor))
        position = max(cursor, position + 1)
    return runs


def looks_letter_spaced(text: str) -> bool:
    """True when the text hides a word in single letters (``s h o u l d``)."""
    return bool(_spaced_letter_runs(text))


def _despace_letters(text: str) -> str:
    """Rejoin letter-spaced words in place, keeping the rest of the text untouched.

    Must run *before* whitespace is flattened: once every gap is a single space there is
    nothing left to tell "should kill" from "s h o u l d k i l l", and padding the word gaps is
    exactly what evaders started doing once filters learned to squeeze whitespace.
    """
    runs = _spaced_letter_runs(text)
    if not runs:
        return text
    out: list[str] = []
    cursor = 0
    for start, end in runs:
        out.append(text[cursor:start])
        letters = [ch for ch in text[start:end] if ch.isalpha()]
        tail = ""
        for ch in reversed(text[start:end]):
            if ch.isspace():
                continue
            tail = ch if not ch.isalnum() else ""
            break
        out.append("".join(letters) + tail)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def sanitize_text(
    value: str,
    *,
    max_chars: int | None = None,
    normalize_unicode: bool = True,
    strip_control_chars: bool = True,
    collapse_repeats: bool = True,
) -> SanitizedText:
    """Produce the canonical text that the pipeline moderates and returns."""
    raw = value if isinstance(value, str) else str(value)
    result = SanitizedText(raw=raw, text=raw)

    if normalize_unicode:
        # NFKC folds fullwidth/compatibility forms (ｈｅｌｌｏ -> hello) which is both
        # a legitimate normalisation and an anti-evasion measure.
        normalized = unicodedata.normalize("NFKC", result.text)
        # NFKC turns NBSP etc. into spaces, and leaves combining marks behind.
        normalized = _COMBINING_MARK_RE.sub("", normalized)
        if normalized != result.text:
            result.normalized = True
            result.text = normalized

    if strip_control_chars:
        kept: list[str] = []
        for ch in result.text:
            if ch in _INVISIBLE_CHARS:
                result.removed_invisible_chars += 1
                continue
            if ch in _BIDI_CONTROLS:
                result.removed_bidi_controls += 1
                continue
            if _CONTROL_RE.match(ch) or not _char_is_allowed(ch):
                result.removed_control_chars += 1
                continue
            kept.append(ch)
        result.text = "".join(kept)

    if looks_letter_spaced(result.text):
        # Order matters: this has to happen while the 2-3 space word gaps still exist, and
        # before the generic whitespace flatten below.
        despaced = _despace_letters(result.text)
        if despaced != result.text:
            result.flags.append("letter_spacing")
            result.text = despaced

    if collapse_repeats:
        collapsed = _REPEAT_RE.sub(lambda m: m.group(1) * 3, result.text)
        if collapsed != result.text:
            result.collapsed_repeats = True
            result.flags.append("repeat_collapse")

    # \r is not useful for us; \n\n\n\n\n is just noise.
    result.text = result.text.replace("\r\n", "\n").replace("\r", "\n")
    result.text = _WS_RE.sub(" ", result.text)
    result.text = "\n".join(line.rstrip() for line in result.text.split("\n"))
    result.text = result.text.strip()

    if result.removed_invisible_chars:
        result.flags.append("zero_width")
    if result.removed_bidi_controls:
        result.flags.append("bidi_control")
    if max_chars is not None and len(result.text) > max_chars:
        result.text = result.text[:max_chars]
        result.truncated = True
        result.flags.append("truncated")
    return result


def build_detection_view(text: str) -> str:
    """Same-length ASCII-folded view of *text* with leetspeak/homoglyphs resolved.

    Length is preserved because every substitution is 1 char -> 1 char. That is the
    whole point: spans located in this view can be applied to the sanitized text
    directly, so Stage 2 can mask/rewrite real characters with no offset math.
    """
    if not text:
        return text

    chars = list(text)
    # Pass 1: homoglyphs and diacritic folding (always 1:1).
    for index, ch in enumerate(chars):
        if ch in _HOMOGLYPH_MAP:
            chars[index] = _HOMOGLYPH_MAP[ch]
            continue
        if not ch.isascii():
            folded = unicodedata.normalize("NFKD", ch)
            stripped = "".join(c for c in folded if not unicodedata.combining(c))
            if len(stripped) == 1:
                chars[index] = stripped.lower()
            # else: keep verbatim (Bangna/CJK/...), language rules match on these.
    # Pass 2: lowercase ASCII so the lexicon/regex rules only need lower-case forms.
    chars = [ch.lower() if ch.isascii() else ch for ch in chars]
    # Pass 3: leet digits/symbols, but only inside word-like runs so that real
    # numbers ("call 5551234", "score: 4") are not silently mangled.
    joined = "".join(chars)
    for index, ch in enumerate(chars):
        if ch in _LEET_MAP and _adjacent_to_letters(joined, index):
            chars[index] = _LEET_MAP[ch]
    return "".join(chars)


def _adjacent_to_letters(text: str, index: int) -> bool:
    prev = text[index - 1] if index > 0 else ""
    nxt = text[index + 1] if index + 1 < len(text) else ""
    return bool((prev and prev.isalpha()) or (nxt and nxt.isalpha()))


_SINGLE_CHAR_RUN_RE = re.compile(r"(?:\b\w[ ._\-]){2,}\w\b")


def collapse_letter_spacing(text: str) -> str:
    """Undo per-letter spacing so keyword rules can match again.

    ``i g n o r e  p r e v i o u s`` -> ``ignore previous``. Word boundaries survive
    because :func:`sanitize_text` keeps runs of 3+ spaces as a double space, and
    because a multi-character token is never merged with its neighbours.

    Only ever used for *scoring*: the result has different offsets than the input, so
    no span/masking operation may be derived from it.
    """

    def _join(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if "  " in chunk:  # explicit word boundary inside the run
            parts = [p for p in chunk.split("  ") if p.strip()]
            return "  ".join(re.sub(r"[ ._\-]", "", p) for p in parts)
        return re.sub(r"[ ._\-]", "", chunk)

    return _SINGLE_CHAR_RUN_RE.sub(_join, text)


# ------------------------------------------------------------------- payloads

_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
# Deliberately *separator-aware*. The loose version matched any long run of hex-ish
# characters, so a JWT, a git SHA, a UUID or a session token - i.e. the ordinary debris of
# developer traffic - scored as an "encoded payload" and got blocked. Real hex smuggling has
# byte separators or an 0x prefix, so require one of those shapes.
_HEX_RE = re.compile(r"(?<![0-9a-fA-F])(?:0x[0-9a-fA-F]{20,}|(?:[0-9a-fA-F]{2}[ ,]){8,})")
_URL_ESCAPE_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){4,}")
_UNICODE_ESCAPE_RE = re.compile(r"(?:\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}){3,}")
_DATA_URI_RE = re.compile(r"data:[^;,\s]{1,40};base64,", re.IGNORECASE)


def shannon_entropy(chunk: str) -> float:
    if not chunk:
        return 0.0
    counts: dict[str, int] = {}
    for ch in chunk:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(chunk)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _looks_like_prompt(decoded: str) -> bool:
    lowered = decoded.lower()
    needles = ("system", "instruction", "prompt", "ignore", "you are", "assistant", "api_key", "token")
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\n\t")
    if not decoded or printable / max(len(decoded), 1) < 0.85:
        return False
    return any(needle in lowered for needle in needles)


def _looks_like_natural_language(decoded: str) -> bool:
    """Decoded blob that reads like prose, so it can be handed to the detectors.

    Deliberately permissive: the point is not to *classify* the decoded text, only to stop
    throwing away human-readable content while the caller's own text gets screened. Binary
    payloads (images, protobufs, gzipped data) fail the printable/letter checks and are skipped,
    so a JWT or an inline PNG does not become a fake "document" in the detection view.
    """
    if not decoded or len(decoded) < 20:
        return False
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in " \n\t")
    if printable / len(decoded) < 0.9:
        return False
    letters = sum(1 for ch in decoded if ch.isalpha())
    spaces = decoded.count(" ")
    if letters / len(decoded) < 0.55 or spaces < 3:
        return False
    return True


def detect_payloads(text: str, detection_view: str | None = None) -> PayloadSignals:
    """Find encoded payloads (base64/hex/URL/rot13) that hide instructions."""
    signals = PayloadSignals()
    for match in _BASE64_RE.finditer(text):
        chunk = match.group(0)
        try:
            decoded = base64.b64decode(chunk + "=" * (-len(chunk) % 4), validate=False).decode("utf-8", "ignore")
        except (binascii.Error, ValueError):
            continue
        decoded = decoded.strip()
        if decoded and _looks_like_natural_language(decoded):
            signals.decoded_language_spans.append((match.start(), match.end(), decoded))
            signals.max_entropy = max(signals.max_entropy, shannon_entropy(chunk))
        if _looks_like_prompt(decoded):
            signals.base64_spans.append((match.start(), match.end(), decoded))
            signals.max_entropy = max(signals.max_entropy, shannon_entropy(chunk))
    if _HEX_RE.search(text):
        signals.hex_spans.append((0, 0))
    if _URL_ESCAPE_RE.search(text):
        signals.url_encoded = True
    if _UNICODE_ESCAPE_RE.search(text):
        signals.unicode_escape = True
    if _DATA_URI_RE.search(text):
        signals.url_encoded = True
    candidate = detection_view or text
    rot = "".join(
        chr((ord(c) - ord("a") + 13) % 26 + ord("a")) if "a" <= c <= "z" else chr((ord(c) - ord("A") + 13) % 26 + ord("A")) if "A" <= c <= "Z" else c
        for c in candidate
    )
    if _looks_like_prompt(rot) and re.search(r"(vf|fbe|gung|gb|Fnatnag)", rot, re.IGNORECASE):
        signals.rot13_suspect = True
    if re.search(r"\{\s*\"?(role|content|instructions?)\"\s*:", candidate):
        signals.prompt_like_json = True

    score = 0.0
    if signals.base64_spans:
        score += 0.75
    if signals.hex_spans:
        score += 0.35
    if signals.url_encoded:
        score += 0.30
    if signals.unicode_escape:
        score += 0.30
    if signals.rot13_suspect:
        score += 0.50
    if signals.prompt_like_json:
        score += 0.25
    signals.score = round(min(1.0, score), 4)
    return signals


# ------------------------------------------------------------------ utilities


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, sentence) with offsets valid on *text*."""
    out: list[tuple[int, int, str]] = []
    boundary = re.compile(r"(?<=[.!?\u0964\u0983])\s+|\n+")
    cursor = 0
    for match in boundary.finditer(text):
        chunk = text[cursor : match.start()]
        if chunk.strip():
            out.append((cursor, match.start(), chunk))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        out.append((cursor, len(text), tail))
    return out


def truncate_for_storage(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... truncated {omitted} chars]", True


def redact_middle(text: str, keep: int = 80) -> str:
    """Keep a short head/tail excerpt - used when raw text must not be stored."""
    if len(text) <= keep * 2:
        return text
    return f"{text[:keep]} … [{len(text) - keep * 2} chars withheld] … {text[-keep:]}"


def is_multilingual(text: str) -> dict[str, float]:
    """Rough script mix, so we can pick language-aware rules."""
    if not text:
        return {}
    buckets = {"latin": 0, "bengali": 0, "arabic": 0, "cjk": 0, "other": 0}
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        code = ord(ch)
        if 0x0980 <= code <= 0x09FF:
            buckets["bengali"] += 1
        elif 0x0600 <= code <= 0x06FF:
            buckets["arabic"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:
            buckets["cjk"] += 1
        elif ch.isascii():
            buckets["latin"] += 1
        else:
            buckets["other"] += 1
    total = sum(buckets.values())
    if not total:
        return {}
    return {name: round(count / total, 4) for name, count in buckets.items() if count}
