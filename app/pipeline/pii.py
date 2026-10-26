"""PII detection for Stage 2: Microsoft Presidio when installed, precise regex always.

The regex layer is not a toy fallback - it is tuned for Bangladeshi identifier
formats that stock Presidio does not know:

======================  ==========================================================
Entity                  Shape
======================  ==========================================================
Mobile phone            ``01[3-9]\\d{8}``, ``+880 1X …``, ``01X-XXXXXXX``
Fixed line              ``02-9XXXXXXX``, ``+880-2-9XXXXXXX``
NID number              10 digits, or 13 (old format), optional spaces/dashes
Birth certificate       17 alnum (``19901234560123456``)
Passport                ``[ABCEFGHJKLMN]\\d{7}`` (BD passport style)
Tax / trade licence     ``\\d{9}-\\d{4}`` e-TIN style, trade licence 9 digits
Driving licence         ``\\d{2}-[A-Z]{2,3}-\\d{4}-\\d{4,6}``
Bank account / IBAN     ``BD\\d{2}[A-Z]{4}\\d{12,16}``, 13-19 digit runs
Card number             13-19 digits validated with Luhn
Email / URL / IP        standard
======================  ==========================================================

Presidio runs when importable and adds ``PERSON`` / ``LOCATION`` / ``DATE_OF_BIRTH``
recall; spans from both detectors are merged, and each span records which detector
found it so the trace stays honest about *why* something was masked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

EMAIL_RE = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_RE = re.compile(r"(?<![\w:])(?:[A-Fa-f0-9]{1,4}:){3,7}[A-Fa-f0-9]{1,4}(?![\w:])")
# BD mobile = 01[3-9] + 8 digits (11 total), optionally written with the +880 country
# code and with dashes/space groups. A leading 0 or +88 is required so that a bare
# 10-digit run (invoice no, NID, timestamp) is never mistaken for a phone number.
BD_MOBILE_RE = re.compile(r"(?<![\d+])(?:\+?88[\s-]?)?0[\s-]?1[3-9](?:[\s-]?\d){8}(?![\d])")
INTL_PHONE_RE = re.compile(r"(?<![\d+\-])\+\d{1,3}[-\s]?(?:\(\d{1,4}\)[-\s]?)?\d{2,4}[-\s]?\d{3,4}[-\s]?\d{0,4}\d")
BD_LANDLINE_RE = re.compile(r"(?<!\d)(?:\+?88[-\s]?)?02[-\s]?9?\d{2}[-\s]?\d{3}[-\s]?\d{3}(?!\d)|(?<!\d)0[3-9]\d[-\s]?\d{6}(?!\d)")
NID_RE = re.compile(r"(?<![\d])(?:\d{4}[-\s]?){2}\d{2}(?:[-\s]?\d{3})?(?![\d])")
NID_13_RE = re.compile(r"(?<!\d)\d{2}[-\s]?\d{3}[-\s]?\d{6}(?!\d)")
BIRTH_CERT_RE = re.compile(r"(?<!\d)\d{8}[A-Z]{2}\d{6}(?!\w)|(?<!\d)\d{17}(?!\d)")
#: Bangladesh issues 1-2 letters followed by 6-7 digits; the stricter [A-Z]\d{7} form missed
#: the "BD1234567" series that shows up in the corpus, and a bare letter+digit token is also
#: what an internal product code looks like - hence the keyword gate in the detector.
PASSPORT_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{1,2}\d{6,7}(?![A-Z0-9])")
ETIN_RE = re.compile(r"(?<!\d)\d{9}-\d{4}(?!\d)|(?<!\d)\d{12}(?!\d)")
DRIVING_LIC_RE = re.compile(r"(?<![A-Z0-9-])\d{2}-[A-Z]{2,4}-\d{4}-\d{4,6}(?![A-Z0-9-])")
CARD_RE = re.compile(r"(?<![\d])(?:\d{4}[ -]){3}\d{3,4}(?![\d])|(?<![\d])\d{13,19}(?![\d])")
IBAN_RE = re.compile(r"(?<![A-Z0-9])(?:BD|PK|IN|GB)[0-9]{2}[A-Z]{4}[0-9]{10,18}(?![A-Z0-9])|(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?![A-Z0-9])")
ADDRESS_RE = re.compile(
    # A number before a street word is an address only when something actually separates them,
    # or when the number carries an ordinal suffix: "3 rd avenue" and "3rd Avenue" yes,
    # "3rd grade" no. The unpunctuated version of this rule masked half the corpus.
    r"(?:\d{1,5}[A-Za-z]?(?:(?:st|nd|rd|th)\s+|[ ,#]+\s*)(?:[\w'-]+\s+){0,2}(?:road|rd|lane|ln|street|st|avenue|ave|appartment|apt|flat|house)\b[\w\s,.\-#]{0,45}"
    r"|(?:house|flat|apartment)\s*(?:#|no\.?|number)?\s*\d+[A-Za-z]?\s*,\s*[\w\s\-&'.]{3,40}(?:,\s*(?:dhaka|chattogram|chittagong|sylhet|khulna|rajshahi|barishal|barisal|rangpur|mymensingh)\b)?)"
    r"|(?:\b(?:গুলশান|ধানমন্ডি|উত্তরা|মিরপুর|মোহাম্মদপুর|বনানী|বাড্ডা|ফার্মগেট|চকবাজার|আগারগাঁও)\b[\s,\w\-]{0,30})",
    re.IGNORECASE,
)
DOB_RE = re.compile(r"(?<!\d)(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}(?!\d)")
PASSPORT_KEYWORDS = ("passport", "পাসপোর্ট", "travel document", "book no")
DATE_KEYWORDS = ("dob", "date of birth", "জন্ম তারিখ", "birth date", "born on")
NID_KEYWORDS = ("nid", "national id", "জাতীয় পরিচয়", "স্মার্ট কার্ড", "identity card")

#: Long digit runs are masked as account numbers only when the sentence calls them one: the same
#: 10 digits appear in invoices, order ids, tracking numbers and timestamps, and blanking those
#: out makes business email useless. Bengali spellings included because the corpus is bilingual.
ACCOUNT_KEYWORDS = ("account", "acct", "a/c", "iban", "swift", "beneficiary", "হিসাব", "অ্যাকাউন্ট")

HANDLE_RE = re.compile(r"(?<![\w])@[A-Za-z][A-Za-z0-9_]{4,28}(?![\w])")
US_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

#: Presidio entity type -> our canonical entity type
PRESIDIO_MAP = {
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "CREDIT_CARD": "CREDIT_CARD",
    "US_SSN": "US_SSN",
    "IP_ADDRESS": "IP_ADDRESS",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "MEDICAL_LICENSE": "MEDICAL_LICENSE",
    "US_PASSPORT": "PASSPORT_NUMBER",
    "US_DRIVER_LICENSE": "DRIVERS_LICENSE",
    "NRP": "RESIDENCY_NUMBER",
    "CRYPTO": "CRYPTO_ADDRESS",
    "DOMAIN_NAME": "DOMAIN",
    "URL": "URL",
    "IBAN": "IBAN",
    "AU_ABN": "TAX_NUMBER",
    "UK_NHS": "NHS_NUMBER",
    "IN_PAN": "TAX_NUMBER",
}

DEFAULT_ENTITIES = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "NID_NUMBER",
    "BIRTH_CERTIFICATE",
    "PASSPORT_NUMBER",
    "CREDIT_CARD",
    "IBAN",
    "BANK_ACCOUNT",
    "ADDRESS",
    "IP_ADDRESS",
    "DATE_OF_BIRTH",
    "US_SSN",
    "DRIVERS_LICENSE",
    "TAX_NUMBER",
    "URL",
    "HANDLE",
)


def _near_keyword(text: str, start: int, end: int, keywords: tuple[str, ...], *, window: int = 40, whole_line: bool = False) -> bool:
    """Ambiguous numeric shapes are only trusted when the sentence names them."""
    if whole_line:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        haystack = text[line_start : (line_end if line_end != -1 else len(text))].lower()
        return any(k.lower() in haystack for k in keywords)
    left = max(0, start - window)
    right = min(len(text), end + window)
    haystack = text[left:right].lower()
    return any(k.lower() in haystack for k in keywords)


@dataclass(slots=True)
class PiiSpan:
    """A detected identifier with its provenance."""

    entity_type: str
    start: int
    end: int
    text: str
    score: float = 0.9
    source: str = "regex"
    context: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "text": self.text if len(self.text) <= 6 else self.text[:3] + "…" + self.text[-1],
            "score": round(self.score, 3),
            "source": self.source,
        }

    def overlaps(self, other: PiiSpan) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(slots=True)
class PiiReport:
    spans: list[PiiSpan] = field(default_factory=list)
    presidio_used: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    types: list[str] = field(default_factory=list)

    @property
    def risk(self) -> float:
        """0..1 aggregate exposure used as a scorecard input."""
        if not self.spans:
            return 0.0
        critical = {"CREDIT_CARD", "NID_NUMBER", "US_SSN", "IBAN", "BANK_ACCOUNT", "PASSPORT_NUMBER", "BIRTH_CERTIFICATE"}
        strong = sum(1 for s in self.spans if s.entity_type in critical)
        medium = len(self.spans) - strong
        return round(min(1.0, 0.45 + 0.25 * strong + 0.12 * medium), 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spans": [s.as_dict() for s in self.spans],
            "count": len(self.spans),
            "types": self.types,
            "presidio_used": self.presidio_used,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "risk": self.risk,
        }


# --------------------------------------------------------------------- validators


def luhn_valid(digits: str) -> bool:
    nums = [int(ch) for ch in re.sub(r"\D", "", digits)]
    if len(nums) < 12:
        return False
    checksum = 0
    for index, value in enumerate(reversed(nums)):
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _looks_like_date_run(text: str) -> bool:
    """10-digit NIDs and dates are confusable; drop obviously date-shaped runs."""
    compact = re.sub(r"\D", "", text)
    if len(compact) != 8:
        return False
    year, month, day = compact[:4], compact[4:6], compact[6:8]
    return year.startswith(("19", "20")) and month.isdigit() and 1 <= int(month) <= 12 and 1 <= int(day) <= 31


@lru_cache(maxsize=1)
def _presidio_analyzer() -> Any:
    """Load Presidio once, or return ``None`` when the extra is not installed."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    except ImportError:
        log.info("presidio not installed - using regex PII detector only")
        return None
    except Exception as exc:  # pragma: no cover - broken install
        log.warning("presidio import failed", extra={"error": str(exc)[:200]})
        return None
    try:
        return AnalyzerEngine()
    except Exception as exc:
        log.warning("presidio analyzer init failed", extra={"error": str(exc)[:200]})
        return None


class PiiDetector:
    """Regex + optional Presidio, with span merging and per-detector provenance."""

    def __init__(
        self,
        *,
        use_presidio: bool = True,
        entities: Iterable[str] = DEFAULT_ENTITIES,
        min_score: float = 0.5,
        language: str = "en",
    ) -> None:
        self.use_presidio = use_presidio
        self.entities = tuple(entities)
        self.min_score = min_score
        self.language = language
        self._analyzer: Any = None
        self._analyzer_checked = False

    # ------------------------------------------------------------------ public
    def detect(self, text: str) -> PiiReport:
        import time

        started = time.perf_counter()
        spans: list[PiiSpan] = []
        error: str | None = None

        spans.extend(self._regex_spans(text))

        presidio_used = False
        if self.use_presidio:
            self._presidio_attempted = True
            try:
                analyzer = self._get_analyzer()
                if analyzer is not None:
                    results = analyzer.analyze(text=text, language=self.language, entities=None)
                    presidio_spans = []
                    for result in results or []:
                        entity = PRESIDIO_MAP.get(str(result.entity_type), str(result.entity_type))
                        score = float(getattr(result, "score", 0.85))
                        if score < self.min_score:
                            continue
                        start, end = int(result.start), int(result.end)
                        presidio_spans.append(
                            PiiSpan(
                                entity_type=entity,
                                start=start,
                                end=end,
                                text=text[start:end],
                                score=score,
                                source="presidio",
                                context=self._context(text, start, end),
                            )
                        )
                    spans.extend(self._merge_presidio(presidio_spans, text))
                    presidio_used = True
            except Exception as exc:  # pragma: no cover - presidio runtime issues
                error = f"{exc.__class__.__name__}: {exc}"[:200]
                log.warning("presidio detection failed; regex spans kept", extra={"error": error})

        merged = self._merge(spans)
        types = sorted({span.entity_type for span in merged})
        return PiiReport(
            spans=merged,
            presidio_used=presidio_used,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
            types=types,
        )

    # ---------------------------------------------------------------- internals
    def _get_analyzer(self) -> Any:
        """Load Presidio on first use and remember the answer either way."""
        if not self.use_presidio:
            return None
        if not self._analyzer_checked:
            self._analyzer = _presidio_analyzer()
            self._analyzer_checked = True
        return self._analyzer

    @staticmethod
    def _context(text: str, start: int, end: int, window: int = 24) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right].replace("\n", " ")

    def _merge_presidio(self, presidio_spans: list[PiiSpan], text: str) -> list[PiiSpan]:
        """Keep PERSON/LOCATION only when they add information, never to mask a name alone.

        Names are masked only in combination with a hard identifier (handled by the
        ``pii_personal_name`` policy via ``requires_policy_hits``), so a name span is
        recorded for the trace but never becomes an action by itself.
        """
        return presidio_spans

    def _regex_spans(self, text: str) -> list[PiiSpan]:
        found: list[PiiSpan] = []

        def add(entity: str, match: re.Match[str], score: float, source: str = "regex") -> None:
            start, end = match.start(), match.end()
            found.append(
                PiiSpan(
                    entity_type=entity,
                    start=start,
                    end=end,
                    text=match.group(0),
                    score=score,
                    source=source,
                    context=self._context(text, start, end),
                )
            )

        for match in EMAIL_RE.finditer(text):
            add("EMAIL_ADDRESS", match, 0.99)
        for match in BD_MOBILE_RE.finditer(text):
            add("PHONE_NUMBER", match, 0.97)
        for match in BD_LANDLINE_RE.finditer(text):
            add("PHONE_NUMBER", match, 0.9)
        for match in US_SSN_RE.finditer(text):
            add("US_SSN", match, 0.97)
        for match in IBAN_RE.finditer(text):
            add("IBAN", match, 0.95)
        for match in DRIVING_LIC_RE.finditer(text):
            add("DRIVERS_LICENSE", match, 0.95)
        for match in BIRTH_CERT_RE.finditer(text):
            add("BIRTH_CERTIFICATE", match, 0.9)
        for match in ETIN_RE.finditer(text):
            compact = re.sub(r"\D", "", match.group(0))
            if len(compact) in (9, 12, 13) and not _looks_like_date_run(compact):
                add("TAX_NUMBER", match, 0.75)
        for match in PASSPORT_RE.finditer(text):
            candidate = match.group(0)
            if not _near_keyword(text, match.start(), match.end(), PASSPORT_KEYWORDS):
                continue
            # BD passport leading letters are limited; skip random 8-char tokens that
            # only look like a passport number (product codes, ticket numbers...)
            if candidate[0] in "ABCEFGHJKLMN" and not re.search(r"[AEIOU]{4,}", candidate):
                add("PASSPORT_NUMBER", match, 0.8)
        for match in DOB_RE.finditer(text):
            if _near_keyword(text, match.start(), match.end(), DATE_KEYWORDS):
                add("DATE_OF_BIRTH", match, 0.85)
        for match in IPV4_RE.finditer(text):
            octets = match.group(0).split(".")
            if all(int(part) <= 255 for part in octets) and not match.group(0).startswith(("0.", "127.")):
                add("IP_ADDRESS", match, 0.9)
        for match in IPV6_RE.finditer(text):
            add("IP_ADDRESS", match, 0.75)
        for match in HANDLE_RE.finditer(text):
            add("HANDLE", match, 0.7)
        for match in ADDRESS_RE.finditer(text):
            add("ADDRESS", match, 0.72)
        for match in URL_RE.finditer(text):
            add("URL", match, 0.65)

        # NID: 10-digit or 13-digit numeric runs that are not phones/dates/cards
        # A 10/13-digit run is only an NID when the sentence says so; without that
        # context gate we would mask invoice numbers, order IDs and timestamps.
        nid_context = _near_keyword(text, 0, len(text), NID_KEYWORDS, whole_line=True)
        for match in re.finditer(r"(?<![\d+\-])\d[\d \-]{8,12}(?![\d\-])", text):
            compact = re.sub(r"\D", "", match.group(0))
            if len(compact) == 10 and not BD_MOBILE_RE.fullmatch(match.group(0).strip()):
                if nid_context and not _looks_like_date_run(compact):
                    add("NID_NUMBER", match, 0.85)
        for match in re.finditer(r"(?<![\d+\-])\d[\d \-]{11,15}(?![\d\-])", text):
            compact = re.sub(r"\D", "", match.group(0))
            if len(compact) == 13 and nid_context:
                add("NID_NUMBER", match, 0.9)

        # Bank/reference numbers ("close account 0987654321") - gated on the same kind of
        # context as the NID rule above, and skipped where the NID rule already took the span.
        if _near_keyword(text, 0, len(text), ACCOUNT_KEYWORDS, whole_line=True):
            taken = {span.start for span in found}
            for match in re.finditer(r"(?<![\d+\-])\d[\d \-]{7,12}(?![\d\-])", text):
                compact = re.sub(r"\D", "", match.group(0))
                if match.start() in taken:
                    continue
                if 9 <= len(compact) <= 12 and not _looks_like_date_run(compact):
                    add("BANK_ACCOUNT", match, 0.8)

        # Cards need Luhn; digits-only check avoids "1234 5678 9012 3456" as a phone
        for match in CARD_RE.finditer(text):
            if luhn_valid(match.group(0)) and len(re.sub(r"\D", "", match.group(0))) in range(13, 20):
                add("CREDIT_CARD", match, 0.95)
            elif len(re.sub(r"\D", "", match.group(0))) in (13, 16, 19) and re.search(r"\d[\s-]\d", match.group(0)):
                # 16-digit grouping that failed Luhn is still worth flagging as an identifier
                add("BANK_ACCOUNT", match, 0.6)

        # Public IPs / non-routable guard: drop addresses that are actually versions
        found = [
            span
            for span in found
            if not (span.entity_type == "IP_ADDRESS" and re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", span.text) and self._is_version_like(text, span))
        ]
        return found

    @staticmethod
    def _is_version_like(text: str, span: PiiSpan) -> bool:
        prefix = text[max(0, span.start - 18) : span.start].lower()
        return bool(re.search(r"(?:version|v(?:er)?\.?|release|build|semver|node|python|php|android|ios)\s*[:=]?\s*$", prefix))

    def _merge(self, spans: list[PiiSpan]) -> list[PiiSpan]:
        """Keep the tightest, most confident span per region.

        Sorted by ascending width so precise matches (a phone inside a digit run, an
        email inside a URL) are claimed first; a wider span is then only accepted if
        it does not *contain* an already-kept confident span - that is what stops
        "+880 1812345678" from being re-labelled as a card number.
        """
        ordered = sorted(spans, key=lambda s: (s.end - s.start, -s.score, s.start))
        kept: list[PiiSpan] = []
        for span in ordered:
            contained = [other for other in kept if other.start >= span.start and other.end <= span.end and other is not span]
            if contained and all(other.score >= span.score - 0.1 for other in contained):
                continue
            overlapping = [other for other in kept if span.overlaps(other)]
            if overlapping:
                best = max(overlapping, key=lambda o: o.score)
                if span.score > best.score + 0.05 and (span.start, span.end) != (best.start, best.end):
                    kept.append(span)  # complementary identifier, e.g. NID + DOB
                elif span.score > best.score:
                    kept[kept.index(best)] = span
                continue
            kept.append(span)
        kept.sort(key=lambda s: s.start)
        return kept


def apply_masks(text: str, spans: list[PiiSpan], placeholders: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """Replace each span with its placeholder; returns the masked text + action rows."""
    if not spans:
        return text, []
    actions: list[dict[str, Any]] = []
    out = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        if span.entity_type == "PERSON":
            continue  # names alone are not masked (see pii_personal_name policy rationale)
        placeholder = placeholders.get(span.entity_type)
        if not placeholder:
            continue
        out = out[: span.start] + placeholder + out[span.end :]
        actions.append(
            {
                "entity_type": span.entity_type,
                "action": "mask",
                "start": span.start,
                "end": span.end,
                "original": span.text,
                "replacement": placeholder,
                "confidence": span.score,
                "detector": span.source,
                "context": span.context,
            }
        )
    actions.reverse()
    return out, actions
