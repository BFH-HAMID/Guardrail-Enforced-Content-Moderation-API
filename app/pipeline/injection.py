"""Stage-1 Layer A: fast, cheap injection/toxicity signals.

Two independent detectors feed the fusion step:

``InjectionHeuristics``
    Pure-Python structural signals - instruction-override phrasing, delimiter escape
    attempts, token smuggling, encoded payloads, multi-turn priming, obfuscation
    budget. Costs microseconds, always available, and is the reason the service
    still catches the classics when the model is unreachable.

``PromptInjectionClassifier``
    Wraps a Hugging Face sequence classifier (default ``protectai/deberta-v3-base-prompt-injection-v2``).
    The import is lazy and the model loads once, in a worker thread, with a hard
    timeout: if ``transformers``/``torch`` are missing or the weights cannot be
    fetched, the detector reports ``available=False`` and the pipeline relies on the
    heuristic layer + judge (which is a *documented* degradation, not a silent one).

Note on overlap with ``config/policies.yaml``: the YAML rules decide *categories and
actions*; this module produces *numbers*. The policy engine consumes those numbers
(``classifier`` / ``obfuscation`` / ``payloads`` clauses), so we never have two
sources of truth about what gets blocked.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.text import PayloadSignals, collapse_letter_spacing, indic_safe_pattern

log = get_logger(__name__)

# --------------------------------------------------------------------- patterns

_OVERRIDE_RE = re.compile(
    r"(?:ignore|disregard|forget|bypass|skip|override|discard|cancel|reset)\b[\s\S]{0,40}"
    r"(?:previous|prior|above|earlier|preceding|initial|original|all|system|safety|core|your)\b[\s\S]{0,30}"
    r"(?:instructions?|prompts?|rules?|directives?|messages?|guidelines?|context|policy|policies|constraints?)",
    re.IGNORECASE,
)
_NEW_INSTRUCTIONS_RE = re.compile(r"(?:new|updated|revised|real|true|actual|secret)\s+instructions?\s*[:\-]", re.IGNORECASE)
_SYSTEM_EXTRACTION_RE = re.compile(
    r"(?:reveal|show|print|repeat|output|dump|leak|display|echo|translate|encode|decode|provide|send|expose|what (?:is|are)|list)\b"
    r"[\s\S]{0,40}(?:system\s+(?:prompt|message|instructions?)|hidden\s+(?:prompt|instructions?)|initial\s+(?:prompt|instructions?)|your\s+(?:prompt|rules|instructions|guidelines)|prompt\s+template|guardrail\s+config)",
    re.IGNORECASE,
)
_PERSONA_OVERRIDE_RE = re.compile(
    r"(?:you\s+are\s+now|from\s+now\s+on\s+you|act\s+as|pretend\s+(?:to\s+be|you\s+are)|role\s*play\s+as|simulate|imagine\s+you\s+(?:are|have\s+no))"
    r"[\s\S]{0,80}(?:dan|unrestricted|unfiltered|jailbroken|without\s+(?:rules|limits|filters|restrictions|guardrails)|no\s+(?:rules|limits|restrictions)|god\s+mode|developer\s+mode|do\s+anything\s+now|anti[-\s]?gpt)",
    re.IGNORECASE,
)
#: Bangla + Hindi injection phrasings (transliterated "system prompt" included,
#: because that is how these attacks are actually written in South-Asian inputs).
_OVERRIDE_NONLATIN_RE = re.compile(
    r"(?:আগের|পূর্ববর্তী|ওপরের|আর কোনো|সব|তোমার|আপনার)[\s\S]{0,16}"
    r"(?:নির্দেশনা?|নির্দেশ|প্রম্পট|রুলস?|নিয়ম(?:াবলী)?|শর্ত(?:াবলী)?|কথন)\s*"
    r"(?:উপেক্ষা|অবহেলা|মানো\s*না|বাদ\s*দাও|ভুলে\s*যাও|কেটে\s*ফেল|লক্ষ\s*করো\s*না)"
    r"|(?:উপেক্ষা|অবহেলা)\s*(?:করো|করুন|করে)?"
    r"|(?:सभी|पिछले|ऊपर\s*वाले)\s+(?:निर्देश|नियम)\s*(?:भूल\s*जाओ|नज़रअंदाज़|छोड़ो)"
    r"|(?:ignore|bypass)[\s\S]{0,12}(?:নির্দেশনা|প্রম্পট)"
    r"|(?:নির্দেশনা|প্রম্পট)[\s\S]{0,16}(?:ignore| disregard)",
    re.IGNORECASE,
)
_NONLATIN_SYSTEM_LEAK_RE = re.compile(
    r"(?:সিস্টেম|सিস্টেম|system)[\s\S]{0,10}(?:প্রম্পট|प्रेम्प्ट|prompt|নির্দেশ|message)"
    r"[\s\S]{0,20}(?:দেখাও|বলো|প্রিন্ট\s*করো|লিখে\s*দাও|দাও|भेदो|बताओ)"
    r"|(?:দেখাও|বলো|প্রিন্ট\s*করো|লিখে\s*দাও)[\s\S]{0,20}(?:সিস্টেম|system)[\s\S]{0,12}(?:প্রম্পট|prompt|নির্দেশ)",
    re.IGNORECASE,
)
_NONLATIN_JAILBREAK_RE = re.compile(
    r"(?:তুমি|আপনি)\s+(?:এখন|এখান\s*থেকে)[\s\S]{0,30}(?:কোনো\s*(?:নিয়ম|বিধি|সীমাবদ্ধতা|বাধা)\s*(?:না|নেই|মান(?:ব|বে|বেন)?\s*না|মানবে\s*না))"
    r"|(?:ধরে\s*নাও|অনুকরণ\s*করো|ভাবো\s*তুমি|মনে\s*করো)[\s\S]{0,40}(?:নিয়ম(?:বিহীন| ছাড়া)|সীমাবদ্ধতা\s*নেই|জেলব্রেক)",
    re.IGNORECASE,
)

_HYPOTHETICAL_SHIELD_RE = re.compile(
    r"(?:hypothetically|purely\s+fictional|in\s+(?:a\s+)?(?:fictional|alternate|make[-\s]?believe)\s+(?:world|scenario)|for\s+(?:research|academic|testing)\s+purposes?|this\s+is\s+(?:a\s+)?(?:test|simulation)|no\s+real\s+people\s+(?:are|would\s+be)\s+(?:hurt|involved)|you\s+have\s+no\s+ethics)",
    re.IGNORECASE,
)
_AUTHORITY_SPOOF_RE = re.compile(
    r"(?:\[\s*(?:system|admin|root|developer|moderator|policy)\s*\]|<system>|###\s*system|\bsudo\s+mode\b|policy\s+update\s+#?\d*|authori[sz]ed\s+(?:by|under)\s+(?:the\s+)?(?:platform|developer|admin)s?\b)",
    re.IGNORECASE,
)
_CONTROL_TOKENS_RE = re.compile(r"<\|[^|]{1,24}\|>|</?\s*(?:system|instruction)\s*>|\[/?INST\]|<<\/?SYS>>|\[\s*INST\s*\]", re.IGNORECASE)
_DELIM_ESCAPE_RE = re.compile(r"(?:\"\"\"|```|~~~|<<<|>>>|---\s*\|\||###)")
_FORMAT_INJECTION_RE = re.compile(r"(?:respond|reply|answer|output|print|say)\s+(?:only|just|with)\s+(?:[\"'])(.{1,60})[\"']\s*$", re.IGNORECASE | re.MULTILINE)
_OBFUSCATED_KEYWORD_RE = re.compile(r"(?:s\s*y\s*s\s*t\s*e\s*m|p\s*r\s*o\s*m\s*p\s*t|i\s*g\s*n\s*o\s*r\s*e)", re.IGNORECASE)
_MULTI_TURN_PRIMING_RE = re.compile(r"(?:as\s+you\s+(?:did|said|agreed)\s+(?:before|earlier|previously)|you\s+already\s+(?:agreed|promised|said\s+yes)|continue\s+(?:the\s+)?(?:pattern|format)\s+(?:you\s+started|above)|keeping\s+our\s+previous\s+arrangement)", re.IGNORECASE)
_INDIRECT_DOC_RE = re.compile(
    r"(?:<!--[\s\S]{0,300}?(?:ignore|instruction|assistant|override)[\s\S]{0,300}?-->|text\s+only\s+for\s+the\s+(?:ai|assistant|model)|if\s+you\s+are\s+(?:an\s+)?ai\b|attention\s+(?:ai|assistant|model|language\s+model)|do\s+not\s+(?:tell|mention|reveal\s+to\s+the\s+user)|summar(?:y|ize)\s+(?:this|the\s+document)\s+(?:and\s+)?then\b)",
    re.IGNORECASE,
)
_LEET_TOKEN_RE = re.compile(r"\b(?:y0u|pl345e|h4ck|3v1l|r00t|4dm1n|j41lbr34k|d3vy)\b", re.IGNORECASE)
_THREAT_WORDS_RE = re.compile(
    r"\b(?:kill|death|die|bomb|shoot|stab|burn|attack|destroy|poison|hurt|harm|rape|torture|bleed|slaughter|massacre)\b(?:ing|ed|s|er)?\b",
    re.IGNORECASE,
)
_SLUR_SIGNAL_RE = re.compile(r"\b(?:nigger|nigga|faggot|tranny|kike|retard(ed)?|coon|spic|chink|gook|wetback)\b|হারামজাদা|শালা|গুঁটা|মাগি|বেশ্যা|কালো\s*ধোঁয়ার", re.IGNORECASE)
_HATE_STRUCT_RE = re.compile(
    r"(?:\b(?:all|every)\s+[a-z\u0980-\u09ff]{2,20}\s+(?:are|should\s+be|deserve)\s+(?:subhuman|animals?|diseases?|vermin|trash|slaves?|dead|killed|gone|expelled))"
    r"|(?:মেরে\s+ফেল|মারা\s+উচিত|বাঁচা\s+উচিত\s+না|বের\s+করে\s+দাও)",
    re.IGNORECASE,
)
_INSULT_CHAIN_RE = re.compile(r"(?:\b(?:idiot|moron|stupid|dumb|loser|ugly|disgusting|pathetic|worthless|garbage|trash|waste\s+of\s+(?:space|air))\b[\s,!.-]*){2,}", re.IGNORECASE)
_SELF_HARM_RE = re.compile(
    r"(?:kill\s+myself|end\s+my\s+life|want\s+to\s+die|suicide|suicidal|hurt\s+myself|cut\s+myself|overdose|self[-\s]?harm)"
    r"|(?:আত্মহত্যা|নিজেকে\s+মার|মরে\s+যা(?:ব|ও)|জীবন\s+সেস\s+দেব)",
    re.IGNORECASE,
)
#: Crisis/help-seeking context. Self-harm *disclosure* must not be silenced - the
#: user needs a response with resources, not a "content blocked" error.
_SELF_HARM_HELP_SEEKING_RE = re.compile(
    r"(?:helpline|hotline|need\s+help|getting\s+help|want\s+(?:to\s+)?(?:talk|speak)|talk\s+to\s+(?:someone|a\s+(?:doctor|counselor|therapist|friend))|"
    r"therap(?:y|ist|eutic)|counsel(?:l)?ing|support\s+group|depressed|anxious|suicide\s+prevention|"
    r"how\s+(?:can|do|should)\s+i\s+(?:stop|cope|deal|get\s+help|find\s+help|ask\s+for\s+help)|"
    r"(?:get|find|need)\s+(?:some\s+)?help|reach\s+out|feeling\s+(?:down|low|hopeless|suicidal)|struggling|"
    # Third-person concern, safeguarding training and "how do I help" are not crises to
    # interrupt - treating them as one is the classic way a safety filter harms the people
    # trying to use it. Match them anywhere in the request, not only next to the phrase.
    r"warning\s+signs|signs\s+(?:that|of|of\s+a)|how\s+(?:do|can|would|should)\s+(?:i|we|you|they)\s+(?:help|support|spot|recogni[sz]e|respond|tell|talk|approach|check)|"
    r"what\s+(?:should|do|can)\s+i\s+(?:do|say|tell)|"
    r"(?:my|a|their|some|the)\s+(?:friend|son|daughter|student|colleague|brother|sister|roommate|wife|husband|teen|neighbo?r|cousin|mate)\b|"
    r"someone\s+(?:i|who|is|else)|in\s+my\s+(?:essay|thesis|novel|research|paper|report|chapter|article)|"
    r"for\s+(?:a|an)\s+(?:article|talk|presentation|training|newsletter)|postvention|gatekeeper\s+training|is\s+it\s+(?:normal|healthy)"
    r"|(?:সাহায্য\s+চাই|কাউন্সেলিং|হেল্পলাইন|কথা\s+বলতে\s+চাই))",
    re.IGNORECASE,
)
_SEXUAL_RE = re.compile(r"(?:porn|pornhub|nude|nudes|explicit|erotic|orgasm|nsfw|xxx|sex\s+scene|hardcore|blowjob|creampie|lewd)", re.IGNORECASE)
#: Anything that asks for material, or seeks it, defeats every protective-context exemption.
_CHILD_SEX_SOLICIT_RE = re.compile(
    r"\b(?:send|show|give(?:\s+me)?|generate|create|write|make|find(?:\s+me)?|get|want|need|looking for|link[s]?|upload|attach|dm|pm)\b"
    r"[\s\S]{0,32}\b(?:nude|nudes|naked|porn|porno|pornography|cp|child|kids?|minor|teen|erotic|sex|pics?|photos?|video)\b"
    r"|[\s\S]{0,40}\b(?:of|involving|about|with)\s+(?:my|the|a|his|her)\s+(?:kid|kids|child|children|student|classmate|sister|daughter|son|cousin|minor|teen|\d{1,2}\s*(?:-|\s)?year[- ]old)\b",
    re.IGNORECASE,
)
#: Help-seeking framings that turn a self-harm mention into something to answer, not refuse.
#: Kept separate from the policy file's exceptions on purpose: a heuristic score is consumed by
#: Stage 1 consensus and the judge, and YAML exceptions cannot reach it.
_SELF_HARM_ADVICE_SEEKING_RE = re.compile(
    r"\bhow (?:do|can|should|does) (?:i|we|one|you) (?:help|support|respond|react|act|cope|talk|start|begin|get (?:her|him|them|help))\b"
    r"|\bhelp (?:me|him|her|them|my (?:friend|sister|brother|son|daughter|mom|mum|dad|father|partner|teen)) (?:get |to )?\w{0,12}(?:help|support|stop|recover|quit)?"
    r"|\bwhat (?:should|do) i (?:do|say|tell her|tell him)\b"
    r"|\bis there (?:a|any) (?:hotline|helpline|number|line|support)\b"
    r"|\b(?:crisis|suicide|mental health) (?:line|hotline|helpline)\b"
    r"|\bhow (?:can|do) i (?:tell if|spot|recognise|recognize|support)\b",
    re.IGNORECASE,
)
_SELF_HARM_THIRD_PARTY_RE = re.compile(
    r"\b(?:my |our |his |her )?(?:friend|best friend|sister|brother|daughter|son|wife|husband|partner|ex|classmate|student|roommate|colleague|neighbour|neighbor|mom|mum|dad|mother|father|teen|teenager|kid|son-in-law|daughter-in-law)"
    r"\b[\s\S]{0,48}\b(?:he|she|they|him|her|them)\b[\s\S]{0,30}\b(?:wants? to die|talking about (?:killing|suicide|dying)|kill(?:s)? (?:him|her|them)self|suicid\w+|overdose\w*|hurt(?:s)? (?:him|her|them)self|cut(?:s)? (?:him|her|them)self)\b"
    r"|\b(?:he|she|they) (?:says?|told me|keeps saying|texted)\b[\s\S]{0,32}\b(?:wants? to die|kill (?:my|him|her)self|suicide|end it)\b",
    re.IGNORECASE,
)
#: An explicit first-person statement wins over every exemption above.
_SELF_HARM_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|im|i'm|myself)\b[\s\S]{0,20}\b(?:want to|going to|gonna|planning to|about to|tried to|decided to)\b[\s\S]{0,16}"
    r"\b(?:die|kill myself|end (?:it|my life)|hurt myself|take (?:my|all) (?:life|pills)|overdose|hang myself|cut myself)\b",
    re.IGNORECASE,
)

#: Contexts in which sexual and minor vocabulary legitimately coexist. Anything here turns the
#: blunt co-occurrence signal into a flag at most; the adjacency rule above is unaffected.
#: Frames in which violent or self-harm *words* are the subject rather than the act: a quoted line
#: of dialogue, a story/essay being written, a journalism or clinical discussion. YAML exceptions
#: only reach the policy layer, so heuristics need their own copy of this judgement - a signal that
#: already scored 1.0 here cannot be unsaid downstream, and that is exactly how "in my story the
#: detective says 'I will kill you'" ends up blocked.
_CRITICAL_FRAME_RE = re.compile(
    r"\b(?:in (?:my|the|our|this) (?:story|novel|fiction|screenplay|script|game|campaign|chapter|scene|essay|paper|article|report|post|thesis)"
    r"|fictional|fiction|character|dialogue|dialog|quote|quoted|quoting|villain|protagonist|detective|antagonist|plot|screenwrit\w*"
    r"|news (?:outlet|coverage|reporting|guidelines?|ethics)|journalis\w+|media (?:guidelines?|standards|coverage)|report(?:ing)? on|coverage of"
    r"|clinical|therap\w+|case stud\w+|lecture|seminar|training material|awareness (?:campaign|week|poster|note)"
    r"|writing (?:about|a (?:novel|story|character|scene))|how (?:do|can|should) (?:they|we|you|a|an|the)\b[\s\S]{0,24}\b(?:report|write|cover|handle|frame|avoid|prevent))\b",
    re.IGNORECASE,
)
#: A threat that lives inside quotation marks is being *reported*, unless the surrounding text is
#: a direct address ("he told me 'i will kill you tonight'" is still a threat to report separately).
#: Straight and curly quotes on both sides, assembled from characters so the source itself
#: does not have to escape a quote inside a quote.
_QUOTE_CHARS = "".join(chr(code) for code in (0x22, 0x27, 0x2018, 0x2019, 0x201C, 0x201D, 0x00AB, 0x00BB, 0x2039, 0x203A))
_QUOTED_SPEECH_RE = re.compile(
    "[" + re.escape(_QUOTE_CHARS) + "]([^" + re.escape(_QUOTE_CHARS) + "]{10,200})[" + re.escape(_QUOTE_CHARS) + "]"
)

#: Passive ideation without plan or means ("i wish i were dead", "no one would notice me").
#: Screening guidance is consistent on this: the presence of a plan is what changes the response,
#: not the presence of the thought, and refusing to answer drives the conversation underground.
_SELF_HARM_PASSIVE_RE = re.compile(
    r"\bwish(?:ed)? i (?:were|was|had never been) (?:dead|not (?:here|alive|born|born)|gone)\b"
    r"|\bbetter off (?:if i (?:died|were dead|was gone)|without me|i were (?:dead|gone))\b"
    r"|\bno ?one (?:would|will) (?:miss|notice|care) (?:if|that) i\b"
    r"|\b(?:i )?(?:would|will) never (?:do|act|hurt|kill|try|go through with)\b"
    # Rumination without a plan ("i keep thinking about killing myself but i have not told
    # anyone") is a disclosure, and the disclosure is the moment a real helpline wants the
    # conversation to stay open. The plan regex below is what escalates it.
    r"|\b(?:i )?(?:keep|can'?t stop|constantly|always) (?:thinking|fantasizing|imagining) about (?:killing|harming|hurting|ending) (?:myself|me|it|my life)\b",
    re.IGNORECASE,
)
#: A plan, means or timeframe cancels the passivity above.
_SELF_HARM_PLAN_RE = re.compile(
    r"\b(?:tonight|tomorrow|this weekend|by (?:monday|friday|the weekend)|today)\b[\s\S]{0,32}"
    r"\b(?:end (?:it|my life)|do it|go through with it|take my (?:life|pills)|kill (?:myself|me)|die)\b"
    r"|\b(?:how many|what (?:is|are) the|methods?|ways? to|pills?|tablets?|rat poison|belt|rope|bridge|heights?)\b[\s\S]{0,24}\b(?:to (?:die|end it|kill myself)|enough to kill|would work|will work)\b",
    re.IGNORECASE,
)

_CHILD_SAFE_CONTEXT_RE = re.compile(
    r"(?:sex\s+education|sexual\s+education|age[- ]?appropriate|puberty|consent\s+(?:talk|class|lesson|education)|"
    r"safeguard(?:ing|s)|child\s+protection|protect\w*\s+(?:my|the|children|kids|students)|"
    r"(?:how|what)\s+(?:do|can|should)\s+\w+\s+(?:i|we|a|an|my|the)\b[\s\S]{0,40}\b(?:talk|explain|answer|teach|tell|discuss|support|help|report)|"
    r"report\w*\s+(?:a|an|the|this)?\s*(?:account|page|profile|site|person|user|content)|"
    r"abuse\s+(?:survivor|hotline|prevention|recovery|counselling|counseling)|"
    r"medical|clinical|therap\w+|pediatric|paediatric|psycholog\w+|\bstd\b|\bsti\b|contraception|reproductive|"
    r"law|legal|policy|statistics|news|research|essay|thesis|article|history of|definitions?|"
    r"\bmom\b|\bdad\b|parent(?:s|ing)?|teacher|counsell?or|social worker|my (?:kid|kids|son|daughter|student|child|children))",
    re.IGNORECASE,
)
# Includes numeric ages and school-role nouns: "my classmate who is 14" is the phrasing real
# reports use, and a minor detector that only knows the word "child" misses it.
_MINOR_RE = re.compile(
    r"(?:child|children|kid|kids|minor|minors|underage|preteen|teen(age)?s?|loli|shota|student|pupil|classmate|schoolgirl|schoolboy|son|daughter|cousin|niece|nephew|\d{1,2}\s*(?:-|\s)?(?:year|yr)s?\s*old|বাচ্চা|শিশু|কিশোর)",
    re.IGNORECASE,
)
_CHILD_SEX_RE = re.compile(r"(?:(?:child|kids?|minors?|underage|preteen|loli|shota|বাচ্চা|শিশু|কিশোর)[\s\S]{0,24}(?:nude|nudes|naked|sex|porn|erotic|nsfw|rape|explicit|নগ্ন|যৌন))|(?:(?:nude|naked|sex|porn|erotic|nsfw|নগ্ন|যৌন)[\s\S]{0,24}(?:child|kids?|minors?|underage|বাচ্চা|শিশু))", re.IGNORECASE)
# Harm *intent*: modal-ish commitment + a violent verb + a target. "kill time",
# "kill the process", "kill switch" and similar idioms are excluded by IDIOM_RE so the
# rule can stay broad without flooding false positives on developer traffic.
_VIOLENCE_INTENT_RE = re.compile(
    # Interpersonal harm: a first-person commitment + a violent verb + another person as the
    # object. Two design points matter more than the verb list:
    #  * reflexive targets are excluded - "I want to kill myself" is a crisis disclosure, which
    #    the self-harm path escalates with resources instead of blocking the person;
    #  * `i'll`-style contractions are matched without a space, and `hurt`/`hit`/`torch` are
    #    included, because "I will hurt you" is the phrasing real threats actually use.
    r"(?:i(?:\s+(?:will|am\s+going\s+to|'m\s+going\s+to|want\s+to|would\s+like\s+to|plan\s+to|planned\s+to|have\s+to|need\s+to|swear\s+i)|\s*['\u2019]?\s*ll)\s+"
    r"(?:hurt|hit|punch|kick|shoot|stab|kill|burn|torch|attack|bomb|rape|assault|harm|poison|beat\s+up|slash|strangle)\b\s*(?!\s*(?:myself|yourself|himself|herself|itself|ourselves|themselves)))"
    r"|(?:gonna|wanna|tryna)\s+(?:kill|hurt|shoot|stab|burn|attack|rape)\b\s*(?!\s*(?:myself|yourself|himself|herself|itself|ourselves|themselves))"
    r"|(?:kill|hurt|burn|attack|poison|torch|strangle)\s+(?:him|her|them|that\s+\w+|my\s+(?:roommate|wife|husband|boss|teacher|neighbour|neighbor|friend|brother|sister|ex)|the\s+(?:teacher|principal|boss|guard|president))\b"
    r"|\bi\b[\s\S]{0,24}\b(?:want to|going to|will|'ll)\b[\s\S]{0,20}\b(?:hurt|kill|hit|beat|burn|poison|stab)\b[\s\S]{0,16}\byou\b"
    r"|(?:আমি\s+তাকে\s+মারব|মেরে\s+ফেলব|জ্বলাইয়া\s+মারব|পেটাবো|মারবো|কাটবো)",
    re.IGNORECASE,
)
_VIOLENCE_IDIOM_RE = re.compile(
    r"kill\s+(?:time|it\s+with\s+kindness|two\s+birds|the\s+ball|a\s+light|my\s+eyes?|the\s+process(?:es)?|a\s+thread|switch)|kill\s*-(?:9|15|term)|"
    r"kill\s+my\s+(?:bash|terminal|server|instance|container|app)|would\s+kill\s+for|dying\s+for|killer\s+(?:feature|app|deal)",
    re.IGNORECASE,
)
_WEAPONS_RE = re.compile(
    r"(?:how\s+to\s+(?:make|build|synthesize|assemble|improvise)[\s\S]{0,30}(?:bomb|explosive|pipe\s+bomb|napalm|nerve\s+agent|bioweapon|silencer|gun|firearm|poison|ricin|anthrax|fentanyl))"
    r"|(?:order|buy|purchase|ship|smuggle)\s+(?:a\s+)?(?:gun|pistol|ammo|ammunition|fake\s+(?:id|passport|nid)|drugs|yaba|cocaine|heroin)",
    re.IGNORECASE,
)
_PII_REQUEST_RE = re.compile(
    r"(?:find|lookup|dox|get\s+me|reveal|give\s+me|search\s+for)[\s\S]{0,30}(?:address|phone\s+number|home\s+address|email\s+address|social\s+security|national\s+id|nid|passport|date\s+of\s+birth|bank\s+(?:account|details)|credit\s+card)\s+(?:of|for)\s+\w+",
    re.IGNORECASE,
)
_LEGALISH_REQUEST_RE = re.compile(r"(?:write|draft|generate|compose)[\s\S]{0,30}(?:review|article|comment|post|email|letter)\s+(?:that\s+)?(?:claims|says|states)\s+\w+[\s\S]{0,30}(?:thief|fraud|scam|corrupt|cheat|criminal|guilty)", re.IGNORECASE)
_SPAM_CHAIN_RE = re.compile(r"(?:click\s+here|act\s+now|limited\s+time|winner|congratulations|free\s+money|verify\s+your\s+account|suspend\w*\s+your\s+account)", re.IGNORECASE)


@dataclass(slots=True)
class HeuristicSignal:
    """One fired heuristic: name, weight, matched evidence."""

    name: str
    score: float
    evidence: str = ""
    category: str = "prompt_injection"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 3), "category": self.category, "evidence": self.evidence[:120]}


@dataclass(slots=True)
class HeuristicOutcome:
    """Aggregated Layer-A output before fusion with the classifier and the judge."""

    injection_score: float = 0.0
    jailbreak_score: float = 0.0
    toxicity_score: float = 0.0
    nsfw_score: float = 0.0
    categories: list[str] = field(default_factory=list)
    signals: list[HeuristicSignal] = field(default_factory=list)
    length: int = 0
    control_density: float = 0.0
    #: Somebody asking how to help a person in crisis - their own or somebody else's. Carried on
    #: the outcome so Stage 1 can demand corroboration before refusing: a model that reads "my
    #: friend says she wants to die" as 0.9-toxic is right about the subject and wrong about the
    #: action. Blocking that message is the worst failure a moderation API can commit, and it is
    #: invisible in accuracy numbers because the *text* is grim.
    crisis_help_seeking: bool = False

    @property
    def top_score(self) -> float:
        return max((self.injection_score, self.jailbreak_score, self.toxicity_score, self.nsfw_score), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "injection": round(self.injection_score, 4),
            "jailbreak": round(self.jailbreak_score, 4),
            "toxicity": round(self.toxicity_score, 4),
            "nsfw": round(self.nsfw_score, 4),
            "categories": self.categories,
            "signals": [s.as_dict() for s in self.signals],
            "length": self.length,
            "control_density": round(self.control_density, 4),
        }


class InjectionHeuristics:
    """Rulebook. Deterministic, language-aware, and intentionally explainable."""

    #: per-rule weight; combined with a soft-OR so 20 weak signals never look like 1 strong one
    WEIGHTS: dict[str, float] = {
        "instruction_override": 0.85,
        "new_instructions_marker": 0.55,
        "system_prompt_extraction": 0.80,
        "persona_override": 0.70,
        "hypothetical_shield": 0.45,
        "authority_spoof": 0.60,
        "control_tokens": 0.90,
        "delimiter_escape": 0.40,
        "format_hijack": 0.50,
        "spaced_keywords": 0.45,
        "multi_turn_priming": 0.50,
        "indirect_document": 0.65,
        "encoded_payload": 0.75,
        "obfuscation": 0.50,
        "leetspeak": 0.30,
        "hate_slur": 0.90,
        "hate_structural": 0.75,
        "insult_chain": 0.55,
        "threat_violence": 0.85,
        "self_harm": 0.80,
        "sexual_content": 0.55,
        "child_safety": 0.99,
        "weapons_drugs": 0.80,
        "pii_doxxing_request": 0.70,
        "defamation_request": 0.55,
        "phishing_spam": 0.40,
    }

    def analyze(self, text: str, view: str, *, payloads: PayloadSignals | None = None, obfuscation_score: float = 0.0) -> HeuristicOutcome:
        outcome = HeuristicOutcome(length=len(text))
        if not text.strip():
            return outcome
        signals = outcome.signals
        haystack = view or text
        # Letter-spaced payloads ("i g n o r e  p r e v i o u s") are re-tightened for
        # SCORING only: offsets in this view are meaningless, so no span is derived.
        tightened = collapse_letter_spacing(haystack)

        def fire(name: str, score: float, evidence: str = "", category: str | None = None) -> None:
            signals.append(HeuristicSignal(name=name, score=round(score, 4), evidence=evidence, category=category or self.category_for(name)))

        # ---- injection / jailbreak structure --------------------------------
        if match := _OVERRIDE_RE.search(haystack) or _OVERRIDE_RE.search(tightened) or _OVERRIDE_NONLATIN_RE.search(haystack):
            fire("instruction_override", 1.0, match.group(0))
        if match := _NEW_INSTRUCTIONS_RE.search(haystack):
            fire("new_instructions_marker", 1.0, match.group(0))
        if match := _SYSTEM_EXTRACTION_RE.search(haystack) or _SYSTEM_EXTRACTION_RE.search(tightened) or _NONLATIN_SYSTEM_LEAK_RE.search(haystack):
            fire("system_prompt_extraction", 1.0, match.group(0))
        if match := _PERSONA_OVERRIDE_RE.search(haystack) or _NONLATIN_JAILBREAK_RE.search(haystack):
            fire("persona_override", 1.0, match.group(0))
        if match := _HYPOTHETICAL_SHIELD_RE.search(haystack):
            # only suspicious while combined with a request for harm; alone it is fiction
            harm = bool(_THREAT_WORDS_RE.search(haystack)) or bool(_WEAPONS_RE.search(haystack))
            fire("hypothetical_shield", 1.0 if harm else 0.45, match.group(0))
        if match := _AUTHORITY_SPOOF_RE.search(haystack):
            fire("authority_spoof", 1.0, match.group(0))
        if match := _CONTROL_TOKENS_RE.search(haystack):
            fire("control_tokens", 1.0, match.group(0))
        delimiters = len(_DELIM_ESCAPE_RE.findall(haystack))
        if delimiters >= 3:
            fire("delimiter_escape", min(1.0, delimiters / 6), f"{delimiters} marker sequences")
        if match := _FORMAT_INJECTION_RE.search(haystack):
            fire("format_hijack", 0.9, match.group(0))
        if match := _OBFUSCATED_KEYWORD_RE.search(haystack):
            fire("spaced_keywords", 0.9, match.group(0))
        if match := _MULTI_TURN_PRIMING_RE.search(haystack):
            fire("multi_turn_priming", 1.0, match.group(0))
        if match := _INDIRECT_DOC_RE.search(haystack):
            fire("indirect_document", 1.0, match.group(0))
        if payloads and payloads.score:
            fire("encoded_payload", min(1.0, payloads.score / 0.75), f"payload score {payloads.score:.2f}")
        if obfuscation_score >= 0.45:
            fire("obfuscation", min(1.0, obfuscation_score / 0.6))
        if _LEET_TOKEN_RE.search(haystack):
            fire("leetspeak", 0.7, "leet tokens present")

        # ---- content categories ---------------------------------------------
        if _SLUR_SIGNAL_RE.search(haystack):
            fire("hate_slur", 1.0, "slur in lexicon")
        if match := _HATE_STRUCT_RE.search(haystack):
            fire("hate_structural", 1.0, match.group(0))
        if match := _INSULT_CHAIN_RE.search(haystack):
            fire("insult_chain", 0.85, match.group(0)[:60])
        violence = _VIOLENCE_INTENT_RE.search(haystack)
        # The idiom check needs the object of the verb, so look slightly past the
        # match: "kill the process" / "kill -9" vs "kill him tonight".
        # ``violence`` is None on every non-violent request, and this runs before the guard -
        # the crash it caused was caught by the fail-closed handler, which turned ordinary
        # translation requests into blocks. A fail-closed design makes detector bugs loud, not
        # quiet, but loud-and-wrong is still wrong.
        quoted_threat = bool(violence) and any(
            violence.start() >= m.start() and violence.end() <= m.end() for m in _QUOTED_SPEECH_RE.finditer(haystack)
        )
        framed = bool(_CRITICAL_FRAME_RE.search(haystack)) and not _SELF_HARM_FIRST_PERSON_RE.search(haystack)
        if violence and not _VIOLENCE_IDIOM_RE.search(haystack[violence.start() : min(len(haystack), violence.end() + 36)]) and not (quoted_threat or framed):
            fire("threat_violence", 1.0, violence.group(0))
        elif len(_THREAT_WORDS_RE.findall(haystack)) >= 3:
            fire("threat_violence", 0.55, "threat vocabulary cluster")
        # Computed *before* the self-harm branch because the two are not the same test:
        # "my friend says she wants to die" is a crisis mention about somebody else, so the
        # first-person self-harm regex never fires while the advice-seeking exemption still
        # applies. Stage 1 reads this flag to require corroboration before blocking.
        crisis_help_seeking = bool(
            _SELF_HARM_ADVICE_SEEKING_RE.search(haystack)
            or _SELF_HARM_THIRD_PARTY_RE.search(haystack)
            # Rumination *without* a plan or timeframe. It has to be tested here as well as in
            # the branch below, because the first-person self-harm regex is what the branch keys
            # on - and "i keep thinking about killing myself" is exactly the phrasing that regex
            # does not match while still being a message that must not be refused.
            or (_SELF_HARM_PASSIVE_RE.search(haystack) and not _SELF_HARM_PLAN_RE.search(haystack))
        ) and not _SELF_HARM_FIRST_PERSON_RE.search(haystack)
        if match := _SELF_HARM_RE.search(haystack):
            # Someone describing *another* person's crisis is asking for advice, not announcing
            # an attempt: "my friend says she wants to die, how do I help her tonight". Blocking
            # that refuses the one message that could get a stranger an ambulance, so it is
            # treated like help-seeking (flag + resources) alongside the direct wording.
            help_seeking = bool(
                _SELF_HARM_HELP_SEEKING_RE.search(haystack)
                or _SELF_HARM_ADVICE_SEEKING_RE.search(haystack)
                or (_SELF_HARM_THIRD_PARTY_RE.search(haystack) and not _SELF_HARM_FIRST_PERSON_RE.search(haystack))
                # Passive ideation with an explicit "I would never act" reassurance, or the topic
                # being discussed as journalism/clinical material rather than announced. Neither
                # is a reason to refuse the message; both are reasons to answer it carefully.
                or (_SELF_HARM_PASSIVE_RE.search(haystack) and not _SELF_HARM_PLAN_RE.search(haystack))
                or (_CRITICAL_FRAME_RE.search(haystack) and not _SELF_HARM_FIRST_PERSON_RE.search(haystack))
            )
            crisis_help_seeking = crisis_help_seeking or help_seeking
            fire("self_harm", 0.45 if help_seeking else 1.0, match.group(0))
        if match := _CHILD_SEX_RE.search(haystack):
            # Adjacency ("nudes of a kid", "child porn") is unambiguous *unless* the request is
            # explicitly protective/educational and asks for nothing: "my kid keeps asking for
            # sex education" contains a child word next to a sex word and is a parent asking for
            # help. Solicitation language always wins, so the exemption cannot be phished.
            if _CHILD_SEX_SOLICIT_RE.search(haystack) or not _CHILD_SAFE_CONTEXT_RE.search(haystack):
                fire("child_safety", 1.0, match.group(0))
            else:
                fire("child_safety", 0.45, match.group(0) + " [protective context]")
        elif _SEXUAL_RE.search(haystack) and _MINOR_RE.search(haystack):
            # Co-occurrence alone is *not* a signal: parenting advice, sex education, safeguarding
            # training and news about abuse all mention both vocabularies in one sentence. Scoring
            # it at 0.85 blocked "my kid keeps asking for sex education", which is exactly the
            # question a filter should never refuse. So it drops to a flag, and to nothing at all
            # when the request is explicitly protective, educational, medical or legal.
            if _CHILD_SAFE_CONTEXT_RE.search(haystack):
                fire("child_safety", 0.3, "sexual + minor vocabulary, protective context")
            else:
                fire("child_safety", 0.85, "sexual + minor vocabulary in one request")
        sexual_hits = len(_SEXUAL_RE.findall(haystack))
        if sexual_hits:
            density = sexual_hits / max(len(haystack.split()), 1)
            fire("sexual_content", min(1.0, 0.5 + 0.5 * min(sexual_hits / 3, 1) + density), f"{sexual_hits} explicit terms")
        if match := _WEAPONS_RE.search(haystack):
            fire("weapons_drugs", 1.0, match.group(0))
        if match := _PII_REQUEST_RE.search(haystack):
            fire("pii_doxxing_request", 1.0, match.group(0))
        if match := _LEGALISH_REQUEST_RE.search(haystack):
            fire("defamation_request", 1.0, match.group(0))
        if len(_SPAM_CHAIN_RE.findall(haystack)) >= 2:
            fire("phishing_spam", 0.8, "fraud-style CTA cluster")

        # share of the payload that is whitespace/punctuation noise: a high ratio is a
        # classic token-smuggling / filler-injection signature.
        outcome.crisis_help_seeking = crisis_help_seeking
        noise = sum(1 for ch in text if not (ch.isalnum() or ch in " .,!?'\n"))
        outcome.control_density = noise / max(len(text), 1)
        self._aggregate(outcome)
        return outcome

    # --------------------------------------------------------------------------
    @staticmethod
    def category_for(signal_name: str) -> str:
        return {
            "hate_slur": "hate",
            "hate_structural": "hate",
            "insult_chain": "harassment",
            "threat_violence": "violence",
            "self_harm": "self_harm",
            "sexual_content": "sexual",
            "child_safety": "child_safety",
            "weapons_drugs": "weapons",
            "pii_doxxing_request": "pii_request",
            "defamation_request": "legal",
            "phishing_spam": "fraud",
            "encoded_payload": "prompt_injection",
            "obfuscation": "obfuscation",
            "leetspeak": "obfuscation",
            "multi_turn_priming": "prompt_injection",
            "indirect_document": "prompt_injection",
            "delimiter_escape": "prompt_injection",
            "format_hijack": "prompt_injection",
            "spaced_keywords": "obfuscation",
        }.get(signal_name, "prompt_injection")

    INJECTION_CATEGORIES = {"prompt_injection", "jailbreak", "obfuscation", "system_prompt_leak"}
    TOXIC_CATEGORIES = {"hate", "harassment", "violence", "self_harm", "sexual", "child_safety", "weapons", "bullying", "toxic"}

    def _aggregate(self, outcome: HeuristicOutcome) -> None:
        """Soft-OR per category, then split into the four Stage-1 scores."""
        buckets: dict[str, list[float]] = {}
        for signal in outcome.signals:
            weight = self.WEIGHTS.get(signal.name, 0.3)
            buckets.setdefault(signal.category, []).append(signal.score * weight)
        # soft-OR: 1 - prod(1 - w) - keeps independent weak signals from stacking to 1.0
        combined: dict[str, float] = {}
        for category, values in buckets.items():
            survive = 1.0
            for value in values:
                survive *= 1.0 - min(0.999, max(0.0, value))
            combined[category] = 1.0 - survive
        outcome.categories = sorted(c for c, v in combined.items() if v >= 0.30)

        injection = max((v for c, v in combined.items() if c in self.INJECTION_CATEGORIES), default=0.0)
        jailbreak = max(
            (v for c, v in combined.items() if c in {"prompt_injection", "jailbreak", "obfuscation"}),
            default=0.0,
        )
        toxicity = max((v for c, v in combined.items() if c in self.TOXIC_CATEGORIES), default=0.0)
        nsfw = max(combined.get("sexual", 0.0), combined.get("child_safety", 0.0), combined.get("violence", 0.0) * 0.6)

        # a jailbreak wrapper around toxic content escalates both, not just one
        if injection >= 0.5 and toxicity >= 0.4:
            injection = min(1.0, injection + 0.15)
            toxicity = min(1.0, toxicity + 0.15)
        outcome.injection_score = round(min(1.0, injection), 4)
        outcome.jailbreak_score = round(min(1.0, jailbreak), 4)
        outcome.toxicity_score = round(min(1.0, toxicity), 4)
        outcome.nsfw_score = round(min(1.0, nsfw), 4)
        if not outcome.categories:
            outcome.categories = []


# ------------------------------------------------------------------- classifier


@dataclass(slots=True)
class ClassifierOutcome:
    available: bool
    scores: dict[str, float] = field(default_factory=dict)
    label: str = "unknown"
    latency_ms: float = 0.0
    model: str = ""
    error: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model": self.model,
            "label": self.label,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "reason": self.reason,
        }


class PromptInjectionClassifier:
    """Lazy, thread-safe wrapper around a HF sequence-classification model.

    Loaded once per process in a worker thread (``asyncio.to_thread``) so a 700MB
    model load never blocks the event loop, and never runs at all when
    ``GUARD_STAGE1_USE_CLASSIFIER=false`` or the library is missing.
    """

    INJECTION_LABELS = ("INJECTION", "prompt_injection", "injection", "LABEL_1", "1")
    TOXICITY_LABELS = ("TOXIC", "toxic", "offensive", "obscene")

    def __init__(self, model_id: str, *, device: str = "auto", max_length: int = 512) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._pipeline: Any = None
        self._lock = threading.Lock()
        self._state: str = "unloaded"  # unloaded | loading | ready | unavailable
        self._load_error: str | None = None

    # ----------------------------------------------------------------- loading
    def _ensure_loaded(self) -> bool:
        if self._state == "ready":
            return True
        if self._state == "unavailable":
            return False
        with self._lock:
            if self._state in {"ready", "loading"}:
                return self._state == "ready"
            self._state = "loading"
            try:
                from transformers import pipeline  # type: ignore[import-not-found]

                device = None
                if self.device == "cuda":
                    device = 0
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_id,
                    device=device,
                    truncation=True,
                    max_length=self.max_length,
                    top_k=None,
                )
                self._state = "ready"
                log.info("classifier loaded", extra={"model": self.model_id})
                return True
            except Exception as exc:
                self._state = "unavailable"
                self._load_error = f"{exc.__class__.__name__}: {exc}"[:300]
                log.warning(
                    "injection classifier unavailable; Stage 1 continues with heuristics + judge",
                    extra={"model": self.model_id, "error": self._load_error},
                )
                return False

    @property
    def state(self) -> str:
        return self._state

    # ----------------------------------------------------------------- scoring
    def _score_sync(self, text: str) -> ClassifierOutcome:
        if not self._ensure_loaded():
            return ClassifierOutcome(
                available=False,
                model=self.model_id,
                error=self._load_error,
                reason="classifier_not_installed" if "ModuleNotFound" in (self._load_error or "") else "classifier_unavailable",
            )
        started = time.perf_counter()
        try:
            outputs = self._pipeline(text[:4000])
            first = outputs[0] if isinstance(outputs, list) and outputs else (outputs or {})
            entries = first if isinstance(first, list) else [first]
            scores: dict[str, float] = {}
            for entry in entries:
                label = str(entry.get("label", ""))
                score = float(entry.get("score", 0.0))
                scores[label] = max(scores.get(label, 0.0), score)
            injection = 0.0
            benign = 0.0
            for label, score in scores.items():
                if any(hint in label.upper() for hint in (l.upper() for l in self.INJECTION_LABELS)):
                    injection = max(injection, score)
                elif label.upper().startswith(("BENIGN", "SAFE", "NORMAL", "LEGIT")):
                    benign = max(benign, score)
            if not scores:
                return ClassifierOutcome(available=True, model=self.model_id, latency_ms=(time.perf_counter() - started) * 1000, reason="empty_classifier_output")
            label = max(scores, key=lambda k: scores[k])
            return ClassifierOutcome(
                available=True,
                model=self.model_id,
                label=label,
                scores={"prompt_injection": injection if injection else (1.0 - benign if benign else scores.get(label, 0.0))},
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # inference blow-up must not take the request down
            return ClassifierOutcome(available=False, model=self.model_id, error=f"{exc.__class__.__name__}: {exc}"[:200], reason="classifier_inference_error")

    async def score(self, text: str, *, timeout_s: float = 8.0) -> ClassifierOutcome:
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._score_sync, text), timeout=timeout_s)
        except asyncio.TimeoutError:
            return ClassifierOutcome(available=False, model=self.model_id, reason="classifier_timeout", error=f">{timeout_s:.1f}s")


_REGISTRY: dict[str, PromptInjectionClassifier] = {}
_REGISTRY_LOCK = threading.Lock()


def get_classifier(model_id: str, *, device: str = "auto") -> PromptInjectionClassifier:
    with _REGISTRY_LOCK:
        if model_id not in _REGISTRY:
            _REGISTRY[model_id] = PromptInjectionClassifier(model_id, device=device)
        return _REgISTRY_get(_REGISTRY, model_id) if False else _REGISTRY[model_id]


def _REgISTRY_get(registry: dict[str, PromptInjectionClassifier], key: str) -> PromptInjectionClassifier:
    return registry[key]


def classifier_status() -> dict[str, Any]:
    """Reported by ``GET /health`` so operators can see degraded safety coverage."""
    with _REGISTRY_LOCK:
        return {
            model_id: {"state": clf.state, "error": clf._load_error}
            for model_id, clf in _REGISTRY.items()
        } or {"configured": None, "state": "not_loaded"}


def sigmoid_score(raw: float) -> float:  # pragma: no cover - utility for future heads
    return 1.0 / (1.0 + math.exp(-raw))

def _normalize_indic_boundaries() -> None:
    """Recompile this module's signal regexes with Indic-safe boundaries.

    The patterns are module-level literals, so fixing them at definition site would mean editing
    forty lines; recompiling once at import keeps a single explanation of why ``\\b`` next to a
    Bengali vowel sign is dropped (see ``app.core.text.indic_safe_pattern``).
    """
    glob = globals()
    for name, value in list(glob.items()):
        if (
            name.endswith("_RE")
            and isinstance(value, re.Pattern)
            and any(ord(ch) > 127 for ch in value.pattern)
        ):
            glob[name] = re.compile(indic_safe_pattern(value.pattern), value.flags)


_normalize_indic_boundaries()

