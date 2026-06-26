import json
import math
import os
import re
import sys
import time
from curl_cffi import requests
import genanki
from dotenv import load_dotenv

load_dotenv()

# Lists the user's saved words (needs MW_COOKIE).
MW_BASE = "https://www.merriam-webster.com/lapi/v1/wordlist"
# Collegiate Dictionary: definitions, pronunciation, etymology, examples.
MW_DICT = "https://www.dictionaryapi.com/api/v3/references/collegiate/json"
MW_DICT_KEY = os.environ.get("MW_DICT_KEY", "")
# Collegiate Thesaurus: synonyms, antonyms.
MW_THES = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json"
MW_THES_KEY = os.environ.get("MW_THESAURUS_KEY", "")

# Full MW records (source of truth, costly to refetch); lean card/viz subset.
FULL_CACHE = "words.full.json"
CACHE_FILE = "words.json"

OUTPUT_DECK = "mw_words.apkg"
PER_PAGE = 16


# --- Merriam-Webster markup ---
# MW text embeds tokens like {it}..{/it}, {bc}, {sx|word||}, {d_link|word|id}.
# https://dictionaryapi.com/products/json#sec-2.tokens
_MW_ENTITIES = {
    "{bc}": ": ",
    "{ldquo}": "“", "{rdquo}": "”",
    "{p_br}": " ",
    "{inf}": "", "{/inf}": "",
    "{sup}": "", "{/sup}": "",
}
# tokens of the form {tag|word|id|...} where the first arg is the display word
_MW_LINK_RE = re.compile(r"\{(?:sx|dx_def|d_link|a_link|i_link|dxt|et_link|mat|dx_ety)\|([^|}]*)[^}]*\}")
# {ds|...} date-sense markers, {dx ...}..{/dx} cross-ref blocks, and any leftover {..}
_MW_ANY_TAG_RE = re.compile(r"\{[^}]*\}")


def _strip_mw_tags(text):
    """Convert MW markup tokens into plain readable text."""
    if not text:
        return ""
    for tok, rep in _MW_ENTITIES.items():
        text = text.replace(tok, rep)
    text = _MW_LINK_RE.sub(r"\1", text)
    text = _MW_ANY_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lstrip(": ").strip()


# Audio CDN path; subdirectory is derived from the filename per MW's rules.
def _audio_url(audio):
    if not audio:
        return ""
    if audio.startswith("bix"):
        sub = "bix"
    elif audio.startswith("gg"):
        sub = "gg"
    elif audio[0].isdigit() or not audio[0].isalpha():
        sub = "number"
    else:
        sub = audio[0]
    return f"https://media.merriam-webster.com/audio/prons/en/us/mp3/{sub}/{audio}.mp3"


def _first_pron(prs):
    """(written pronunciation, audio url) from a list of `prs` objects."""
    for p in prs or []:
        written = p.get("mw") or p.get("ipa") or ""
        audio = _audio_url(p.get("sound", {}).get("audio", ""))
        if written or audio:
            return written, audio
    return "", ""


def _extract_dt(dt, defs, examples):
    """Walk a sense's `dt` list, collecting definition text and `vis` examples."""
    for item in dt or []:
        if not isinstance(item, list) or len(item) != 2:
            continue
        kind, val = item
        if kind == "text":
            t = _strip_mw_tags(val)
            if t:
                defs.append(t)
        elif kind == "vis":
            for ex in val:
                if isinstance(ex, dict) and ex.get("t"):
                    examples.append(_strip_mw_tags(ex["t"]))
        elif kind == "uns":  # usage note may itself nest dt/vis
            for grp in val:
                _extract_dt(grp, defs, examples)


def _walk_sseq(sseq, senses, examples):
    """Recursively collect (sense_number, text) pairs from a `def[].sseq`."""
    for node in sseq or []:
        if not isinstance(node, list):
            continue
        # a [type, obj] pair?
        if len(node) == 2 and node[0] in ("sense", "sen"):
            obj = node[1]
            sn = obj.get("sn", "")
            local_defs = []
            _extract_dt(obj.get("dt", []), local_defs, examples)
            # nested sub-senses (sdsense)
            sd = obj.get("sdsense")
            if sd:
                _extract_dt(sd.get("dt", []), local_defs, examples)
            for d in local_defs:
                senses.append((sn, d))
        else:
            # bs / pseq / nested sequence -> recurse
            _walk_sseq(node, senses, examples)

MW_HEADERS = {
    "cookie": os.environ["MW_COOKIE"],
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "referer": "https://www.merriam-webster.com/",
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
}


def fetch_all_words():
    r = requests.get(f"{MW_BASE}/get-total-count", headers=MW_HEADERS, impersonate="chrome")
    r.raise_for_status()
    total = r.json()["data"]["data"]["total_count"]
    pages = math.ceil(total / PER_PAGE)
    print(f"Fetching {total} words across {pages} pages...")

    words = []
    for page in range(1, pages + 1):
        r = requests.get(
            f"{MW_BASE}/search",
            params={"search": "", "sort": "newest", "filter": "dt", "page": page, "perPage": PER_PAGE},
            headers=MW_HEADERS,
            impersonate="chrome",
        )
        r.raise_for_status()
        for item in r.json()["data"]["data"]["items"]:
            words.append(item["word"])
        time.sleep(0.3)

    return words


class MWError(Exception):
    """Raised when an MW API call fails in a way that should halt the run."""


def _mw_get(url, key, word, retries=3):
    """GET an MW endpoint, retrying transient errors. Raises MWError on auth
    failure or persistent error so the run halts (resumable from the cache)."""
    if not key:
        raise MWError(f"missing API key for {url}")
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params={"key": key}, impersonate="chrome", timeout=20)
        except Exception as e:  # network/timeout
            last = e
            time.sleep(min(2 ** attempt, 8))
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception as e:
                # Invalid key returns a non-JSON body; do not retry — it won't fix itself.
                raise MWError(f"non-JSON response for '{word}' (check API key): {r.text[:120]}") from e
            if not isinstance(data, list):
                raise MWError(f"unexpected response shape for '{word}': {type(data).__name__}")
            return data
        if r.status_code in (429, 500, 502, 503, 504):  # transient
            last = MWError(f"HTTP {r.status_code} for '{word}'")
            time.sleep(min(2 ** attempt, 8))
            continue
        # 4xx other than 429 (e.g. 403 bad key) — fatal
        raise MWError(f"HTTP {r.status_code} for '{word}': {r.text[:120]}")
    raise MWError(f"giving up on '{word}' after {retries} attempts: {last}")


def _matching_entries(data, word):
    """Entries whose headword id matches `word` (ignoring :homograph suffix)."""
    matches = [
        e for e in data
        if isinstance(e, dict)
        and re.sub(r":.*$", "", e.get("meta", {}).get("id", "")).lower() == word.lower()
    ]
    # fall back to all dict entries if MW's id casing/variant differs
    return matches or [e for e in data if isinstance(e, dict)]


def parse_dict(data, word):
    """Turn a raw Collegiate Dictionary response into clean fields (no network)."""
    empty = {"dict_found": False}
    if not data or not isinstance(data[0], dict):
        return empty  # suggestion-string list -> no real entry
    entries = _matching_entries(data, word)
    if not entries:
        return empty

    headword = re.sub(r":.*$", "", entries[0].get("meta", {}).get("id", "")) or word
    parts_of_speech, definitions, short_definitions = [], [], []
    examples, etymology, first_use = [], "", ""
    pronunciation, audio = "", ""
    inflections, related_forms, quotes = [], [], []

    for e in entries:
        fl = e.get("fl", "")
        if fl and fl not in parts_of_speech:
            parts_of_speech.append(fl)
        if not pronunciation and not audio:
            pronunciation, audio = _first_pron(e.get("hwi", {}).get("prs", []))
        # short, clean definitions (the headline sense list)
        for sd in e.get("shortdef", []):
            tagged = f"({fl}) {sd}" if fl else sd
            if tagged not in short_definitions:
                short_definitions.append(tagged)
        # full numbered senses from def -> sseq
        for section in e.get("def", []):
            senses = []
            _walk_sseq(section.get("sseq", []), senses, examples)
            vd = section.get("vd", "")  # verb divider, e.g. "transitive verb"
            for sn, text in senses:
                prefix = f"({fl}) " if fl else ""
                num = f"{sn} " if sn else ""
                vdp = f"[{vd}] " if vd else ""
                line = f"{prefix}{vdp}{num}{text}".strip()
                if line not in definitions:
                    definitions.append(line)
        # etymology
        if not etymology:
            for et in e.get("et", []):
                if isinstance(et, list) and len(et) == 2 and et[0] == "text":
                    etymology = _strip_mw_tags(et[1])
                    break
        # first known use
        if not first_use and e.get("date"):
            first_use = _strip_mw_tags(e["date"])
        # inflections (plurals, verb forms)
        for ins in e.get("ins", []):
            form = ins.get("if", "").replace("*", "")
            if form and form not in inflections:
                inflections.append(form)
        # related word forms (run-ons): aphorism -> aphorist, aphoristic
        for uro in e.get("uros", []):
            form = uro.get("ure", "").replace("*", "")
            if form:
                related_forms.append({"word": form, "part_of_speech": uro.get("fl", "")})
        # real attributed usage quotes
        for q in e.get("quotes", []):
            text = _strip_mw_tags(q.get("t", ""))
            if not text:
                continue
            aq = q.get("aq", {})
            quotes.append({
                "text": text,
                "author": aq.get("auth", ""),
                "source": _strip_mw_tags(aq.get("source", "")),
            })

    # dedupe examples, preserve order
    seen, ex_clean = set(), []
    for ex in examples:
        if ex and ex not in seen:
            seen.add(ex)
            ex_clean.append(ex)

    return {
        "word": headword,
        "part_of_speech": parts_of_speech,
        "pronunciation": pronunciation,
        "audio": audio,
        "definitions": definitions or short_definitions,
        "short_definitions": short_definitions,
        "examples": ex_clean[:5],
        "etymology": etymology,
        "first_known_use": first_use,
        "inflections": inflections,
        "related_forms": related_forms,
        "quotes": quotes[:3],
        "dict_found": True,
    }


def parse_thesaurus(data, word):
    """Synonyms/antonyms/related from a raw Thesaurus response (no network)."""
    out = {"synonyms": [], "antonyms": [], "related_words": [], "near_antonyms": []}
    if not data or not isinstance(data[0], dict):
        return out

    def _extend(dst, groups):
        for group in groups or []:
            for item in group:
                w = item.get("wd") if isinstance(item, dict) else item
                if w and w not in dst:
                    dst.append(w)

    for e in _matching_entries(data, word):
        meta = e.get("meta", {})
        _extend(out["synonyms"], meta.get("syns", []))
        _extend(out["antonyms"], meta.get("ants", []))
        # richer related/near-antonym groups live inside def -> sseq -> sense
        for section in e.get("def", []):
            for seq in section.get("sseq", []):
                for node in seq:
                    if isinstance(node, list) and len(node) == 2 and node[0] == "sense":
                        sense = node[1]
                        _extend(out["related_words"], sense.get("rel_list", []))
                        _extend(out["near_antonyms"], sense.get("near_list", []))
    return out


def fetch_dict_data(word):
    """Fetch and merge the Dictionary + Thesaurus records for `word`."""
    entry = parse_dict(_mw_get(f"{MW_DICT}/{word}", MW_DICT_KEY, word), word)
    if not entry.get("dict_found"):
        return {"dict_found": False}
    entry.update(parse_thesaurus(_mw_get(f"{MW_THES}/{word}", MW_THES_KEY, word), word))
    return entry


def enrich(word):
    print(f"  enriching: {word}")
    entry = fetch_dict_data(word)
    time.sleep(0.15)
    entry["enriched"] = bool(entry.get("dict_found"))
    if not entry["enriched"]:
        print(f"    no MW entry for '{word}' — skipping")
    return entry


# --- Pruning: full MW record -> lean, card-ready entry ---
# Keep only what a recall card needs; the rest stays in words.full.json.
def prune_entry(v):
    if not v.get("dict_found"):
        return None
    examples = v.get("examples") or ([v["example"]] if v.get("example") else [])
    return {
        "word": v.get("word", ""),
        "pronunciation": v.get("pronunciation", ""),
        "audio": v.get("audio", ""),
        "part_of_speech": v.get("part_of_speech", []),
        # concise sense list (shortdef), not the verbose numbered senses
        "definition": (v.get("short_definitions") or v.get("definition") or [])[:3],
        "example": examples[0] if examples else "",
        # MW lists synonyms primary-sense-first; cap to drop off-sense spillover.
        "synonyms": v.get("synonyms", [])[:10],
        "antonyms": v.get("antonyms", [])[:8],
        "etymology": v.get("etymology", ""),
    }


def write_lean(full_cache):
    """Derive the lean words.json (card/viz input) from the full MW cache."""
    lean = {w: prune_entry(v) for w, v in full_cache.items()}
    lean = {w: e for w, e in lean.items() if e}  # drop words MW had no entry for
    with open(CACHE_FILE, "w") as f:
        json.dump(lean, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(lean)} lean entries to {CACHE_FILE}")
    return lean


def _section(label, value):
    """Render an answer-card section, or nothing when the value is empty."""
    if not value:
        return ""
    return (
        f'<div style="font-size:0.8em;font-style:italic;color:#bbb;margin:0 0 6px">{label}</div>'
        f'<div style="font-size:0.92em;line-height:1.6;color:#444;margin-bottom:16px">{value}</div>'
    )


CARD_MODEL = genanki.Model(
    1607392320,
    "MW Vocabulary",
    fields=[
        {"name": "Word"},
        {"name": "Pronunciation"},
        {"name": "PartOfSpeech"},
        {"name": "Definition"},
        {"name": "Example"},
        {"name": "Synonyms"},
        {"name": "Antonyms"},
        {"name": "Etymology"},
    ],
    templates=[{
        "name": "Card 1",
        "qfmt": """
<div style="font-family:Georgia,serif;font-size:2em;font-weight:700;margin-bottom:4px">{{Word}}</div>
<div style="font-size:0.85em;color:#999;font-style:italic">{{Pronunciation}} &nbsp; {{PartOfSpeech}}</div>
{{tts en_US:Word}}
""",
        "afmt": """
<div style="font-family:Georgia,'Times New Roman',serif;max-width:560px;margin:0 auto;padding:8px;text-align:left">
  <div style="font-size:2em;font-weight:700;margin-bottom:2px">{{Word}}</div>
  <div style="font-size:0.85em;color:#999;font-style:italic;margin-bottom:18px">{{Pronunciation}} &nbsp; {{PartOfSpeech}}</div>
  {{tts en_US:Word}}
  <hr style="border:none;border-top:1px solid #e5e5e5;margin:0 0 16px">
  {{Definition}}
  {{Example}}
  {{Synonyms}}
  {{Antonyms}}
  {{Etymology}}
</div>
""",
    }],
)


def build_deck(cache):
    """Build the Anki deck from the lean cache (entries already pruned)."""
    deck = genanki.Deck(2059400110, "Merriam-Webster Saved Words")
    for word, data in cache.items():
        definition = "<br>".join(data.get("definition", []))
        example = data.get("example", "")
        example_html = f'<span style="font-style:italic;color:#666">“{example}”</span>' if example else ""
        note = genanki.Note(
            model=CARD_MODEL,
            fields=[
                data.get("word", word),
                data.get("pronunciation", ""),
                ", ".join(data.get("part_of_speech", [])),
                _section("Definition", definition),
                _section("Example", example_html),
                _section("Synonyms", ", ".join(data.get("synonyms", []))),
                _section("Antonyms", ", ".join(data.get("antonyms", []))),
                _section("Etymology", f'<span style="font-style:italic;color:#777">{data.get("etymology","")}</span>'),
            ],
        )
        deck.add_note(note)
    genanki.Package(deck).write_to_file(OUTPUT_DECK)
    print(f"Wrote {len(deck.notes)} cards to {OUTPUT_DECK}")


ENRICH_ONLY = "--enrich-only" in sys.argv
PRUNE_ONLY = "--prune" in sys.argv


def _load_full_cache():
    """Load the full MW cache, falling back to a legacy words.json."""
    for path in (FULL_CACHE, CACHE_FILE):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def main():
    cache = _load_full_cache()

    # --prune: just re-derive the lean words.json + deck from existing full data.
    if PRUNE_ONLY:
        build_deck(write_lean(cache))
        return

    if ENRICH_ONLY:
        words = list(cache.keys())
    else:
        words = fetch_all_words()

    needs_enrich = [w for w in words if w not in cache or not cache[w].get("enriched")]

    print(f"{len(needs_enrich)} words to enrich (skipping {len(words)-len(needs_enrich)} cached)")

    for word in needs_enrich:
        cache[word] = enrich(word)
        with open(FULL_CACHE, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    build_deck(write_lean(cache))


if __name__ == "__main__":
    main()
