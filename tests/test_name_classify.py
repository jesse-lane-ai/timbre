"""Test suite for clips.name_classify.classify_from_names.

Run directly:
    .venv/bin/python tests/test_name_classify.py

Prints a table of path -> parsed fields, then a PASS/FAIL summary.
"""

from __future__ import annotations

import sys
import os

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from timbre.names import classify_from_names

# ---------------------------------------------------------------------------
# Test cases: (description, path, expected_fields)
# expected_fields: dict of field -> expected value (None means "should be None")
# ---------------------------------------------------------------------------

CASES = [
    # 1. Folder-path: key + scale + bpm from parent dirs; category from filename
    (
        "House Loops / Cm 124bpm / bassline",
        "/Samples/House Loops/Cm 124bpm/bassline.wav",
        {"kind": "loop", "key": "C", "scale": "minor", "bpm": 124.0, "category": "bass"},
    ),
    # 2. Oneshots folder + kick keyword in filename
    (
        "Oneshots/kick_punch",
        "/Packs/Oneshots/kick_punch.wav",
        {"kind": "one-shot", "category": "kick"},
    ),
    # 3. Drum Loops folder provides kind; bpm from subfolder
    (
        "Drum Loops / 140bpm / clap loop",
        "/Samples/Drum Loops/140bpm/clap loop.wav",
        {"kind": "loop", "bpm": 140.0, "category": "clap"},
    ),
    # 4. Synth Leads / Am / stab → key+scale from folder, kind from filename keyword
    (
        "Synth Leads / Am / synth_lead_stab",
        "/Library/Synth Leads/Am/synth_lead_stab.wav",
        {"kind": "one-shot", "key": "A", "scale": "minor", "category": "lead"},
    ),
    # 5. G_maj_pad_80bpm — all in filename
    (
        "G_maj_pad_80bpm",
        "/Library/Pads/G_maj_pad_80bpm.wav",
        {"key": "G", "scale": "major", "bpm": 80.0, "category": "pad"},
    ),
    # 6. fx_riser_oneshot_Fm — kind + key + scale + category all in filename
    (
        "fx_riser_oneshot_Fm",
        "/Library/fx_riser_oneshot_Fm.wav",
        {"kind": "one-shot", "key": "F", "scale": "minor", "category": "fx"},
    ),
    # 7a. BPM format: "128bpm"
    (
        "BPM format 128bpm",
        "/Samples/loop_128bpm.wav",
        {"bpm": 128.0},
    ),
    # 7b. BPM format: "bpm128"
    (
        "BPM format bpm128",
        "/Samples/bpm128_loop.wav",
        {"bpm": 128.0},
    ),
    # 7c. BPM format: bare number "_90_"
    (
        "BPM format _90_",
        "/Samples/loop_90_beats.wav",
        {"bpm": 90.0},
    ),
    # 7d. BPM format: "120 BPM" in folder name
    (
        "BPM format '120 BPM' folder",
        "/Samples/120 BPM/lead.wav",
        {"bpm": 120.0},
    ),
    # 8. AceStep fallback guard: a generic filename with no recognisable tokens
    #    should produce low confidence (< 0.5), signalling the fallback should fire.
    (
        "Low-confidence path (should trigger AceStep fallback)",
        "/Library/Pack1/track01.wav",
        {"_low_confidence": True},  # special sentinel
    ),
    # 9. Sharpened root
    (
        "F# minor loop",
        "/Packs/F#m Loops/bass_loop.wav",
        {"key": "F#", "scale": "minor", "kind": "loop"},
    ),
    # 10. Flat root normalised to sharp
    (
        "Db major (-> C# major)",
        "/Packs/Db major/synth_chord.wav",
        {"key": "C#", "scale": "major", "category": "chord"},
    ),
    # 11. "s"=sharp convention (Loopmasters/Splice), no scale token
    (
        "Ds (-> D#) via trailing-s sharp",
        "/Packs/Hip Hop/87_Ds_GoodlifeSoulString_SP.wav",
        {"key": "D#", "bpm": 87.0},
    ),
    # 12. "s"=sharp glued to scale keyword
    (
        "Asmin (-> A# minor)",
        "/Packs/120_Asmin_thing.wav",
        {"key": "A#", "scale": "minor", "bpm": 120.0},
    ),
    # 13. trailing-s sharp must NOT fire on a real word ("Bass")
    (
        "Bass is not a B# key",
        "/Packs/Bass_loop.wav",
        {"key": None, "category": "bass"},
    ),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

COL_W = 42


def _check(result: dict, expected: dict) -> tuple[bool, list[str]]:
    failures = []
    for field, want in expected.items():
        if field == "_low_confidence":
            got_conf = result.get("confidence", 1.0)
            if got_conf >= 0.5:
                failures.append(f"confidence={got_conf:.3f} (expected < 0.5 for AceStep fallback trigger)")
        else:
            got = result.get(field)
            if got != want:
                failures.append(f"{field}: got {got!r}, want {want!r}")
    return (len(failures) == 0), failures


def main():
    passed = 0
    failed = 0
    rows = []

    for desc, path, expected in CASES:
        result = classify_from_names(path)
        ok, failures = _check(result, expected)

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        rows.append((status, desc, result, failures))

    # Print table header
    sep = "-" * 110
    print(sep)
    print(f"{'ST':4} {'Description':<{COL_W}} {'kind':<10} {'cat':<10} {'key':<5} {'scale':<7} {'bpm':<7} {'conf':<6}")
    print(sep)

    for status, desc, result, failures in rows:
        mark = "✓" if status == "PASS" else "✗"
        print(
            f"{mark} {status}  {desc:<{COL_W}}"
            f"  {str(result.get('kind') or ''):<10}"
            f"  {str(result.get('category') or ''):<10}"
            f"  {str(result.get('key') or ''):<5}"
            f"  {str(result.get('scale') or ''):<7}"
            f"  {str(result.get('bpm') or ''):<7}"
            f"  {result.get('confidence', 0):.3f}"
        )
        for f in failures:
            print(f"       ↳ {f}")

    print(sep)
    print(f"Results: {passed} passed, {failed} failed  ({len(CASES)} total)")

    # AceStep fallback guard — verify that the low-confidence case would trigger
    # the fallback (confidence < 0.5 is the threshold checked in _fuse_category).
    print()
    print("AceStep fallback trigger guard:")
    low_conf_path = "/Library/Pack1/track01.wav"
    r = classify_from_names(low_conf_path)
    threshold = 0.5
    triggers = r["confidence"] < threshold
    print(f"  path={low_conf_path!r}  confidence={r['confidence']:.3f}  triggers_fallback={triggers}")
    if triggers:
        print("  → name_classify returned low confidence; AceStep would be invoked (guarded, not actually called here)")
    else:
        print("  → WARNING: confidence unexpectedly high; check test case")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
