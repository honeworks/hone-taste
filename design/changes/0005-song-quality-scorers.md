# 0005: Song quality scorers: long songs without running out of memory, objective audio checks

## Status

`implemented in 0.1.0` for options 2 and 3 (approved with the other demo-app records, 2026-09-28;
renumbered from a second `0004`). `songeval(max_seconds=...)` scores a middle excerpt and halves it after
an out-of-memory error (no `on_oom` switch: halving is the only strategy; without `max_seconds` nothing
changes), and
`tt.audio_checks` / `tt.audio_report` port OneShotStudio's measurements. Option 4 (MuQ-Eval, phoneme
error rate) waits for the owner's GPU run, as recommended. The real SongEval excerpt path is untested on
a GPU so far.

## Context

Found in OneShotStudio's quality overhaul (the OneShotStudio demo app, its change 0002, REPORT.md §6-7). The app
scores every song take with `tt.audiobox()` and `tt.songeval(accept_license=True)`.

- **SongEval ran out of GPU memory on 3 of 6 four-minute takes** on an 8 GB card, even after the app
  freed Whisper first: MuQ's features for a whole song are large. The app now wraps the scorer
  (`oneshot_studio/music.py`, `excerpted`): it scores a 90 s middle excerpt with every other model
  released, and after an out-of-memory error retries on half the length (down to 30 s); `details`
  record `excerpt_s`.
- **"Does this song annoy the ear?" needed objective checks** that no hone-taste scorer gives:
  loudness (BS.1770 integrated LUFS, EBU loudness range), true peak, clipping, harshness (2-5 kHz vs
  the body), sibilance, mud, spectral centroid, glitches (spectral-flux spikes on noise-like frames),
  and tempo / key / meter against a request. The app wrote them on numpy / scipy / librosa
  (`oneshot_studio/audio_qa.py`, ~370 lines, CPU, ~3 s per song) with pass / fail limits.
- Two more scorers were planned and not built: **MuQ-Eval** (a per-clip music quality predictor,
  reported CC BY 4.0) and **phoneme error rate** for sung intelligibility (the app uses Whisper word
  recall today).

## Problem

- Any app scoring whole songs with SongEval on a small GPU hits the same OOM; the fix (excerpt,
  release, halve on OOM) is generic.
- The objective checks are domain knowledge every audio app rewrites; their `None`-not-`0` handling
  and limits belong next to the other audio scorers.

## Options

1. **Leave both to applications** (today).
2. **`songeval(max_seconds=90, on_oom="halve")`**: the scorer itself scores an excerpt of long input
   and retries shorter after an OOM, recording the length used in `details`.
3. **`tt.audio_checks(limits=None, *, bpm=None, key=None, meter=None)`**: a no-model scorer (extra
   `songs`, numpy / scipy / librosa) whose `value` is the share of checks passed and whose `details`
   hold every measurement and verdict; `tt.audio_report(path)` for the raw measurements.
4. Later, with GPU tests: `tt.muq_eval()` and `tt.phoneme_error_rate(lyrics)` as `songs` scorers.

## Recommendation

2 and 3 now (the app's code and tests can move almost as they are), 4 after the owner's GPU turn
confirms the models and licences.

## Consequences

- OneShotStudio would drop `excerpted` and `audio_qa.py` for the package's versions.
- `audio_checks` limits are first guesses from nine songs; the package should document them as
  defaults to calibrate, not as truths.
