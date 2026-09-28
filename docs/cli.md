# Command line

Install with `pip install "hone-taste[cli]"`. Exit codes: 0 ok, 1 when a file could not be scored, 2 for
usage or configuration errors.

```bash
hone-taste score --scorer slop --domain lyrics lyrics/*.txt     # value, file, reason per line
hone-taste score --scorer slop --json draft.md                  # JSON rows: file, value, reason, error
hone-taste score --scorer audiobox --json song.wav              # extra "songs"
hone-taste score --scorer songeval --accept-license song.wav    # read the license note first
hone-taste profile ask --name ana --items lyrics/ --budget 5 # asks about the closest calls
hone-taste agreement --name ana --scorer slop --json         # how often slop agrees with your picks
hone-taste models                                               # models, whose taste, licenses
```

Scorers: `slop` and `binoculars` read text files; `songeval` and `audiobox` take audio paths.
`profile ask` compares the `.txt` / `.md` files in `--items` and needs a terminal (it refuses to run
when stdin is not a TTY; use `profile.ask(pairs, io=...)` from Python instead).

Example: [`command_line.py`](../examples/command_line.py) runs each command in a subprocess, the way a
script or CI job would.
