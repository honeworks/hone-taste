# Third-party notices

hone-taste is licensed under the Apache License 2.0 (see `LICENSE`). This file lists the third-party
material it includes or uses. **No model weights are included**: each model is downloaded from its upstream
source at first use, under that source's terms. Several of those terms are non-commercial or unclear
(SongEval, MuQ, PickScore); `accept_license=True` only confirms that you have checked them yourself.
Users are responsible for checking and complying with the licence of every model they use.

## slop-score word and trigram lists
`src/hone_taste/data/slop/*.json` come from https://github.com/sam-paech/slop-score (MIT License,
Copyright (c) Sam Paech). See `src/hone_taste/data/slop/NOTICE.md`.

## SongEval scoring head
`src/hone_taste/scorers/songeval.py` (`_songeval_head`) re-implements the `Generator` module of
https://github.com/ASLP-lab/SongEval (`model.py`, sizes from `config.yaml`) so that SongEval's published
checkpoint can be loaded. The upstream licensing is not consistent: the repository metadata names the
Apache License 2.0, while its README states that the project is released under CC BY-NC-SA 4.0
(non-commercial). The checkpoint itself is not redistributed; it is downloaded from the upstream
repository at first use, and `tt.songeval(...)` requires `accept_license=True`. The MuQ encoder it uses
(OpenMuQ/MuQ-large-msd-iter) is licensed CC BY-NC 4.0 (non-commercial).

## Audiobox Aesthetics
`tt.audiobox(...)` (extra `songs`) uses the `audiobox-aesthetics` package and its model
(https://github.com/facebookresearch/audiobox-aesthetics, Meta Platforms, Inc.), licensed CC BY 4.0 for
code and weights. That licence requires attribution: credit Meta's Audiobox Aesthetics (arXiv:2502.05139)
when you publish results made with it. CC BY 4.0 is not an OSI software licence.

## MuQ (through the `muq` package)
The `songs` extra installs `muq` (code MIT). Its weights, used by SongEval (see above), are CC BY-NC 4.0:
non-commercial use only.

## PickScore
`tt.image_preference("pickscore")` (extra `images`) downloads yuvalkirstain/PickScore_v1
(https://huggingface.co/yuvalkirstain/PickScore_v1) through `transformers`. Its model card states **no
licence**, so its terms of use are unknown; `tt.image_preference("pickscore", ...)` requires
`accept_license=True`. Nothing from PickScore is included in hone-taste.

## Skywork-Reward-V2, Binoculars models, DINOv2, HPSv3
Downloaded at first use, not included: Skywork/Skywork-Reward-V2-Qwen3-0.6B (Apache-2.0),
Qwen/Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct (Apache-2.0; the Binoculars method is BSD-3-Clause),
facebook/dinov2-small (Apache-2.0), MizzenAI/HPSv3 (Apache-2.0 weights, MIT code; installed separately
with `pip install hpsv3`).

## Copyleft components in optional extras
None of these are included in or redistributed with hone-taste; they are installed only with an optional
extra, and hone-taste imports them at run time:

| Package | Licence | Pulled in by |
|---|---|---|
| easydict | LGPL-3.0 | extra `songs` (through `muq`) |
| soxr | LGPL-2.1-or-later | extra `songs` (through `librosa`) |

The core package (standard library and pydantic) has no copyleft dependencies.

## Packaged reference set
`src/hone_taste/references/text/reward_model.json` holds numbers only: raw scores that the default reward
model gave to hand-written example answers (written for this project; see
`scripts/make_reference_sets.py`). It contains no model weights and no third-party text.

## Model weights
No model weights are included in this package. Each taste model downloads its weights from the upstream
source at first use; `hone-taste models` and the README list each model's licence as checked on
2026-09-27. Upstream licences can change; check the model card before relying on them.
