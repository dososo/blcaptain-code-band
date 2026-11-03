# Code Band

[中文](README.md) · **English**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> An agent skill for Claude Code and Codex that writes instrumental background music as code: the agent composes a score, a local sampler plays real recorded instruments note by note, and you get an **editable score project** plus a **WAV**. No generative audio model, no uploads.

**▶ Listen**: <https://dososo.github.io/blcaptain-code-band/site/>. Eight pieces, covering Chinese–Western ensemble, guqin, guzheng, orchestra, piano and electronic.

## What it does

- From a one-line brief, the agent writes theme, harmony, orchestration and sections, then renders 48 kHz / 24-bit WAV through FluidSynth.
- Real instruments: MuseScore General plus 60 extension sample packs: VSCO 2 orchestra, piano, nylon guitar, guqin, guzheng, erhu, pipa, dizi, xiao, sheng, Peking-opera percussion and more. Per-pack licenses are listed in [docs/third-party-soundfonts.md](docs/third-party-soundfonts.md).
- `arrange-check` verifies an arrangement plan against the score (call-and-response collisions, a single peak, motif recognition).
- Targeted edits (gain, shorten, loop, add a part) come with score and event diffs that prove nothing else changed.

## Limits

- Writing "in the style of" a reference track has **not** passed human listening tests.
- Several Chinese instruments (suona, liuqin, sanxian, zhongruan) have no usable samples in this release; the skill reports them as missing instead of substituting.
- The Surge XT plugin route from the development version depends on GPL components and is not included in v1.
- Fully tested on macOS (Apple Silicon) only.

## Install

```bash
brew install fluid-synth ffmpeg
git clone https://github.com/dososo/blcaptain-code-band.git && cd blcaptain-code-band
python3 -m venv .venv && .venv/bin/python -m pip install -r skills/code-band-bgm/requirements.txt
python3 scripts/fetch_soundfonts.py          # ~3 GB from the GitHub Release, SHA-256 verified
python3 skills/code-band-bgm/scripts/install_skill.py install --host both --project /path/to/your/project
```

Claude Code plugin: `/plugin marketplace add dososo/blcaptain-code-band`, then `/plugin install code-band@blcaptain-code-band` (not yet tested before release).

## License

Code and docs: [Apache License 2.0](LICENSE); see [NOTICE](NOTICE). Sample packs keep their own upstream licenses (CC0, CC BY 3.0/4.0, MIT, Iowa MIS terms); works rendered with CC BY packs must keep the attribution files the skill copies into each output folder.
