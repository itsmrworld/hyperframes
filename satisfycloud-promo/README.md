# SatisfyCloud — welcome / promo film (16:9)

72 s English film built with HyperFrames. A 1:1 sibling lives in `../satisfycloud-promo-square`
and is **generated** from this project (`python3 scripts/make-square.py`), so edit here, then regenerate.

## What is in it

| Time          | Scene                                                                      | File                        |
| ------------- | -------------------------------------------------------------------------- | --------------------------- |
| 0 – 14.4 s    | 3 a.m. hook: a person, a screen, the idea (the orb) is born                | `compositions/hook.html`    |
| 14.4 – 24.9 s | The problem: slow, crashing, complicated. The orb goes cold and sinks      | `compositions/problem.html` |
| 24.9 – 28.6 s | Arrival: a held breath, ignition, the logo lands on "This is SatisfyCloud" | `compositions/arrive.html`  |
| 28.6 – 58.2 s | The lineup: nine glass cards, one continuous push through them             | `compositions/lineup.html`  |
| 58.2 – 64.5 s | Use cases: fintech, e-commerce, SaaS, multiplayer, joined by one globe     | `compositions/worlds.html`  |
| 64.5 – 74 s   | End card: brand, "Build without limits.", address, click                   | `compositions/cta.html`     |

Shared layers: `bg.html` (galaxy canvas, palette script), `fx.html` (vignette, grain), `captions.html`
(generated from `data/words.json` by `scripts/build-captions.py`). The orb is one element in `index.html`
that carries the story across scenes.

No price appears anywhere. Product and series names are the official lineup from the brief.

## Things you will want to swap

- **Portraits.** `assets/img/portrait*.jpg` are a **placeholder cropped from the reference screenshot**.
  Replace with your own licensed or generated portraits (same lighting: royal-blue backdrop, warm rim light).
  `portrait.jpg` (original grade) is used on the end card; `portrait_night.jpg` (darker, mirrored) in the hook.
- **Logo.** `assets/logo/*` are keyed from your four logo files (white, blue and dark-text variants, plus the
  mark and wordmark split). White variants sit on the galaxy, blue/dark on the white end card.
- **Voice.** `assets/audio/voice.mp3` is the ElevenLabs take (**not committed**: put your file there before rendering). `voice_master.wav` is the same voice with a gentle
  EQ, compressor and loudness match (-17.5 LUFS). If you re-generate the voice, re-time:
  1. update the segment list in the Python block that produced `data/words.json` (pause-detected word timing),
  2. `python3 scripts/build-captions.py`,
  3. nudge the cue times in the scene files and in `scripts/score.py`.

## Sound

`scripts/score.py` synthesises the whole score deterministically (numpy only, no samples): a sub drone and heartbeat
in the hook, a cold dissonant bed under the problem, a held breath of near-silence before arrival, a
Shepard-Risset rising tone through the lineup (the "ever ascending" feeling), plucks, whooshes on each push-through,
UI ticks and clicks on every on-screen action, long reverb, and sidechain ducking driven by the voice.

```bash
scripts/master-voice.sh              # writes assets/audio/voice_master.wav (git-ignored, regenerable)
python3 scripts/score.py            # writes assets/audio/score.wav (git-ignored, regenerable)
```

It was measured, not heard: mix about -16.5 LUFS, peak about -2 dBFS. If you have a licensed track you prefer,
drop it in `assets/audio/` and point the `#score` audio tag in `index.html` at it.

## Render

```bash
npx hyperframes lint && npx hyperframes check
npx hyperframes render -o renders/satisfycloud_16x9.mp4
```
