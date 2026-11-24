# SatisfyCloud domains ad (16:9, ~50 s)

Story ad for top-level domains: a name, a search, five low prices (USD), the name becomes yours, the site goes live.
Built from the real site's look (screenshots of satisfycloud.com: dark UI, `#0B64F8` buttons, glass cards, domain search card, pricing table).

| Time          | Scene                                                      | File                      |
| ------------- | ---------------------------------------------------------- | ------------------------- |
| 0 - 7 s       | Animation only, no voice                                   | `compositions/intro.html` |
| 7 - 12.6 s    | The story: a late night, a blank screen, one idea          | `story.html`              |
| 12.6 - 17.9 s | Homepage, dive into the search bar, type the name, results | `search.html`             |
| 17.9 - 32 s   | Pricing table, one row per voice cue, big USD price        | `prices.html`             |
| 32 - 36.9 s   | Add to cart, pay, the name is yours                        | `checkout.html`           |
| 36.9 - 41.2 s | Add hosting, the site goes live                            | `live.html`               |
| 41.2 - 50 s   | Brand end card, URL, click                                 | `end.html`                |

- Script for ElevenLabs: `SCRIPT.md`. The voice starts at 4.5 s.
- Prices come from satisfycloud.com/domain (Popular tab). Update them in `compositions/prices.html` if they change.
- Art: crops from the site's own images in `assets/img/` (low resolution; swap in the originals).
- `python3 scripts/make-square.py` generates the 1:1 project; `python3 scripts/score.py` rebuilds the score.
- Captions come from `data/words.json` (`scripts/build-words.py`, estimated until the real voice is aligned) via `scripts/build-captions.py`.
