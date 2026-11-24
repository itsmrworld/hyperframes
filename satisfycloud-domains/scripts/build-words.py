#!/usr/bin/env python3
"""Word timings. ESTIMATE until the real ElevenLabs file is aligned (see align-voice.py)."""
import json, re
segs = [
 (4.5, 7.0, "Every story begins with a name."),
 (7.6, 12.6, "A late night. A blank screen. One idea you can't let go."),
 (13.1, 14.4, "So you search."),
 (14.9, 17.4, "And your name appears. Live."),
 (17.9, 21.0, "Dot shop, one dollar twelve a year."),
 (21.2, 23.6, "Dot online, one forty-six."),
 (23.8, 26.0, "Dot xyz, two fifty-five."),
 (26.3, 28.8, "Dot org, eight forty-nine."),
 (29.0, 31.6, "Dot com, eleven twenty-six."),
 (32.2, 34.0, "Add to cart. Pay."),
 (34.5, 36.3, "The name is yours."),
 (36.9, 40.8, "Then give it a home, and go live the same day."),
 (41.4, 42.4, "SatisfyCloud."),
 (42.8, 45.6, "Claim the name your brand deserves."),
]
def syl(w): return max(1, len(re.findall(r'[aeiouy]+', re.sub(r'[^a-z]', '', w.lower()))))
words, lines = [], []
for a, b, t in segs:
    toks = t.split(); wt = [syl(w) + (1.2 if re.search(r'[.?,]$', w) and i < len(toks) - 1 else 0) for i, w in enumerate(toks)]
    tot = sum(wt); cur = a; first = len(words)
    for w, x in zip(toks, wt):
        d = (b - a) * x / tot; words.append({"text": w, "start": round(cur, 3), "end": round(cur + max(d - (0.1 if x > syl(w) else 0), 0.1), 3)}); cur += d
    lines.append({"text": t, "start": a, "end": b, "first": first, "last": len(words) - 1})
json.dump({"duration": 50, "words": words, "lines": lines}, open("data/words.json", "w"), indent=1); print(len(words), "words")
