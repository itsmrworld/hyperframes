#!/usr/bin/env python3
"""Word timings. Aligned to the real ElevenLabs take: pause-detected phrase windows, words spread by syllable. Voice starts at 4.5 s.."""
import json, re
segs = [
 (4.71, 5.82, 'Before the logo.'),
 (6.29, 7.45, 'Before the website.'),
 (7.84, 9.46, 'Before the first customer,'),
 (10.28, 11.08, 'there was a name.'),
 (11.61, 12.64, "Now it's your turn."),
 (13.04, 13.64, 'Type it in.'),
 (13.92, 14.82, 'See it appear.'),
 (15.20, 15.74, 'Live.'),
 (16.05, 18.28, 'Dot shop, one dollar twelve a year.'),
 (18.51, 20.37, 'Dot online, one forty-six.'),
 (20.58, 22.93, 'Dot xyz, two fifty-five.'),
 (23.17, 24.84, 'Dot org, eight forty-nine.'),
 (25.14, 25.77, 'Dot com,'),
 (25.93, 27.13, 'eleven twenty-six.'),
 (27.39, 28.14, 'Add to cart.'),
 (28.51, 28.85, 'Pay.'),
 (29.28, 30.24, 'And just like that,'),
 (30.50, 31.59, 'the name is yours.'),
 (31.89, 32.79, 'Then give it a home,'),
 (33.03, 34.58, 'and go live the same day.'),
 (34.86, 35.96, 'SatisfyCloud.'),
 (36.28, 38.39, 'Claim the name your brand deserves.'),
]
def syl(w): return max(1, len(re.findall(r'[aeiouy]+', re.sub(r'[^a-z]', '', w.lower()))))
words, lines = [], []
for a, b, t in segs:
    toks = t.split(); wt = [syl(w) + (1.2 if re.search(r'[.?,]$', w) and i < len(toks) - 1 else 0) for i, w in enumerate(toks)]
    tot = sum(wt); cur = a; first = len(words)
    for w, x in zip(toks, wt):
        d = (b - a) * x / tot; words.append({"text": w, "start": round(cur, 3), "end": round(cur + max(d - (0.1 if x > syl(w) else 0), 0.1), 3)}); cur += d
    lines.append({"text": t, "start": a, "end": b, "first": first, "last": len(words) - 1})
json.dump({"duration": 43, "words": words, "lines": lines}, open("data/words.json", "w"), indent=1); print(len(words), "words")
