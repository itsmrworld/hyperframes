#!/usr/bin/env python3
"""Derive the 1080x1080 (1:1) project from the 16:9 one. Scenes adapt with @container (aspect-ratio) rules."""
import shutil, os, re, sys
src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dst = src + "-square"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".hyperframes", "renders", "node_modules", "__pycache__"))
def sub(path, pairs):
    s = open(path).read()
    for a, b in pairs: s = s.replace(a, b)
    open(path, "w").write(s)
for root, _, files in os.walk(os.path.join(dst, "compositions")):
    for f in files:
        if f.endswith(".html"): sub(os.path.join(root, f), [('data-width="1920" data-height="1080"', 'data-width="1080" data-height="1080"')])
sub(os.path.join(dst, "index.html"), [
    ('data-width="1920" data-height="1080"', 'data-width="1080" data-height="1080"'),
    ('content="width=1920, height=1080"', 'content="width=1080, height=1080"'),
    ("width: 1920px; height: 1080px;", "width: 1080px; height: 1080px;"),
    ("var W = 1920, H = 1080, sq = false;", "var W = 1080, H = 1080, sq = true;"),
])
sub(os.path.join(dst, "meta.json"), [("satisfycloud-promo", "satisfycloud-promo-square")]) if os.path.exists(os.path.join(dst, "meta.json")) else None
print("square project at", dst)
