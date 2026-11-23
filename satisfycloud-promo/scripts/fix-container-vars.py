#!/usr/bin/env python3
"""A container query can't restyle its own container, so layout variables live on each scene's stage child, not on #root."""
import sys, os
d = sys.argv[1]
def patch(f, pairs):
    p = os.path.join(d, "compositions", f); s = open(p).read()
    for a, b in pairs:
        if a not in s: print("MISSING in", f, ":", a[:70]); continue
        s = s.replace(a, b, 1)
    open(p, "w").write(s)
patch("hook.html", [
 ("color: #fff;\n      --pw: 1187px; --ph: 2400px; --pl: 930px; --pt: -1064px; }", "color: #fff; }\n    #hook-stage { --pw: 1187px; --ph: 2400px; --pl: 930px; --pt: -1064px; }"),
 ("#root { --pw: 742px; --ph: 1500px; --pl: 330px; --pt: -440px; }", "#hook-stage { --pw: 742px; --ph: 1500px; --pl: 330px; --pt: -440px; }")])
patch("lineup.html", [
 ("color: #fff;\n      --cl: 350px; --ct: 160px; --cw: 1220px; --ch: 690px; --ey: 92px; }", "color: #fff; }\n    #ln-stage { --cl: 350px; --ct: 160px; --cw: 1220px; --ch: 690px; --ey: 92px; }"),
 ("#root { --cl: 50px; --ct: 190px; --cw: 980px; --ch: 760px; --ey: 100px; }", "#ln-stage { --cl: 50px; --ct: 190px; --cw: 980px; --ch: 690px; --ey: 100px; }")])
patch("arrive.html", [("color: #fff; --lw: 1100px; }", "color: #fff; }\n    #ar-stage { --lw: 1100px; }"), ("@container (max-aspect-ratio: 1.2) { #root { --lw: 900px; }", "@container (max-aspect-ratio: 1.2) { #ar-stage { --lw: 900px; }")])
patch("cta.html", [("color: #0B1220; --lw: 880px; }", "color: #0B1220; }\n    #cta-stage { --lw: 880px; }"), ("#root { --lw: 800px; }", "#cta-stage { --lw: 800px; }")])
patch("captions.html", [("--fs: 56px; --bot: 118px; }", "}"), (".cap { position: absolute;", ".cap { --fs: 56px; --bot: 118px; position: absolute;"), ("#root { --fs: 48px; --bot: 70px; } .cap { padding: 0 70px; }", ".cap { --fs: 46px; --bot: 64px; padding: 0 70px; }")])
patch("../scripts/captions.tpl.html", [("--fs: 56px; --bot: 118px; }", "}"), (".cap { position: absolute;", ".cap { --fs: 56px; --bot: 118px; position: absolute;"), ("#root { --fs: 48px; --bot: 70px; } .cap { padding: 0 70px; }", ".cap { --fs: 46px; --bot: 64px; padding: 0 70px; }")])
print("patched", d)
