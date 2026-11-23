import json,re,html
d=json.load(open("data/words.json")); W=d["words"]; lines=d["lines"]
chunks=[]
for ln in lines:
    ws=W[ln["first"]:ln["last"]+1]
    # split at sentence ends, then at <=6 words
    cur=[]
    parts=[]
    for w in ws:
        cur.append(w)
        if re.search(r'[.?!]$',w["text"]) and not w["text"].endswith("..."): parts.append(cur);cur=[]
    if cur: parts.append(cur)
    for p in parts:
        while len(p)>6:
            parts2=p[:5]; chunks.append(parts2); p=p[5:]
        chunks.append(p)
# drop captions that the CTA visuals already carry
chunks=[c for c in chunks if c[0]["start"]<68.85]
out=[];js=[]
for i,c in enumerate(chunks):
    s=c[0]["start"]-0.04; e=c[-1]["end"]+0.22
    nxt=chunks[i+1][0]["start"]-0.04 if i+1<len(chunks) else 99
    e=min(e,nxt-0.02)
    spans="".join(f'<span class="w" id="cw{i}_{j}">{html.escape(w["text"])}</span> ' for j,w in enumerate(c))
    light=' lt' if c[0]["start"]>=64.5 else ''
    out.append(f'<div id="cap{i}" class="clip cap{light}" data-start="{s:.3f}" data-duration="{e-s:.3f}" data-track-index="1"><div class="ci">{spans.strip()}</div></div>')
    js.append({"i":i,"s":round(s,3),"d":round(e-s,3),"w":[round(w["start"]-s,3) for w in c]})
tpl=open("scripts/captions.tpl.html").read()
open("compositions/captions.html","w").write(tpl.replace("{{CAPS}}","\n      ".join(out)).replace("{{DATA}}",json.dumps(js)))
print(len(chunks),"caption chunks")
