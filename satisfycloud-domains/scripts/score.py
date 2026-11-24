#!/usr/bin/env python3
"""SatisfyCloud score + sound design, synthesised deterministically (numpy only). Output: assets/audio/score.wav (48k stereo)."""
import numpy as np, subprocess, wave, sys
SR = 48000; DUR = 50.0; N = int(DUR * SR)
rng = np.random.default_rng(20261123)
t = np.arange(N) / SR

def env(pts):
    xs, ys = zip(*pts); return np.interp(t, xs, ys).astype(np.float32)
def smooth_env(x, ms):
    k = max(1, int(SR * ms / 1000)); w = np.hanning(k * 2 + 1); w /= w.sum()
    return np.convolve(x, w, mode="same").astype(np.float32)
def hz(note):  # "D3" -> Hz
    names = {"C":0,"C#":1,"D":2,"Eb":3,"E":4,"F":5,"F#":6,"G":7,"Ab":8,"A":9,"Bb":10,"B":11}
    n = note[:-1]; o = int(note[-1]); return 440.0 * 2 ** ((names[n] + 12 * (o + 1) - 69) / 12)
def pink(n):
    w = rng.standard_normal(n); W = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1 / SR); f[0] = 1; W /= np.sqrt(f / 100); return np.fft.irfft(W, n).astype(np.float32)
def place(buf, x, t0, pan=0.0):
    i = int(t0 * SR); 
    if i >= N: return
    j = min(N, i + len(x)); x = x[: j - i]
    gl = np.cos((pan + 1) * np.pi / 4); gr = np.sin((pan + 1) * np.pi / 4)
    buf[0, i:j] += x * gl; buf[1, i:j] += x * gr
def bandnoise(dur, fc0, fc1, bw=0.6, seed=0):
    """noise band-passed by an STFT mask whose centre glides exponentially fc0->fc1"""
    r = np.random.default_rng(seed); n = int(dur * SR); H = 1024; fr = 2 * H
    x = r.standard_normal(n + fr); out = np.zeros(n + fr); win = np.hanning(fr); fq = np.fft.rfftfreq(fr, 1 / SR)
    for s in range(0, n, H):
        k = s / max(n, 1); fc = fc0 * (fc1 / fc0) ** k
        m = np.exp(-0.5 * (np.log2(np.maximum(fq, 20) / fc) / bw) ** 2)
        out[s:s + fr] += np.fft.irfft(np.fft.rfft(x[s:s + fr] * win) * m, fr) * win
    return (out[:n] / (np.abs(out[:n]).max() + 1e-9)).astype(np.float32)
def adsr(n, a, d, s, r_start, r):
    e = np.ones(n, np.float32); na = int(a * SR); e[:na] = np.linspace(0, 1, na) if na else 1
    nr = int(r * SR); e[n - nr:] *= np.linspace(1, 0, nr) if nr else 1; return e

# ---------------------------------------------------------------- buses
music = np.zeros((2, N), np.float32)   # pads / drones / arps  (ducked by voice)
fx = np.zeros((2, N), np.float32)      # hits, whooshes, ticks   (ducked a little)
revsend = np.zeros((2, N), np.float32)

def pad(notes, t0, t1, fi, fo, level, bright=(0.6, 0.6), K=7, spread=0.8, seed=0):
    """additive pad: each note = 2 detuned copies (L/R decorrelated), slow amp LFO, partial count follows 'bright'"""
    r = np.random.default_rng(seed); i0 = int(t0 * SR); i1 = min(N, int(t1 * SR)); n = i1 - i0; tt = t[i0:i1]
    e = np.minimum(np.clip((tt - t0) / fi, 0, 1), np.clip((t1 - tt) / fo, 0, 1)) ** 1.5
    br = np.interp(tt, [t0, t1], bright)
    L = np.zeros(n, np.float32); R = np.zeros(n, np.float32)
    for f in notes:
        for side in (0, 1):
            det = 2 ** ((r.uniform(-7, 7)) / 1200); ph = r.uniform(0, 6.28); lfo = r.uniform(0.04, 0.13); lp = r.uniform(0, 6.28)
            sig = np.zeros(n, np.float32)
            for k in range(1, K + 1):
                a = (1.0 / k ** 1.35) * np.clip(br * K - (k - 1), 0, 1)
                if not np.any(a > 0): continue
                sig += (a * np.sin(2 * np.pi * f * det * k * tt + ph * k)).astype(np.float32)
            sig *= (0.78 + 0.22 * np.sin(2 * np.pi * lfo * tt + lp)).astype(np.float32)
            pw = 1.0 if side == 0 else 1.0
            (L if side == 0 else R)[:] += sig * (1 - 0.5 * spread * (side == 1)) 
            (R if side == 0 else L)[:] += sig * (0.5 * (1 - spread))
    music[0, i0:i1] += level * e * L / len(notes); music[1, i0:i1] += level * e * R / len(notes)

def shepard(t0, t1, level, rate_oct_per_s=1 / 14, fb=55.0, nl=9):
    i0 = int(t0 * SR); i1 = min(N, int(t1 * SR)); tt = t[i0:i1] - t0
    e = np.minimum(np.clip(tt / 4, 0, 1), np.clip((t1 - t0 - tt) / 4, 0, 1)) ** 2
    for side in (0, 1):
        sig = np.zeros(len(tt), np.float32)
        for k in range(nl):
            pos = (k + rate_oct_per_s * tt) % nl            # octaves above fb, wraps
            f = fb * 2 ** pos
            ph = 2 * np.pi * np.cumsum(f) / SR + k * 1.7 + side * 0.9
            a = np.exp(-0.5 * ((pos - nl / 2) / (nl / 4.2)) ** 2)
            sig += (a * np.sin(ph)).astype(np.float32)
        music[side, i0:i1] += level * e * sig / nl * 2.2

def pluck(f, t0, amp, pan=0.0, dec=0.9, send=0.9):
    n = int(min(dec * 4, 3.0) * SR); tt = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt * 9) + 0.12 * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt * 14)) * np.exp(-tt / dec * 2.2)
    x *= np.minimum(1, tt / 0.004); x = x.astype(np.float32) * amp
    place(music, x, t0, pan); place(revsend, x * send, t0, pan)

def tick(t0, f=2600, amp=0.10, pan=0.0, dec=0.03, send=0.35):
    n = int(0.16 * SR); tt = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * tt) * np.exp(-tt / dec) + 0.5 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt / (dec * 0.5)))
    x += rng.standard_normal(n) * np.exp(-tt / 0.003) * 0.6; x = (x * amp).astype(np.float32)
    place(fx, x, t0, pan); place(revsend, x * send, t0, pan)
def click(t0, amp=0.16, pan=0.0):
    n = int(0.3 * SR); tt = np.arange(n) / SR
    x = np.sin(2 * np.pi * 1180 * tt) * np.exp(-tt / 0.012) + 0.6 * np.sin(2 * np.pi * 300 * tt) * np.exp(-tt / 0.02) + rng.standard_normal(n) * np.exp(-tt / 0.002) * 0.7
    x = (x * amp).astype(np.float32); place(fx, x, t0, pan); place(revsend, x * 0.5, t0, pan)
def pop(t0, f=880, amp=0.10, pan=0.0):
    n = int(0.4 * SR); tt = np.arange(n) / SR; fr = f * (1 + 0.5 * np.exp(-tt / 0.02))
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt / 0.07); x = (x * amp).astype(np.float32); place(fx, x, t0, pan); place(revsend, x * 0.6, t0, pan)
def whoosh(t0, dur, f0, f1, amp, pan0=-0.4, pan1=0.4, seed=0, lead=0.5):
    x = bandnoise(dur, f0, f1, 0.7, seed)
    n = len(x); e = np.sin(np.linspace(0, np.pi, n)) ** 2.0 if lead >= 0.5 else np.linspace(0, 1, n) ** 3
    x = x * e.astype(np.float32) * amp
    # slow stereo travel
    pans = np.linspace(pan0, pan1, n); gl = np.cos((pans + 1) * np.pi / 4); gr = np.sin((pans + 1) * np.pi / 4)
    i = int(t0 * SR); j = min(N, i + n); m = j - i
    fx[0, i:j] += x[:m] * gl[:m]; fx[1, i:j] += x[:m] * gr[:m]; revsend[0, i:j] += x[:m] * gl[:m] * 0.5; revsend[1, i:j] += x[:m] * gr[:m] * 0.5
def boom(t0, amp=0.9, dec=2.4, f_end=34.0):
    n = int(dec * SR); tt = np.arange(n) / SR; fr = f_end + 70 * np.exp(-tt / 0.12)
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt / (dec * 0.38))
    x += 0.5 * np.sin(2 * np.pi * np.cumsum(fr * 2.0) / SR) * np.exp(-tt / 0.35)          # harmonic so phones hear it
    nz = bandnoise(0.5, 900, 120, 1.2, 3)[: int(0.5 * SR)]; x[: len(nz)] += nz * np.exp(-np.arange(len(nz)) / SR / 0.09) * 0.6
    x = (x * amp).astype(np.float32); place(fx, x, t0); place(revsend, x * 0.5, t0)
def sparkle(t0, t1, density, amp, scale):
    for _ in range(int(density * (t1 - t0))):
        f = hz(scale[rng.integers(len(scale))]) * 2 ** rng.integers(0, 2); tt0 = rng.uniform(t0, t1)
        pluck(f, tt0, amp * rng.uniform(0.4, 1), pan=rng.uniform(-0.9, 0.9), dec=rng.uniform(0.5, 1.4), send=1.2)


def kick(t0, amp=0.5):
    n = int(0.5 * SR); tt = np.arange(n) / SR; fr = 46 + 90 * np.exp(-tt / 0.045)
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt / 0.18) + 0.3 * np.sin(2 * np.pi * np.cumsum(fr * 2) / SR) * np.exp(-tt / 0.05)
    x += rng.standard_normal(n) * np.exp(-tt / 0.004) * 0.25; place(fx, (x * amp).astype(np.float32), t0)
def hat(t0, amp=0.05, pan=0.3, open_=False):
    n = int((0.22 if open_ else 0.06) * SR); tt = np.arange(n) / SR
    x = bandnoise(n / SR, 7000, 11000, 0.45, int(t0 * 100) % 997)[:n] * np.exp(-tt / (0.09 if open_ else 0.018)); place(fx, (x * amp).astype(np.float32), t0, pan)
def bass(f, t0, dur=0.55, amp=0.28):
    n = int(dur * SR); tt = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * tt) + 0.45 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt / 0.12)) * np.exp(-tt / (dur * 0.55)) * np.minimum(1, tt / 0.006)
    place(music, (x * amp).astype(np.float32), t0)
def stab(notes, t0, amp=0.07, dec=0.45, pan=0.0):
    n = int(dec * 4 * SR); tt = np.arange(n) / SR; x = np.zeros(n)
    for i, f in enumerate(notes):
        for k, a in [(1, 1.0), (2, 0.42), (3, 0.16)]: x += a * np.sin(2 * np.pi * f * k * tt + i)
    x = x * np.exp(-tt / dec) * np.minimum(1, tt / 0.004) / len(notes); x = (x * amp * 3).astype(np.float32)
    place(music, x, t0, pan); place(revsend, x * 0.7, t0, pan)
def swell(t0, dur, f0, f1, amp):
    x = bandnoise(dur, f0, f1, 0.9, int(t0 * 31) % 991) * (np.linspace(0, 1, int(dur * SR)) ** 2.0).astype(np.float32) * amp
    place(fx, x, t0); place(revsend, x * 0.5, t0)
# ---------------------------------------------------------------- score v2 (domains ad): C minor world, groove in the middle, quiet ending
# (different identity from the promo film: no Shepard tone, no sustained high partials, no finale pluck cluster)
CH = {"cm": ["C3", "G3", "Bb3", "D4", "Eb4"], "ab": ["Ab2", "Eb3", "G3", "C4", "Eb4"], "eb": ["Eb3", "Bb3", "D4", "G4"], "bb": ["Bb2", "F3", "Bb3", "D4", "F4"]}
FR = lambda k: [hz(n) for n in CH[k]]
ROOT = {"cm": "C2", "ab": "Ab1", "eb": "Eb2", "bb": "Bb1"}
BEAT = 0.6
# 0 - 7: slow dark swell, low and warm. nothing thin, nothing high.
sd = (np.sin(2 * np.pi * hz("C1") * t) + 0.6 * np.sin(2 * np.pi * hz("C2") * t) + 0.25 * np.sin(2 * np.pi * hz("G2") * t)) * env([(0, 0), (3.0, 0.07), (12.0, 0.08), (13.0, 0.0)])
music[0] += sd; music[1] += sd
pad(FR("cm"), 0.3, 13.2, 3.6, 1.6, 0.42, bright=(0.15, 0.55), K=4, seed=61)
swell(0.8, 3.6, 200, 1800, 0.10); swell(3.8, 2.8, 400, 3200, 0.12)
whoosh(2.5, 1.4, 250, 2000, 0.10, -0.3, 0.3, 61)
whoosh(6.45, 0.5, 500, 4500, 0.24, -0.4, 0.4, 62)
# 7 - 12.6: three things that do not exist yet (low wooden knocks), then the name (warm chord)
for tn, nt in [(7.6, "G3"), (9.0, "Bb3"), (10.6, "D4")]: pluck(hz(nt), tn, 0.12, 0.0, 1.4, 1.0); kick(tn, 0.22)
pad(FR("ab"), 11.3, 15.0, 0.6, 1.2, 0.34, bright=(0.3, 0.7), K=4, seed=63)
stab([hz("Ab3"), hz("Eb4"), hz("G4"), hz("C5")], 11.9, 0.09, 1.2)
whoosh(12.35, 0.55, 500, 4500, 0.24, -0.4, 0.4, 64)
# 12.6 - 17.9: the search. pulse of low thumps, rising as the name is typed
click(13.98, 0.2); whoosh(14.05, 0.9, 400, 3000, 0.16, -0.3, 0.5, 65)
for i in range(8): kick(14.2 + i * BEAT * 0.8, 0.12 + 0.03 * i)
for i in range(9): tick(15.0 + i * 0.094, 2000 + rng.integers(-300, 300), 0.07, -0.1 + 0.025 * i, 0.02, 0.25)
pop(15.8, 740, 0.09, 0.0)
for i, tc in enumerate([16.15, 16.3, 16.45]): pop(tc, 740 + 110 * (i + 1), 0.08, -0.3 + 0.3 * i)
swell(16.6, 1.3, 300, 6500, 0.2)
# 17.9 - 32: prices. a real groove: kick, offbeat hats, bass, chord stabs, a sparse melody on the price cues
GROOVE = [(17.9, 22.4, "cm"), (22.4, 26.8, "ab"), (26.8, 31.2, "eb"), (31.2, 32.0, "bb")]
pad(FR("cm"), 17.4, 23.0, 1.2, 1.4, 0.26, bright=(0.3, 0.6), K=4, seed=70)
pad(FR("ab"), 22.2, 27.4, 1.2, 1.4, 0.26, bright=(0.3, 0.6), K=4, seed=71)
pad(FR("eb"), 26.6, 31.8, 1.2, 1.4, 0.26, bright=(0.3, 0.6), K=4, seed=72)
tb = 17.9; k = 0
while tb < 31.9:
    g = [x for x in GROOVE if x[0] <= tb < x[1]][0][2]; r = hz(ROOT[g])
    kick(tb, 0.34 if k % 4 == 0 else 0.30)
    hat(tb + BEAT / 2, 0.05, 0.4 if k % 2 else -0.4, open_=(k % 4 == 3))
    if k % 4 in (0, 2): bass(r, tb, 0.5)
    if k % 4 == 2: bass(r * 2 ** (7 / 12), tb + BEAT * 0.75, 0.3, 0.2)
    if k % 4 in (1, 3): stab([hz(n) for n in CH[g][1:4]], tb + BEAT / 2, 0.06, 0.3, 0.3 if k % 2 else -0.3)
    tb += BEAT; k += 1
for i, (tc, nt) in enumerate([(18.15, "G5"), (21.25, "Bb5"), (23.85, "C6"), (26.35, "D6"), (29.05, "G6")]):
    pop(tc, hz(nt) * 0.5, 0.10, -0.4 + 0.2 * i); pluck(hz(nt), tc + 0.04, 0.07, 0.3, 0.9, 1.0)
    for j in range(5): tick(tc + 0.2 + j * 0.16, 2000 + 140 * j, 0.03, 0.0, 0.012, 0.3)
swell(31.0, 0.9, 300, 7000, 0.22); whoosh(31.8, 0.5, 500, 4500, 0.24, -0.4, 0.4, 66)
# 32 - 36.9: the groove drops out; two clicks; a held beat; a warm chord when the name is his
click(32.62, 0.2); pop(32.75, 940, 0.12, 0.3); click(33.5, 0.2); pop(33.62, 1100, 0.12, -0.2)
whoosh(34.2, 0.45, 7000, 500, 0.14, 0.3, -0.3, 67)
kick(34.55, 0.5); pad(FR("eb"), 34.5, 38.5, 0.3, 1.6, 0.42, bright=(0.5, 0.8), K=4, seed=73)
stab([hz(n) for n in ["Eb4", "G4", "Bb4", "D5"]], 34.6, 0.10, 1.6)
for i, nt in enumerate(["G5", "Bb5", "D6"]): pluck(hz(nt), 34.9 + i * 0.16, 0.07, -0.3 + 0.3 * i, 1.4, 1.1)
tick(35.2, 1500, 0.10, 0.0, 0.05)
whoosh(36.6, 0.45, 500, 4500, 0.24, -0.4, 0.4, 68)
# 36.9 - 41.2: groove returns lighter; the site comes alive: a rising run, then a lift
click(38.22, 0.2)
for i in range(6): pop(38.35 + i * 0.22, 600 + 60 * i, 0.06, -0.5 + 0.2 * i)
for i in range(8): kick(37.2 + i * BEAT * 1.0, 0.18); hat(37.5 + i * BEAT, 0.04, 0.3)
bass(hz("Bb1"), 37.2, 0.5); bass(hz("Eb2"), 38.4, 0.5); bass(hz("Bb1"), 39.6, 0.5)
for i, nt in enumerate(["Eb5", "G5", "Bb5", "D6", "Eb6"]): pluck(hz(nt), 39.75 + i * 0.07, 0.07, -0.3 + 0.15 * i, 1.1, 1.1)
swell(39.4, 0.7, 300, 7000, 0.20); pop(39.9, 1050, 0.12, 0.0)
whoosh(40.95, 0.45, 500, 4500, 0.24, -0.4, 0.4, 69)
# 41.2 - 50: brand end card. a different ending: the groove stops, one warm chord and a soft mallet motif, then a clean fade (no boom, no sparkle cluster)
pad(FR("cm"), 41.0, 50.0, 1.2, 2.6, 0.40, bright=(0.4, 0.8), K=5, seed=80)
stab([hz(n) for n in ["C4", "Eb4", "G4", "Bb4"]], 41.3, 0.12, 2.2); bass(hz("C2"), 41.3, 2.0, 0.3)
for i, nt in enumerate(["Bb4", "Eb5", "G5", "Bb5"]): pluck(hz(nt), 42.2 + i * 0.5, 0.08, -0.3 + 0.2 * i, 1.6, 1.2)
for i in range(6): tick(42.8 + i * 0.36, 1700 + 100 * i, 0.04, -0.4 + 0.16 * i, 0.02, 0.3)
for i in range(16): tick(46.15 + i * 0.07, 2000 + rng.integers(-200, 200), 0.05, 0.0, 0.02, 0.3)
click(47.6, 0.2); stab([hz(n) for n in ["C4", "G4", "Bb4", "D5", "Eb5"]], 47.62, 0.12, 2.6); bass(hz("C2"), 47.62, 2.4, 0.3)
for i, nt in enumerate(["G5", "Bb5", "D6", "Eb6"]): pluck(hz(nt), 47.9 + i * 0.22, 0.06 - 0.01 * i, -0.3 + 0.2 * i, 2.0, 1.3)

# ---------------------------------------------------------------- reverb (long, dark, stereo-decorrelated)
def make_ir(T60=4.2, pre=0.03, lp=5200, seed=5):
    r = np.random.default_rng(seed); n = int(T60 * 1.3 * SR); out = []
    for ch in range(2):
        x = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.91 / T60); x[: int(pre * SR)] = 0
        X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR); X *= 1 / (1 + (f / lp) ** 2) ** 1.0; X *= 1 / (1 + (60 / np.maximum(f, 1)) ** 2); out.append(np.fft.irfft(X, n))
    ir = np.array(out); ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True)); return ir.astype(np.float32)
def fftconv(x, ir):
    n = 1 << int(np.ceil(np.log2(x.shape[1] + ir.shape[1])))
    out = np.zeros((2, N), np.float32)
    for ch in range(2):
        for src in range(2):
            y = np.fft.irfft(np.fft.rfft(x[src], n) * np.fft.rfft(ir[(ch + src) % 2 if False else ch], n), n)[:N]
            out[ch] += y.astype(np.float32) * 0.5
    return out
ir = make_ir()
# voice: read decoded voice for sidechain + a quiet room around it
import os, json
voice = np.zeros(N, np.float32)
if os.path.exists("assets/audio/voice.mp3"):
    v = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "assets/audio/voice.mp3", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
    voice[: min(N, len(v))] = v[:N]
else:   # no voice yet: duck against the estimated word windows
    for w in json.load(open("data/words.json"))["words"]: voice[int(w["start"] * SR): int(w["end"] * SR)] = 0.08
venv = smooth_env(np.abs(voice), 60); venv = np.clip(venv / 0.045, 0, 1)
duck = 1 - 0.48 * smooth_env(venv, 140); duck_fx = 1 - 0.22 * smooth_env(venv, 140)
revin = revsend + music * 0.45
wet = fftconv(revin, ir)
vwet = fftconv(np.stack([voice, voice]) , make_ir(1.7, 0.012, 7000, 11)) * (0.20 if os.path.exists('assets/audio/voice.mp3') else 0.0)     # tiny natural room for the voice
gap = env([(0, 1), (99, 1)])
mix = (music * duck + fx * duck_fx * 1.0 + wet * duck * 0.9 * 0.9) 
mix[:, :] *= gap
mix += vwet * 1.0
fade = env([(0, 0), (1.2, 1), (48.6, 1), (49.95, 0)]); mix *= fade
# gentle mastering: tilt-free soft clip, then true-peak-safe gain
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
rms = np.sqrt((mix ** 2).mean()); mix *= (10 ** (-26.0 / 20)) / rms
pk = np.abs(mix).max(); 
if pk > 0.89: mix *= 0.89 / pk
pcm = (np.clip(mix, -1, 1).T * 32767).astype("<i2")
with wave.open(sys.argv[1] if len(sys.argv) > 1 else "assets/audio/score.wav", "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("score written; rms dBFS", 20 * np.log10(np.sqrt((mix ** 2).mean())), "peak", 20 * np.log10(np.abs(mix).max()))
