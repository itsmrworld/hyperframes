#!/usr/bin/env python3
"""SatisfyCloud score + sound design, synthesised deterministically (numpy only). Output: assets/audio/score.wav (48k stereo)."""
import numpy as np, subprocess, wave, sys
SR = 48000; DUR = 74.0; N = int(DUR * SR)
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

# ---------------------------------------------------------------- score
D = {"night": ["D2", "A2", "F3", "A3", "E4"], "cold": ["D3", "Eb3", "A3", "Eb4"], "arrive": ["D2", "A2", "D3", "F#3", "A3", "E4", "A4"],
     "bm": ["B1", "F#2", "B2", "D3", "A3", "D4"], "g": ["G1", "D2", "G2", "B2", "F#3", "D4"], "a": ["A1", "E2", "A2", "C#3", "G3", "E4"],
     "warm": ["D3", "A3", "F#4", "A4", "E5", "B4"]}
F = lambda names: [hz(n) for n in names]
# 0 - 14.4  night: low room, breathing pad that grows bright, slow pulse like a heartbeat
rt = pink(N) * env([(0, 0.0), (0.4, 0.05), (13.5, 0.05), (14.6, 0.0)]); music[0] += rt * 0.35; music[1] += np.roll(rt, 900) * 0.35
sd = (np.sin(2 * np.pi * hz("D1") * t) + 0.55 * np.sin(2 * np.pi * hz("D2") * t + 0.4) + 0.22 * np.sin(2 * np.pi * hz("A2") * t)) * env([(0, 0), (2.5, 0.06), (14.2, 0.10), (14.8, 0.0)])
music[0] += sd * 0.9; music[1] += sd * 0.9
D["night_hi"] = ["D3", "A3", "F4", "A4", "E5"]
pad(F(D["night"]), 0.8, 14.8, 4.0, 1.2, 0.30, bright=(0.35, 1.0), seed=1)
pad(F(D["night_hi"]), 2.0, 14.8, 5.0, 1.2, 0.26, bright=(0.4, 1.0), K=8, seed=2)
for k in range(11):                                   # heartbeat: wakes with the screen (3.46), quickens under 'the whole world'
    tb = 3.46 + k * (0.95 if k < 7 else 0.82 - 0.04 * (k - 7))
    n = int(0.5 * SR); tt = np.arange(n) / SR; x = (0.5 * np.sin(2 * np.pi * 52 * tt) + 0.8 * np.sin(2 * np.pi * 104 * tt) + 0.5 * np.sin(2 * np.pi * 156 * tt)) * np.exp(-tt / 0.11) * (0.16 + 0.02 * k)
    place(music, x.astype(np.float32), tb); place(music, (x * 0.55).astype(np.float32), tb + 0.24)
whoosh(3.36, 1.4, 300, 1800, 0.10, -0.2, 0.2, 11)  # screen wakes
pop(4.59, 740, 0.10, 0.3); pad(F(["D4", "A4", "E5"]), 4.59, 14.4, 1.2, 1.4, 0.06, bright=(0.4, 0.7), seed=3)  # the idea is born (orb)
whoosh(11.9, 2.3, 200, 5200, 0.30, -0.5, 0.5, 5)   # swell: idea grows into a world
boom(14.39, 0.7, 2.2); whoosh(14.0, 0.5, 500, 5000, 0.30, -0.3, 0.3, 6)    # seam 1 (push-through)
# 14.4 - 24.5  the problem: cold, thin, dissonant, brittle
pad(F(D["cold"]), 14.2, 24.6, 1.2, 0.6, 0.26, bright=(0.3, 0.45), seed=9, spread=0.3)
trem = (0.5 + 0.5 * np.sin(2 * np.pi * 5.2 * t)).astype(np.float32); music[0] += (np.sin(2 * np.pi * hz("Eb4") * t) * trem * env([(14.4, 0), (16, 0.025), (23, 0.05), (24.4, 0.0)])).astype(np.float32); music[1] += (np.sin(2 * np.pi * hz("Bb3") * t) * trem * env([(14.4, 0), (16, 0.02), (23, 0.045), (24.4, 0.0)])).astype(np.float32)
for tc, a in [(15.9, 0.08), (17.1, 0.09), (18.0, 0.10), (19.2, 0.08), (19.8, 0.07), (20.4, 0.07), (21.0, 0.06)]: tick(tc, 1500 + 80 * (tc % 3), a, rng.uniform(-0.6, 0.6), 0.05)
click(18.45, 0.18); boom(18.5, 0.35, 1.4, 48)                                # the 502 crash
for k in range(5): tick(18.2 + k * 0.06, 3100 - k * 300, 0.07, 0.4, 0.015)    # glitch burst
# the dream goes heavy: everything sinks (descending sub glide) -> then near-silence
n = int(1.6 * SR); tt = np.arange(n) / SR; fg = 140 * np.exp(-tt * 1.6); x = np.sin(2 * np.pi * np.cumsum(fg) / SR) * np.minimum(1, tt / 0.1) * np.exp(-tt * 1.1) * 0.45; place(music, x.astype(np.float32), 23.3)
whoosh(23.3, 1.3, 2400, 160, 0.20, 0.3, -0.3, 8)
# 24.85 arrival: a hard breath of silence (applied below), flash hit, ignition at 25.4, riser into the name at 26.66
boom(24.85, 0.75, 3.2, 38); whoosh(24.6, 0.5, 8000, 8000, 0.0, 0, 0, 1)
x = bandnoise(1.3, 300, 9000, 0.9, 21) * np.linspace(0, 1, int(1.3 * SR)).astype(np.float32) ** 2.2 * 0.38; place(fx, x, 25.36); place(revsend, x * 0.5, 25.36)
pad(F(["D3", "A3", "D4"]), 25.4, 27.0, 0.9, 0.8, 0.12, bright=(0.2, 1.0), seed=21)
sparkle(25.4, 26.6, 9, 0.05, ["D5", "F#5", "A5", "E5"])
boom(26.66, 0.8, 3.8, 36); pad(F(D["arrive"]), 26.62, 29.5, 0.15, 3.2, 0.30, bright=(1.0, 0.8), K=9, spread=1.0, seed=26)
sparkle(26.7, 30.0, 6, 0.06, ["D5", "F#5", "A5", "C#6", "E5"]); click(27.5, 0.0)
# 28.55 -> 58.15  the lineup: rising shepard (endless ascent), pulse of soft plucks, chords that change with the story
whoosh(28.2, 0.55, 500, 6000, 0.28, -0.4, 0.4, 31)
shepard(28.0, 60.0, 0.22)
prog = [(28.55, 35.8, "arrive"), (35.8, 42.0, "bm"), (42.0, 47.8, "g"), (47.8, 53.2, "a"), (53.2, 58.4, "bm"), (58.0, 64.6, "arrive")]
for i, (a, b, c) in enumerate(prog): pad(F(D[c]), a - 0.9 if i else a, b + 1.5, 1.8, 2.2, 0.36, bright=(0.8, 1.0), K=8, seed=40 + i)
CH = {"arrive": ["D4", "F#4", "A4", "C#5", "E5", "A5"], "bm": ["B3", "D4", "F#4", "A4", "D5", "F#5"], "g": ["G3", "B3", "D4", "F#4", "B4", "D5"], "a": ["A3", "C#4", "E4", "G4", "E5", "A5"]}
step = 60 / 112 / 2
pat = [0, 2, 4, 5, 4, 3, 1, 3]; ti = 28.9; k = 0
while ti < 64.3:
    c = [p for p in prog if p[0] <= ti < p[1]][0][2] if ti < 58.15 else "arrive"; ch = CH[c]
    if k % 8 in (0, 2, 3, 5, 6, 7) or ti > 58:
        pluck(hz(ch[pat[k % 8] % len(ch)]), ti, 0.075, pan=((k % 4) - 1.5) / 2.2, dec=0.75)
    ti += step; k += 1
# card-to-card flythroughs: air whooshes with rotating colour (no repeats)
cards = [31.7, 35.9, 38.95, 42.0, 45.55, 47.85, 53.2, 55.8]
for i, tc in enumerate(cards): whoosh(tc - 0.12, 0.62, [700, 1100, 520, 900][i % 4], [4200, 7600, 3200, 6000][i % 4], 0.14, -0.5 + 0.28 * (i % 4), 0.5 - 0.28 * (i % 4), 50 + i)
# UI: typing "yourbrand"(29.1+), register click, terminal lines, cores, pins, deploy, mail, sliders
for i in range(9): tick(29.0 + i * 0.09, 2200 + rng.integers(-250, 250), 0.07, -0.1 + 0.02 * i, 0.02, 0.2)
for i in range(3): pop(29.85 + i * 0.12, 1040 + 120 * i, 0.07, 0.0)
click(30.6, 0.18); pop(30.69, 1320, 0.12); 
for i, tc in enumerate([36.1, 36.65, 37.2, 37.75]): tick(tc, 1900 + 60 * i, 0.08, -0.1, 0.03)
pop(38.0, 980, 0.10, -0.3); pop(38.15, 1170, 0.10, 0.3)
pop(39.1, 700, 0.06, -0.3); pop(40.0, 760, 0.09, 0.2); pop(40.85, 900, 0.07, -0.2); pop(41.05, 1000, 0.07, 0.2)
for i in range(16): tick(42.75 + i * 0.075, 1600 + i * 120, 0.05 + 0.002 * i, -0.5 + i / 16, 0.02, 0.3)
for i in range(8): pop(45.8 + i * 0.26, 640 + 90 * i, 0.07, -0.7 + 0.2 * i)
for i in range(7): tick(48.2 + i * 0.16, 3200 - i * 60, 0.05, 0.3, 0.015)
tick(50.4, 900, 0.1, 0, 0.1); click(52.0, 0.2); pop(52.15, 1180, 0.12)
for i, tc in enumerate([53.7, 54.05, 54.4]): pop(tc, 820 + 90 * i, 0.08, 0.2 * i - 0.2)
tick(55.05, 520, 0.12, 0.4, 0.05)
x = bandnoise(1.8, 400, 6000, 0.8, 77) * np.linspace(0, 1, int(1.8 * SR)).astype(np.float32) ** 1.6 * 0.12; place(fx, x, 56.45)
# 58.15 worlds: tiles pop, counters tick, riser into the end card
whoosh(57.9, 0.5, 500, 6000, 0.28, -0.4, 0.4, 91)
for tc, f in [(58.27, 760), (58.93, 850), (59.77, 950), (60.5, 1060)]: pop(tc, f, 0.11, 0.0)
for i in range(18): tick(59.2 + i * 0.14, 2400 + rng.integers(-300, 300), 0.03, rng.uniform(-0.7, 0.7), 0.015)
x = bandnoise(2.2, 250, 10000, 0.85, 95) * np.linspace(0, 1, int(2.2 * SR)).astype(np.float32) ** 2.4 * 0.40; place(fx, x, 62.3); place(revsend, x * 0.45, 62.3)
pop(62.45, 1200, 0.10, 0.0)
# 64.5 end card: the sky clears; warm, still, luminous
boom(64.5, 0.55, 3.0, 40); whoosh(64.3, 1.2, 6500, 400, 0.30, 0.4, -0.4, 99)
pad(F(D["warm"]), 64.4, 74.0, 1.8, 2.8, 0.40, bright=(0.5, 1.0), K=9, spread=1.0, seed=64)
pluck(hz("D5"), 67.71, 0.16, -0.2, 1.6, 1.3); pluck(hz("A5"), 67.78, 0.10, 0.2, 1.6, 1.3); pluck(hz("F#5"), 67.9, 0.08, 0.0, 1.6, 1.3); boom(67.71, 0.55, 3.2, 40)
for i, f in enumerate(["A4", "D5", "F#5", "A5"]): pluck(hz(f), 68.92 + i * 0.12, 0.07, -0.4 + 0.27 * i, 1.4, 1.1)
for i in range(15): tick(70.5 + i * 0.07, 2300 + rng.integers(-200, 200), 0.05, 0.0, 0.02, 0.3)
click(71.82, 0.2); boom(71.82, 0.6, 3.0, 42)
for i, f in enumerate(["D5", "F#5", "A5", "C#6", "E6"]): pluck(hz(f), 71.86 + i * 0.09, 0.09 - 0.01 * i, -0.5 + 0.25 * i, 2.2, 1.6)
sparkle(71.9, 73.2, 6, 0.04, ["D6", "A5", "F#5", "E6"])

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
v = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "assets/audio/voice.mp3", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
voice = np.zeros(N, np.float32); voice[: min(N, len(v))] = v[:N]
venv = smooth_env(np.abs(voice), 60); venv = np.clip(venv / 0.045, 0, 1)
duck = 1 - 0.48 * smooth_env(venv, 140); duck_fx = 1 - 0.22 * smooth_env(venv, 140)
revin = revsend + music * 0.45
wet = fftconv(revin, ir)
vwet = fftconv(np.stack([voice, voice]) , make_ir(1.7, 0.012, 7000, 11)) * 0.20     # tiny natural room for the voice
gap = env([(0, 1), (24.40, 1), (24.55, 0.02), (24.83, 0.02), (24.86, 1), (99, 1)])     # the held breath before arrival
mix = (music * duck + fx * duck_fx * 1.0 + wet * duck * 0.9 * 0.9) 
mix[:, :] *= gap
mix += vwet * 1.0
fade = env([(0, 0), (0.6, 1), (72.6, 1), (73.95, 0)]); mix *= fade
# gentle mastering: tilt-free soft clip, then true-peak-safe gain
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
rms = np.sqrt((mix ** 2).mean()); mix *= (10 ** (-26.0 / 20)) / rms
pk = np.abs(mix).max(); 
if pk > 0.89: mix *= 0.89 / pk
pcm = (np.clip(mix, -1, 1).T * 32767).astype("<i2")
with wave.open(sys.argv[1] if len(sys.argv) > 1 else "assets/audio/score.wav", "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("score written; rms dBFS", 20 * np.log10(np.sqrt((mix ** 2).mean())), "peak", 20 * np.log10(np.abs(mix).max()))
