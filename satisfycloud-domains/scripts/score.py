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

# ---------------------------------------------------------------- score (SatisfyCloud domains, 50 s)
D = {"a": ["D3", "A3", "F#4", "A4", "E5"], "b": ["B2", "F#3", "D4", "A4", "C#5"], "c": ["G2", "D3", "B3", "D4", "F#4", "A4"], "d": ["A2", "E3", "C#4", "E4", "G4"], "e": ["D3", "A3", "D4", "F#4", "A4", "E5"]}
F = lambda names: [hz(n) for n in names]
CH = {"a": ["D4", "F#4", "A4", "C#5", "E5", "A5"], "b": ["B3", "D4", "F#4", "A4", "C#5", "F#5"], "c": ["G3", "B3", "D4", "F#4", "B4", "D5"], "d": ["A3", "C#4", "E4", "G4", "E5", "A5"], "e": ["D4", "F#4", "A4", "C#5", "E5", "A5"]}
# 0 - 7 : light rises in silence (no voice): sub bed, a pad that opens, one bar of air
sd = (np.sin(2 * np.pi * hz("D1") * t) + 0.5 * np.sin(2 * np.pi * hz("D2") * t + 0.3) + 0.2 * np.sin(2 * np.pi * hz("A2") * t)) * env([(0, 0), (3.0, 0.07), (12.5, 0.09), (13.5, 0.0)])
music[0] += sd; music[1] += sd
pad(F(D["a"]), 0.2, 12.8, 3.8, 1.2, 0.34, bright=(0.25, 1.0), seed=1)
pad(F(["D5", "A5", "E6"]), 1.5, 12.8, 4.0, 1.2, 0.09, bright=(0.3, 0.9), K=6, seed=2)
whoosh(2.4, 1.5, 300, 2400, 0.12, -0.3, 0.3, 11); sparkle(3.4, 6.6, 1.6, 0.04, ["D5", "F#5", "A5", "E5"])
whoosh(6.45, 0.5, 500, 5000, 0.26, -0.4, 0.4, 12)
# 7 - 12.6 : story, slower and warmer; three notes, one per beat of the story
pad(F(D["b"]), 6.8, 13.0, 1.4, 1.4, 0.32, bright=(0.5, 0.95), seed=5)
for tn, nt in [(7.65, "F#4"), (9.3, "A4"), (10.9, "D5")]: pluck(hz(nt), tn, 0.12, 0.1, 1.6, 1.2)
pop(10.9, 780, 0.08, 0.0); whoosh(10.8, 1.2, 600, 4200, 0.12, -0.2, 0.2, 13)
whoosh(12.35, 0.55, 500, 5200, 0.26, -0.4, 0.4, 14)
# 12.6 - 17.9 : the search. click, dive, type, results
click(13.98, 0.2); whoosh(14.05, 0.9, 400, 3400, 0.16, -0.3, 0.5, 15); boom(14.9, 0.35, 1.6, 52)
for i in range(9): tick(15.0 + i * 0.094, 2300 + rng.integers(-250, 250), 0.07, -0.1 + 0.025 * i, 0.02, 0.25)
pop(15.8, 880, 0.09, 0.0)
for i, tc in enumerate([16.15, 16.3, 16.45]): pop(tc, 980 + 150 * i, 0.09, -0.3 + 0.3 * i)
# 17.9 - 32 : prices. chords turn with the story, a soft pulse of plucks, a rising tone underneath
shepard(17.0, 36.0, 0.18, 1 / 12)
prog = [(12.8, 24.0, "c"), (24.0, 28.5, "d"), (28.5, 33.0, "e"), (33.0, 41.4, "e")]
pad(F(D["c"]), 12.6, 24.8, 1.6, 2.2, 0.36, bright=(0.7, 1.0), K=8, seed=20)
pad(F(D["d"]), 24.0, 29.6, 1.6, 2.0, 0.34, bright=(0.8, 1.0), K=8, seed=21)
pad(F(D["e"]), 28.8, 41.6, 1.8, 1.6, 0.38, bright=(0.8, 1.0), K=8, seed=22)
step = 60 / 104 / 2; pat = [0, 2, 4, 5, 4, 3, 1, 3]; ti = 17.95; k = 0
while ti < 41.3:
    ch = CH["c"] if ti < 24.0 else CH["d"] if ti < 28.5 else CH["e"]
    if k % 8 in (0, 2, 3, 5, 6, 7): pluck(hz(ch[pat[k % 8] % len(ch)]), ti, 0.07, pan=((k % 4) - 1.5) / 2.2, dec=0.75)
    ti += step; k += 1
for i, (tc, f) in enumerate([(18.15, 784), (21.25, 880), (23.85, 988), (26.35, 1175), (29.05, 1319)]):
    pop(tc, f, 0.12, -0.4 + 0.2 * i); pluck(hz("D6") if i % 2 == 0 else hz("A5"), tc + 0.05, 0.05, 0.3, 1.2, 1.2)
    for j in range(7): tick(tc + 0.18 + j * 0.13, 2600 + 90 * j, 0.03, 0.0, 0.012, 0.3)    # price counting up
whoosh(31.8, 0.5, 500, 5200, 0.26, -0.4, 0.4, 31)
# 32 - 36.9 : add to cart, pay, the name is yours
click(32.62, 0.2); pop(32.75, 1180, 0.12, 0.3); click(33.5, 0.2); pop(33.62, 1320, 0.12, -0.2); pop(32.92, 1000, 0.07, 0.4)
whoosh(34.2, 0.45, 8000, 600, 0.14, 0.3, -0.3, 32)
boom(34.55, 0.55, 3.0, 40)
for i, f in enumerate(["D5", "F#5", "A5", "C#6"]): pluck(hz(f), 34.6 + i * 0.1, 0.10 - 0.012 * i, -0.4 + 0.27 * i, 2.0, 1.5)
tick(35.2, 1800, 0.1, 0.0, 0.05); sparkle(34.8, 36.2, 5, 0.04, ["D6", "A5", "F#5", "E6"])
whoosh(36.6, 0.45, 500, 5200, 0.26, -0.4, 0.4, 33)
# 36.9 - 41.2 : hosting, the site comes alive
click(38.22, 0.2)
for i in range(6): pop(38.35 + i * 0.22, 700 + 70 * i, 0.06, -0.5 + 0.2 * i)
pop(39.9, 1250, 0.12, 0.0); pluck(hz("A5"), 39.9, 0.10, 0.0, 1.8, 1.4); pluck(hz("E6"), 39.98, 0.06, 0.3, 1.8, 1.4)
x = bandnoise(1.2, 300, 8000, 0.9, 41) * np.linspace(0, 1, int(1.2 * SR)).astype(np.float32) ** 2.2 * 0.26; place(fx, x, 39.7); place(revsend, x * 0.5, 39.7)
whoosh(40.95, 0.45, 500, 5200, 0.26, -0.4, 0.4, 34)
# 41.2 - 50 : brand end card
boom(41.4, 0.7, 3.4, 38); pluck(hz("D5"), 41.4, 0.16, -0.2, 1.8, 1.4); pluck(hz("A5"), 41.48, 0.10, 0.2, 1.8, 1.4); pluck(hz("F#5"), 41.6, 0.08, 0.0, 1.8, 1.4)
sparkle(41.5, 44.0, 3, 0.05, ["D5", "F#5", "A5", "C#6", "E5"])
for i in range(6): tick(42.8 + i * 0.36, 2000 + 120 * i, 0.04, -0.4 + 0.16 * i, 0.02, 0.3)
for i in range(16): tick(46.15 + i * 0.07, 2300 + rng.integers(-200, 200), 0.05, 0.0, 0.02, 0.3)
click(47.6, 0.2); boom(47.6, 0.6, 3.2, 42)
for i, f in enumerate(["D5", "F#5", "A5", "C#6", "E6"]): pluck(hz(f), 47.65 + i * 0.09, 0.09 - 0.01 * i, -0.5 + 0.25 * i, 2.4, 1.6)
sparkle(47.7, 49.0, 5, 0.04, ["D6", "A5", "F#5", "E6"])

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
