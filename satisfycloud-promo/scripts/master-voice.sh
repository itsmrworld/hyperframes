#!/usr/bin/env bash
# Rebuild assets/audio/voice_master.wav from the ElevenLabs take (EQ, gentle compression, loudness match).
set -euo pipefail
cd "$(dirname "$0")/.."
ffmpeg -y -v error -i assets/audio/voice.mp3 \
  -af "highpass=f=75,equalizer=f=250:t=q:w=1:g=-1.5,equalizer=f=3800:t=q:w=1.2:g=2.2,acompressor=threshold=-24dB:ratio=2.6:attack=12:release=160:makeup=2,loudnorm=I=-17.5:TP=-2.0:LRA=7" \
  -ar 48000 -ac 2 assets/audio/voice_master.wav
