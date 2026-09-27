# 🎵 AI Music Generator

A complete beginner-friendly AI music generation app using:

- Python
- PyTorch
- LSTM
- PrettyMIDI
- Streamlit

## 1. Install Python

Python 3.10 or 3.11 is recommended for the simplest setup.

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If PyTorch installation gives an issue on your machine, install the appropriate PyTorch build from the official PyTorch installation page and then run:

```bash
pip install streamlit pretty_midi numpy
```

## 4. Run the application

```bash
streamlit run app.py
```

The browser should open the application.

## 5. Use the application

1. Collect MIDI files from a legal MIDI dataset.
2. Upload one or more `.mid`/`.midi` files.
3. Click **Analyze Dataset**.
4. Adjust sequence length and training settings.
5. Click **Train Model**.
6. After training finishes, click **Generate Music**.
7. Download `ai_generated_music.mid`.

## How the model works

```text
MIDI files
    ↓
Extract MIDI pitches
    ↓
Integer encoding
    ↓
Sliding sequences
    ↓
Embedding
    ↓
LSTM
    ↓
Dense + Softmax
    ↓
Next-note prediction
    ↓
Autoregressive generation
    ↓
Generated MIDI
```

## Important limitation

This version is intentionally a first complete working baseline.

It learns **pitch sequences only**. It does not yet learn:

- exact note durations
- velocity dynamics
- chords
- multiple instruments
- long-term musical structure
- expressive timing

The next major upgrade should be a **note-event representation** such as:

```text
pitch + duration + velocity + time
```

That will make the generated music much more musical.

## Project structure

```text
ai_music_generator/
├── app.py
├── requirements.txt
└── README.md
```
