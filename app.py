import io
import os
import tempfile
from collections import Counter

import numpy as np
import pretty_midi
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model
# -----------------------------
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


# -----------------------------
# MIDI processing
# -----------------------------
def extract_pitches(midi_bytes):
    """Extract pitched note numbers from a MIDI file."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        f.write(midi_bytes)
        path = f.name

    try:
        midi = pretty_midi.PrettyMIDI(path)
        notes = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            # Sort by start time to preserve musical order.
            instrument_notes = sorted(instrument.notes, key=lambda n: (n.start, n.pitch))
            notes.extend(note.pitch for note in instrument_notes)

        return notes
    finally:
        if os.path.exists(path):
            os.remove(path)


def build_sequences(all_notes, sequence_length):
    """Create X -> next-note y training pairs."""
    if len(all_notes) <= sequence_length:
        return None, None

    unique_notes = sorted(set(all_notes))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}

    encoded = [note_to_int[n] for n in all_notes]

    X = []
    y = []

    for i in range(len(encoded) - sequence_length):
        X.append(encoded[i:i + sequence_length])
        y.append(encoded[i + sequence_length])

    return (
        np.asarray(X, dtype=np.int64),
        np.asarray(y, dtype=np.int64),
        unique_notes,
    )


def create_midi(pitches, tempo=100):
    """Convert generated MIDI pitch numbers into a playable MIDI file."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

    step = 60.0 / tempo / 2.0
    time = 0.0

    for pitch in pitches:
        note = pretty_midi.Note(
            velocity=85,
            pitch=int(pitch),
            start=time,
            end=time + step,
        )
        instrument.notes.append(note)
        time += step

    midi.instruments.append(instrument)

    output = io.BytesIO()
    midi.write(temp_path := tempfile.mktemp(suffix=".mid"))

    with open(temp_path, "rb") as f:
        output.write(f.read())

    os.remove(temp_path)
    output.seek(0)
    return output


def train_model(X, y, vocab_size, epochs, batch_size, lr, progress_callback=None):
    model = MusicLSTM(vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_items += batch_x.size(0)

        epoch_loss = total_loss / max(total_items, 1)
        history.append(epoch_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss)

    return model, history


def sample_next(logits, temperature=1.0):
    """Sample a note from the probability distribution."""
    temperature = max(float(temperature), 0.05)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).item()


def generate_music(model, seed, int_to_note, sequence_length, num_notes, temperature):
    model.eval()

    encoded_seed = [int_to_note.index(note) for note in seed]
    generated = list(seed)

    current = encoded_seed[:]

    with torch.no_grad():
        for _ in range(num_notes):
            x = torch.tensor([current[-sequence_length:]], dtype=torch.long).to(DEVICE)
            logits = model(x)
            next_idx = sample_next(logits[0], temperature)
            next_pitch = int_to_note[next_idx]

            generated.append(next_pitch)
            current.append(next_idx)

    return generated


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 AI Music Generator")
st.caption("Generate new MIDI music using an LSTM trained on your MIDI collection.")

with st.sidebar:
    st.header("⚙️ Training Settings")

    sequence_length = st.slider(
        "Sequence length",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
        help="How many previous notes the LSTM sees before predicting the next note.",
    )

    epochs = st.slider(
        "Epochs",
        min_value=1,
        max_value=100,
        value=10,
    )

    batch_size = st.selectbox(
        "Batch size",
        [16, 32, 64, 128],
        index=1,
    )

    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.0001, 0.0003, 0.001, 0.003],
        value=0.001,
    )

    temperature = st.slider(
        "Generation temperature",
        min_value=0.2,
        max_value=2.0,
        value=0.9,
        step=0.1,
        help="Lower = more predictable. Higher = more random.",
    )

    generated_notes_count = st.slider(
        "Generated notes",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    tempo = st.slider(
        "Output tempo (BPM)",
        min_value=60,
        max_value=180,
        value=100,
    )

st.subheader("1. Upload MIDI training data")

files = st.file_uploader(
    "Upload one or more .mid / .midi files",
    type=["mid", "midi"],
    accept_multiple_files=True,
)

if files:
    st.success(f"{len(files)} MIDI file(s) uploaded.")

    if st.button("🔎 Analyze Dataset"):
        with st.spinner("Reading MIDI files..."):
            all_notes = []
            file_stats = []

            for file in files:
                try:
                    notes = extract_pitches(file.getvalue())
                    all_notes.extend(notes)
                    file_stats.append((file.name, len(notes)))
                except Exception as e:
                    st.warning(f"Could not read {file.name}: {e}")

            if len(all_notes) <= sequence_length:
                st.error(
                    f"Not enough notes. You have {len(all_notes)} notes, "
                    f"but sequence length is {sequence_length}. Upload more MIDI data."
                )
            else:
                st.session_state["all_notes"] = all_notes
                st.session_state["file_stats"] = file_stats

if "all_notes" in st.session_state:
    all_notes = st.session_state["all_notes"]

    st.subheader("2. Dataset Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total notes", f"{len(all_notes):,}")
    c2.metric("Unique pitches", len(set(all_notes)))
    c3.metric("Pitch range", f"{min(all_notes)}–{max(all_notes)}")

    with st.expander("Uploaded file statistics"):
        for name, count in st.session_state.get("file_stats", []):
            st.write(f"**{name}** — {count:,} notes")

    result = build_sequences(all_notes, sequence_length)
    X, y, vocabulary = result

    if X is None:
        st.error("Not enough data for the selected sequence length.")
        st.stop()

    st.info(
        f"Training examples: {len(X):,} | "
        f"Vocabulary size: {len(vocabulary)} | "
        f"Device: {DEVICE}"
    )

    st.subheader("3. Train LSTM")

    if st.button("🚀 Train Model"):
        progress = st.progress(0)
        status = st.empty()
        losses = []

        def update_progress(epoch, total, loss):
            progress.progress(epoch / total)
            status.write(f"Epoch {epoch}/{total} — loss: {loss:.4f}")
            losses.append(loss)

        with st.spinner("Training model..."):
            model, history = train_model(
                X,
                y,
                vocab_size=len(vocabulary),
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                progress_callback=update_progress,
            )

        st.session_state["model"] = model
        st.session_state["vocabulary"] = vocabulary
        st.session_state["history"] = history

        st.success("Training complete! 🎉")

        st.line_chart(
            {"Training Loss": history},
            x_label="Epoch",
            y_label="Loss",
        )

if "model" in st.session_state:
    st.subheader("4. Generate New Music")

    vocabulary = st.session_state["vocabulary"]
    model = st.session_state["model"]

    seed_size = min(sequence_length, len(st.session_state["all_notes"]))
    seed = st.session_state["all_notes"][-seed_size:]

    st.write(
        f"The generator starts from the last {seed_size} notes of the training data "
        "and then predicts new notes autoregressively."
    )

    if st.button("🎼 Generate Music"):
        with st.spinner("Generating music..."):
            generated = generate_music(
                model=model,
                seed=seed,
                int_to_note=vocabulary,
                sequence_length=sequence_length,
                num_notes=generated_notes_count,
                temperature=temperature,
            )

            midi_file = create_midi(generated, tempo=tempo)

        st.session_state["generated_midi"] = midi_file.getvalue()
        st.session_state["generated_count"] = len(generated)

        st.success(f"Generated {len(generated):,} notes.")

    if "generated_midi" in st.session_state:
        st.download_button(
            label="⬇️ Download Generated MIDI",
            data=st.session_state["generated_midi"],
            file_name="ai_generated_music.mid",
            mime="audio/midi",
        )

        st.audio(
            st.session_state["generated_midi"],
            format="audio/midi",
        )

st.divider()
st.caption("Note: This first version learns pitch sequences only. It does not yet model rhythm, chords, or multiple instruments.")
