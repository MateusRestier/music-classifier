"""
app.py — Streamlit: upload de áudio ou link do YouTube → análise detalhada + predição de gênero.

Uso:
    streamlit run app/app.py

Requer que os modelos já tenham sido treinados e exportados em models/:
    classifier.pkl, scaler.pkl, label_encoder.pkl
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import librosa.display
import joblib

matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dsp.extract_features import (  # noqa: E402
    extract_features_from_wav,
    SR,
    N_MFCC,
    N_MELS,
    HOP_LENGTH,
    SEGMENT_OFFSET,
    SEGMENT_DURATION,
)

MODEL_DIR   = _PROJECT_ROOT / "models"
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    clf    = joblib.load(MODEL_DIR / "classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return clf, scaler, le


# ---------------------------------------------------------------------------
# Áudio
# ---------------------------------------------------------------------------

def _load_audio_segment(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SR, offset=SEGMENT_OFFSET,
                         duration=SEGMENT_DURATION, mono=True)
    if len(y) < SR:
        y, sr = librosa.load(path, sr=SR, duration=SEGMENT_DURATION, mono=True)
    return y, sr


def _compute_analysis(y: np.ndarray, sr: int) -> dict:
    """Computa todas as features e matrizes necessárias para exibição (uma única vez)."""
    hop = HOP_LENGTH

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
    times    = librosa.times_like(centroid, sr=sr, hop_length=hop)

    key = NOTE_NAMES[int(np.argmax(np.mean(chroma, axis=1)))]

    return dict(
        y=y, sr=sr, times=times, duration=len(y) / sr,
        bpm=float(np.atleast_1d(tempo)[0]), beat_times=beat_times,
        mel_db=mel_db, mfcc=mfcc, chroma=chroma,
        centroid=centroid, rolloff=rolloff, zcr=zcr, rms=rms,
        rms_mean=float(np.mean(rms)), key=key,
    )


def _download_youtube(url: str) -> tuple[str, str]:
    tmp_dir = tempfile.mkdtemp(prefix="mc_yt_")
    output_template = os.path.join(tmp_dir, "audio")
    env = {**os.environ, "PYTHONUTF8": "1"}

    r = subprocess.run(["yt-dlp", "--no-playlist", "--print", "title", url],
                       capture_output=True, env=env)
    title = r.stdout.decode("utf-8", errors="replace").strip() or "audio"

    cmd = [
        "yt-dlp", "--no-playlist",
        "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-acodec pcm_s16le",
        "--output", output_template, url,
    ]
    if FFMPEG_PATH:
        cmd[1:1] = ["--ffmpeg-location", FFMPEG_PATH]

    r = subprocess.run(cmd, capture_output=True, env=env)
    if r.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            (r.stdout + r.stderr).decode("utf-8", errors="replace").strip()
        )

    wav_path = output_template + ".wav"
    if not os.path.exists(wav_path):
        candidates = list(Path(tmp_dir).glob("audio*"))
        if not candidates:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise FileNotFoundError("yt-dlp não gerou nenhum arquivo de saída.")
        wav_path = str(candidates[0])

    return wav_path, title


def _is_youtube_url(text: str) -> bool:
    return bool(re.search(r"(youtube\.com/watch|youtu\.be/)", text))


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _fig_waveform(a: dict) -> go.Figure:
    """Forma de onda com marcadores de beat."""
    y, duration = a["y"], a["duration"]
    t = np.linspace(0, duration, len(y))

    # Downsample para o plot (máx 8000 pts para não travar o browser)
    step = max(1, len(y) // 8000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[::step], y=y[::step],
        mode="lines", line=dict(color="#4C9BE8", width=0.8),
        name="Amplitude",
    ))
    for bt in a["beat_times"]:
        fig.add_vline(x=float(bt), line=dict(color="#FF7F0E", width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#FF7F0E", dash="dot"), name="Beat",
    ))
    fig.update_layout(
        xaxis_title="Tempo (s)", yaxis_title="Amplitude",
        height=220, margin=dict(t=10, b=40, l=60, r=20),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _fig_mel(a: dict) -> plt.Figure:
    """Mel-Spectrogram (matplotlib + librosa.display)."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    img = librosa.display.specshow(
        a["mel_db"], sr=a["sr"], hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-Spectrogram")
    fig.tight_layout()
    return fig


def _fig_mfcc(a: dict) -> plt.Figure:
    """Heatmap dos 40 coeficientes MFCC ao longo do tempo."""
    fig, ax = plt.subplots(figsize=(11, 4))
    img = librosa.display.specshow(
        a["mfcc"], sr=a["sr"], hop_length=HOP_LENGTH,
        x_axis="time", ax=ax, cmap="coolwarm",
    )
    ax.set_ylabel("Coeficiente MFCC")
    ax.set_title("MFCCs (40 coeficientes × tempo)")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def _fig_chroma(a: dict) -> plt.Figure:
    """Heatmap de chroma (12 classes de pitch × tempo)."""
    fig, ax = plt.subplots(figsize=(11, 3))
    img = librosa.display.specshow(
        a["chroma"], sr=a["sr"], hop_length=HOP_LENGTH,
        x_axis="time", y_axis="chroma", ax=ax, cmap="YlOrRd",
    )
    ax.set_title("Chroma — conteúdo harmônico por nota")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def _fig_spectral(a: dict) -> go.Figure:
    """4 features espectrais ao longo do tempo (Plotly, interativo)."""
    t = a["times"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Centróide espectral (Hz)",
            "Rolloff espectral (Hz)",
            "Taxa de cruzamento por zero",
            "Energia RMS",
        ),
        vertical_spacing=0.08,
    )

    traces = [
        (a["centroid"], "#1f77b4", 1),
        (a["rolloff"],  "#ff7f0e", 2),
        (a["zcr"],      "#2ca02c", 3),
        (a["rms"],      "#d62728", 4),
    ]
    for values, color, row in traces:
        fig.add_trace(
            go.Scatter(x=t, y=values, mode="lines",
                       line=dict(color=color, width=1.2), showlegend=False),
            row=row, col=1,
        )

    fig.update_xaxes(title_text="Tempo (s)", row=4, col=1)
    fig.update_layout(height=550, margin=dict(t=40, b=40, l=70, r=20))
    return fig


def _fig_probs(classes: np.ndarray, probs: np.ndarray, pred_idx: int) -> go.Figure:
    sorted_idx = np.argsort(probs)[::-1]
    colors = ["#ff7f0e" if i == pred_idx else "#1f77b4" for i in sorted_idx]
    fig = go.Figure(go.Bar(
        x=[classes[i] for i in sorted_idx],
        y=[probs[i]   for i in sorted_idx],
        text=[f"{probs[i]:.1%}" for i in sorted_idx],
        textposition="auto",
        marker_color=colors,
    ))
    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1], title="Probabilidade"),
        xaxis_title="Gênero",
        showlegend=False,
        height=350,
        margin=dict(t=10),
    )
    return fig


def _fig_radar(classes: np.ndarray, probs: np.ndarray) -> go.Figure:
    cats = list(classes) + [classes[0]]   # fecha o polígono
    vals = list(probs)    + [probs[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        line=dict(color="#1f77b4"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=320, margin=dict(t=20, b=20, l=40, r=40),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Pipeline de análise
# ---------------------------------------------------------------------------

def _run_analysis(wav_path: str, audio_bytes: bytes | None) -> None:
    try:
        with st.spinner("Analisando áudio…"):
            y, sr    = _load_audio_segment(wav_path)
            a        = _compute_analysis(y, sr)
            features = extract_features_from_wav(wav_path)
    except Exception as exc:
        st.error(f"Erro na extração de features: `{exc}`")
        return

    # ── Player ────────────────────────────────────────────────────────────
    if audio_bytes is not None:
        suffix = Path(wav_path).suffix.lower()
        st.audio(audio_bytes, format="audio/wav" if suffix == ".wav" else "audio/mpeg")
    else:
        st.audio(wav_path)

    # ── Métricas rápidas ──────────────────────────────────────────────────
    st.subheader("Visão geral")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segmento analisado", f"{a['duration']:.1f} s")
    c2.metric("BPM estimado",       f"{a['bpm']:.1f}")
    c3.metric("Tonalidade (chroma)", a["key"])
    c4.metric("Energia RMS média",  f"{a['rms_mean']:.4f}")

    # ── Gráficos ──────────────────────────────────────────────────────────
    st.subheader("Análise de sinal")
    tab_wave, tab_mel, tab_mfcc, tab_chroma, tab_spec = st.tabs([
        "Forma de onda",
        "Mel-Spectrogram",
        "MFCCs",
        "Chroma",
        "Features espectrais",
    ])

    with tab_wave:
        st.caption("Amplitude ao longo do tempo. Linhas laranja pontilhadas = beats detectados.")
        st.plotly_chart(_fig_waveform(a), use_container_width=True)

    with tab_mel:
        st.caption("Representação tempo-frequência perceptual (escala Mel, em dB).")
        fig = _fig_mel(a)
        st.pyplot(fig)
        plt.close(fig)

    with tab_mfcc:
        st.caption("Os 40 coeficientes MFCC codificam o timbre da música ao longo do tempo.")
        fig = _fig_mfcc(a)
        st.pyplot(fig)
        plt.close(fig)

    with tab_chroma:
        st.caption(
            f"Energia por classe de pitch (C … B) ao longo do tempo. "
            f"Tonalidade dominante estimada: **{a['key']}**."
        )
        fig = _fig_chroma(a)
        st.pyplot(fig)
        plt.close(fig)

    with tab_spec:
        st.caption(
            "Centróide (brilho), Rolloff (frequência de corte 85%), "
            "ZCR (percussão/ruído) e RMS (energia) quadro a quadro."
        )
        st.plotly_chart(_fig_spectral(a), use_container_width=True)

    # ── Predição ──────────────────────────────────────────────────────────
    st.subheader("Predição de gênero")
    X        = np.array([list(features.values())])
    X_scaled = scaler.transform(X)
    probs    = clf.predict_proba(X_scaled)[0]
    classes  = le.classes_
    pred_idx = int(np.argmax(probs))

    col_m, col_b, col_r = st.columns([1, 2, 1])
    with col_m:
        runner_up = int(np.argsort(probs)[-2])
        st.metric("Gênero previsto",   classes[pred_idx], f"{probs[pred_idx]:.1%}")
        st.metric("2º mais provável",  classes[runner_up], f"{probs[runner_up]:.1%}")
    with col_b:
        st.plotly_chart(_fig_probs(classes, probs, pred_idx), use_container_width=True)
    with col_r:
        st.plotly_chart(_fig_radar(classes, probs), use_container_width=True)


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SoundDNA",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 SoundDNA")
st.caption("Análise de áudio + predição de gênero a partir de arquivo `.mp3`/`.wav` ou link do YouTube.")

models_ok = False
try:
    clf, scaler, le = load_models()
    models_ok = True
except FileNotFoundError as exc:
    st.error(
        f"Modelos não encontrados em `models/`.\n\n"
        f"Rode `python modeling/classify.py` primeiro.\n\n`{exc}`"
    )
except Exception as exc:
    st.error(f"Erro ao carregar modelos: `{exc}`")

if not models_ok:
    st.stop()

tab_upload, tab_youtube = st.tabs(["📂 Upload de arquivo", "▶️ Link do YouTube"])

# ── Aba 1: Upload ──────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Selecione um arquivo `.mp3` ou `.wav`", type=["mp3", "wav"])

    if uploaded:
        suffix     = Path(uploaded.name).suffix.lower()
        bytes_data = uploaded.read()

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name
            _run_analysis(tmp_path, bytes_data)
        except Exception as exc:
            st.error(f"Erro inesperado: `{exc}`")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ── Aba 2: YouTube ─────────────────────────────────────────────────────────
with tab_youtube:
    yt_url      = st.text_input("Cole a URL do vídeo", placeholder="https://www.youtube.com/watch?v=...")
    analyze_btn = st.button("Analisar", disabled=not yt_url)

    if analyze_btn and yt_url:
        if not _is_youtube_url(yt_url):
            st.error("URL inválida. Use um link do YouTube (youtube.com/watch ou youtu.be).")
        else:
            tmp_dir: str | None = None
            try:
                with st.spinner("Baixando áudio do YouTube…"):
                    wav_path, title = _download_youtube(yt_url)
                    tmp_dir = str(Path(wav_path).parent)

                st.caption(f"**{title}**")
                _run_analysis(wav_path, None)

            except Exception as exc:
                st.error(f"Erro ao baixar o vídeo: `{exc}`")
            finally:
                if tmp_dir and os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
