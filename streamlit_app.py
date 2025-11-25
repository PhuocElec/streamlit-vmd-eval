import math
from pathlib import Path

import pandas as pd
import streamlit as st

CSV_PATH = Path("vmd_eval.csv")

AUDIO_DIR = Path("audio")
if "AUDIO_DIR" in st.secrets:
    try:
        AUDIO_DIR = Path(st.secrets["AUDIO_DIR"])
    except Exception:
        pass

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

PAGE_SIZE = 20

st.set_page_config(page_title="Model Comparison - Voicemail", layout="wide")

if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

if "page" not in st.session_state:
    st.session_state.page = 1


@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    def norm_label(x: str) -> str:
        s = str(x).strip().lower().replace("-", "_")
        if "voicemail" in s and "non" not in s:
            return "voicemail"
        if "voicemail" in s and "non" in s:
            return "non_voicemail"
        return s

    df["actual_norm"] = df["actual"].apply(norm_label)
    df["predicted_v1_norm"] = df["predicted_v1"].apply(norm_label)
    df["predicted_v2_norm"] = df["predicted_v2"].apply(norm_label)

    df["correct_v1"] = df["actual_norm"] == df["predicted_v1_norm"]
    df["correct_v2"] = df["actual_norm"] == df["predicted_v2_norm"]

    return df


def get_audio_bytes(filename: str):
    if filename in st.session_state.audio_cache:
        return st.session_state.audio_cache[filename]

    local_path = AUDIO_DIR / filename
    try:
        if local_path.exists():
            with open(local_path, "rb") as f:
                data = f.read()
            st.session_state.audio_cache[filename] = data
            return data
    except Exception:
        pass

    return None


def safe_acc(series):
    if len(series) == 0:
        return 0.0
    return float(series.mean())


df = load_data(CSV_PATH)

with st.sidebar:
    st.markdown("### Audio folder")
    current_dir = "" if AUDIO_DIR is None else str(AUDIO_DIR)
    audio_dir_input = st.text_input("Path", value=current_dir, placeholder="audio or /data/audio")
    if audio_dir_input.strip():
        AUDIO_DIR = Path(audio_dir_input.strip())
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if AUDIO_DIR is not None and not AUDIO_DIR.exists():
        st.warning(f"Folder does not exist: {AUDIO_DIR}")
    st.markdown("---")

tab_main, tab_upload = st.tabs(["Overview", "Upload Audio"])

with tab_upload:
    st.subheader("Upload audio files (filename must match CSV `file`)")

    uploaded_files = st.file_uploader(
        "Choose file(s)",
        type=["wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        added = 0
        for up in uploaded_files:
            name = up.name
            data = up.read()
            target_path = AUDIO_DIR / name
            try:
                with open(target_path, "wb") as f:
                    f.write(data)
                if name in st.session_state.audio_cache:
                    del st.session_state.audio_cache[name]
                added += 1
            except Exception as e:
                st.error(f"Failed to save {name}: {e}")

        st.success(f"Saved {added} file(s) into {AUDIO_DIR}")

    existing_files = {p.name for p in AUDIO_DIR.glob("*") if p.is_file()}
    matched = len(set(df["file"].astype(str)).intersection(existing_files))
    st.info(f"Files in audio folder matching CSV: {matched} / {df['file'].nunique()}")

    st.write("Sample filenames from CSV:")
    st.write(df["file"].head(10).tolist())


with tab_main:
    st.subheader("Model Comparison")

    overall_acc_v1 = safe_acc(df["correct_v1"])
    overall_acc_v2 = safe_acc(df["correct_v2"])

    df_vm = df[df["actual_norm"] == "voicemail"]
    df_nvm = df[df["actual_norm"] == "non_voicemail"]

    vm_acc_v1 = safe_acc(df_vm["correct_v1"])
    vm_acc_v2 = safe_acc(df_vm["correct_v2"])

    nvm_acc_v1 = safe_acc(df_nvm["correct_v1"])
    nvm_acc_v2 = safe_acc(df_nvm["correct_v2"])

    metrics = pd.DataFrame(
        [
            {
                "Model": "v1",
                "Accuracy": f"{overall_acc_v1 * 100:.2f}%",
                "Accuracy (voicemail)": f"{vm_acc_v1 * 100:.2f}%",
                "Accuracy (non_voicemail)": f"{nvm_acc_v1 * 100:.2f}%",
            },
            {
                "Model": "v2",
                "Accuracy": f"{overall_acc_v2 * 100:.2f}%",
                "Accuracy (voicemail)": f"{vm_acc_v2 * 100:.2f}%",
                "Accuracy (non_voicemail)": f"{nvm_acc_v2 * 100:.2f}%",
            },
        ]
    )

    st.dataframe(metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("Browse & Listen")

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        actual_filter = st.selectbox(
            "Actual label",
            options=["All", "voicemail", "non_voicemail"],
            index=0,
        )

    with col_filter2:
        pred_filter = st.selectbox(
            "Prediction condition",
            options=[
                "All",
                "v1 incorrect",
                "v2 incorrect",
                "both incorrect",
                "different between models",
            ],
            index=0,
        )

    filtered = df.copy()

    if actual_filter == "voicemail":
        filtered = filtered[filtered["actual_norm"] == "voicemail"]
    elif actual_filter == "non_voicemail":
        filtered = filtered[filtered["actual_norm"] == "non_voicemail"]

    if pred_filter == "v1 incorrect":
        filtered = filtered[~filtered["correct_v1"]]
    elif pred_filter == "v2 incorrect":
        filtered = filtered[~filtered["correct_v2"]]
    elif pred_filter == "both incorrect":
        filtered = filtered[~filtered["correct_v1"] & ~filtered["correct_v2"]]
    elif pred_filter == "different between models":
        filtered = filtered[
            filtered["predicted_v1_norm"] != filtered["predicted_v2_norm"]
        ]

    st.write(f"Total samples after filtering: **{len(filtered)}**")

    total_rows = len(filtered)
    total_pages = max(1, math.ceil(total_rows / PAGE_SIZE))

    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])

    with col_p1:
        if st.button("⬅️ Prev") and st.session_state.page > 1:
            st.session_state.page -= 1

    with col_p3:
        if st.button("Next ➡️") and st.session_state.page < total_pages:
            st.session_state.page += 1

    with col_p2:
        st.markdown(
            f"<div style='text-align:center'>Page {st.session_state.page}/{total_pages}</div>",
            unsafe_allow_html=True,
        )

    start_idx = (st.session_state.page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = filtered.iloc[start_idx:end_idx]

    st.markdown("---")

    for _, row in page_df.iterrows():
        filename = str(row["file"])
        audio_bytes = get_audio_bytes(filename)

        cols = st.columns([5, 3])

        with cols[0]:
            status_v1 = "✅" if row["correct_v1"] else "❌"
            status_v2 = "✅" if row["correct_v2"] else "❌"

            st.markdown(
                f"**{filename}**  \n"
                f"Actual: `{row['actual']}` | "
                f"v1: `{row['predicted_v1']}` ({status_v1}) | "
                f"v2: `{row['predicted_v2']}` ({status_v2})"
            )

        with cols[1]:
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.error(f"Audio not found: {filename}")

        st.markdown("---")
