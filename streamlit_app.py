import math
from pathlib import Path
import shutil

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


@st.cache_data
def get_error_filenames(df: pd.DataFrame):
    cond_v1 = ~df["correct_v1"]
    cond_v2 = ~df["correct_v2"]
    cond_diff = df["predicted_v1_norm"] != df["predicted_v2_norm"]
    subset = df[cond_v1 | cond_v2 | cond_diff]
    return set(subset["file"].astype(str))


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


def format_acc(count: int, total: int) -> str:
    if total == 0:
        return "0.00% (0/0)"
    pct = count / total * 100.0
    return f"{pct:.2f}% ({count}/{total})"


df = load_data(CSV_PATH)
error_filenames = get_error_filenames(df)

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")
    actual_filter = st.selectbox(
        "Actual label",
        options=["All", "voicemail", "non_voicemail"],
        index=0,
    )
    pred_filter = st.selectbox(
        "Prediction condition",
        options=[
            "v1 incorrect",
            "v2 incorrect",
            "both incorrect",
            "different between models",
        ],
        index=0,
    )

tab_main, tab_upload = st.tabs(["Overview", "Upload files"])

with tab_upload:
    st.subheader("Upload audio files for error/diff cases")

    uploaded_files = st.file_uploader(
        "Choose file(s)",
        type=["wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        imported = 0
        skipped = 0
        failed = 0

        for up in uploaded_files:
            name = up.name

            if name not in error_filenames:
                skipped += 1
                continue

            data = up.read()
            target_path = AUDIO_DIR / name
            try:
                with open(target_path, "wb") as f:
                    f.write(data)
                if name in st.session_state.audio_cache:
                    del st.session_state.audio_cache[name]
                imported += 1
            except Exception:
                failed += 1

        st.success(f"Imported {imported} file(s) into {AUDIO_DIR}")
        if skipped:
            st.info(f"Skipped {skipped} file(s) not belonging to error/diff cases.")
        if failed:
            st.warning(f"Failed to save {failed} file(s).")

    existing_files = {p.name for p in AUDIO_DIR.glob("*") if p.is_file()}
    matched_error = len(error_filenames.intersection(existing_files))
    st.info(
        f"Files in audio folder matching error/diff cases: "
        f"{matched_error} / {len(error_filenames)}"
    )

    st.write("Sample filenames from CSV (error/diff cases):")
    st.write(list(sorted(error_filenames))[:10])


with tab_main:
    st.subheader("Model Comparison")

    overall_total = len(df)
    overall_correct_v1 = int(df["correct_v1"].sum())
    overall_correct_v2 = int(df["correct_v2"].sum())

    df_vm = df[df["actual_norm"] == "voicemail"]
    df_nvm = df[df["actual_norm"] == "non_voicemail"]

    vm_total = len(df_vm)
    nvm_total = len(df_nvm)

    vm_correct_v1 = int(df_vm["correct_v1"].sum())
    vm_correct_v2 = int(df_vm["correct_v2"].sum())

    nvm_correct_v1 = int(df_nvm["correct_v1"].sum())
    nvm_correct_v2 = int(df_nvm["correct_v2"].sum())

    metrics = pd.DataFrame(
        [
            {
                "Model": "v1",
                "Accuracy": format_acc(overall_correct_v1, overall_total),
                "Accuracy (voicemail)": format_acc(vm_correct_v1, vm_total),
                "Accuracy (non_voicemail)": format_acc(nvm_correct_v1, nvm_total),
            },
            {
                "Model": "v2",
                "Accuracy": format_acc(overall_correct_v2, overall_total),
                "Accuracy (voicemail)": format_acc(vm_correct_v2, vm_total),
                "Accuracy (non_voicemail)": format_acc(nvm_correct_v2, nvm_total),
            },
        ]
    )

    st.dataframe(metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("Browse & Listen")

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
