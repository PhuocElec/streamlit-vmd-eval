import math
import requests
from pathlib import Path

import pandas as pd
import streamlit as st

from settings import settings


# ================== CONFIG ==================
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

PAGE_SIZE = 20

st.set_page_config(
    page_title="Voicemail Model Evaluation",
    layout="wide",
)


# ================== SESSION STATE ==================
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

if "page" not in st.session_state:
    st.session_state.page = 1

if "df_eval" not in st.session_state:
    st.session_state.df_eval = None

if "actual_filter" not in st.session_state:
    st.session_state.actual_filter = "All"

if "pred_filter" not in st.session_state:
    st.session_state.pred_filter = "All"


# ================== UTILS ==================
def norm_label(x: str) -> str:
    s = str(x).strip().lower().replace("-", "_")
    if "voicemail" in s and "non" not in s:
        return "voicemail"
    if "voicemail" in s and "non" in s:
        return "non_voicemail"
    return s


def call_model(api_url: str, filename: str, file_bytes: bytes) -> str:
    files = {"file": (filename, file_bytes)}
    resp = requests.post(api_url, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()["label"]


def get_audio_bytes(filename: str):
    if filename in st.session_state.audio_cache:
        return st.session_state.audio_cache[filename]

    path = AUDIO_DIR / filename
    if path.exists():
        data = path.read_bytes()
        st.session_state.audio_cache[filename] = data
        return data
    return None


def format_acc(correct: int, total: int) -> str:
    if total == 0:
        return "0.00% (0/0)"
    pct = correct / total * 100
    return f"{pct:.2f}% ({correct}/{total})"


# ================== SIDEBAR ==================
with st.sidebar:
    st.subheader("Filters")

    st.session_state.actual_filter = st.selectbox(
        "Actual label",
        ["All", "voicemail", "non_voicemail"],
    )

    st.session_state.pred_filter = st.selectbox(
        "Prediction condition",
        [
            "All",
            "old model incorrect",
            "new model incorrect",
            "both incorrect",
            "different between models",
        ],
    )


# ================== TABS ==================
tab_main, tab_eval = st.tabs(
    ["Overview & Listen", "Evaluate (Upload & Predict)"]
)


# ================== OVERVIEW & LISTEN ==================
with tab_main:
    st.subheader("Overview & Listen")

    if st.session_state.df_eval is None:
        st.info("Please run evaluation in the 'Evaluate' tab first.")
    else:
        df = st.session_state.df_eval.copy()

        # ---- compute on-the-fly ----
        df["_actual"] = df["actual"].apply(norm_label)
        df["_old"] = df["predicted_old"].apply(norm_label)
        df["_new"] = df["predicted_new"].apply(norm_label)

        df["_correct_old"] = df["_actual"] == df["_old"]
        df["_correct_new"] = df["_actual"] == df["_new"]

        # ---------- METRICS ----------
        total = len(df)
        vm = df[df["_actual"] == "voicemail"]
        nvm = df[df["_actual"] == "non_voicemail"]

        metrics = pd.DataFrame(
            [
                {
                    "Model": "Old model",
                    "Accuracy": format_acc(df["_correct_old"].sum(), total),
                    "Accuracy (voicemail)": format_acc(
                        vm["_correct_old"].sum(), len(vm)
                    ),
                    "Accuracy (non_voicemail)": format_acc(
                        nvm["_correct_old"].sum(), len(nvm)
                    ),
                },
                {
                    "Model": "New model",
                    "Accuracy": format_acc(df["_correct_new"].sum(), total),
                    "Accuracy (voicemail)": format_acc(
                        vm["_correct_new"].sum(), len(vm)
                    ),
                    "Accuracy (non_voicemail)": format_acc(
                        nvm["_correct_new"].sum(), len(nvm)
                    ),
                },
            ]
        )

        st.dataframe(metrics, use_container_width=True)

        st.markdown("---")
        st.subheader("Browse & Listen")

        # ---------- FILTER ----------
        filtered = df.copy()

        if st.session_state.actual_filter != "All":
            filtered = filtered[
                filtered["_actual"] == st.session_state.actual_filter
            ]

        pf = st.session_state.pred_filter
        if pf == "old model incorrect":
            filtered = filtered[~filtered["_correct_old"]]
        elif pf == "new model incorrect":
            filtered = filtered[~filtered["_correct_new"]]
        elif pf == "both incorrect":
            filtered = filtered[
                ~filtered["_correct_old"] & ~filtered["_correct_new"]
            ]
        elif pf == "different between models":
            filtered = filtered[filtered["_old"] != filtered["_new"]]

        # ---------- PAGINATION ----------
        total_rows = len(filtered)
        total_pages = max(1, math.ceil(total_rows / PAGE_SIZE))
        st.session_state.page = min(st.session_state.page, total_pages)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Prev") and st.session_state.page > 1:
                st.session_state.page -= 1
        with col3:
            if st.button("Next ➡️") and st.session_state.page < total_pages:
                st.session_state.page += 1
        with col2:
            st.markdown(
                f"<div style='text-align:center'>Page {st.session_state.page}/{total_pages}</div>",
                unsafe_allow_html=True,
            )

        start = (st.session_state.page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        page_df = filtered.iloc[start:end]

        # ---------- AUDIO LIST ----------
        for _, row in page_df.iterrows():
            audio = get_audio_bytes(row["file"])
            c1, c2 = st.columns([5, 3])

            with c1:
                st.markdown(
                    f"""
**{row['file']}**  
Actual: `{row['actual']}`  
Old model: `{row['predicted_old']}` {'✅' if row['_correct_old'] else '❌'}  
New model: `{row['predicted_new']}` {'✅' if row['_correct_new'] else '❌'}
"""
                )

            with c2:
                if audio:
                    st.audio(audio)
                else:
                    st.error("Audio not found")

            st.markdown("---")


# ================== EVALUATE ==================
with tab_eval:
    st.subheader("Upload audio & evaluate models")

    uploaded_files = st.file_uploader(
        "Upload audio files",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
    )

    actual_label = st.selectbox(
        "Actual label (ground truth)",
        ["voicemail", "non_voicemail"],
    )

    if st.button("🚀 Run evaluation") and uploaded_files:
        rows = []

        with st.spinner("Calling model APIs..."):
            for up in uploaded_files:
                audio_bytes = up.read()
                filename = up.name

                (AUDIO_DIR / filename).write_bytes(audio_bytes)
                st.session_state.audio_cache.pop(filename, None)

                try:
                    rows.append(
                        {
                            "file": filename,
                            "actual": actual_label,
                            "predicted_old": call_model(
                                settings.OLD_MODEL_API_URL,
                                filename,
                                audio_bytes,
                            ),
                            "predicted_new": call_model(
                                settings.NEW_MODEL_API_URL,
                                filename,
                                audio_bytes,
                            ),
                        }
                    )
                except Exception as e:
                    st.error(f"Failed on {filename}: {e}")

        df = pd.DataFrame(rows)
        st.session_state.df_eval = df
        st.session_state.page = 1

        st.success(f"Evaluated {len(df)} file(s)")

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="voicemail_model_eval.csv",
            mime="text/csv",
        )

        # 🔥 FORCE RERUN để Overview cập nhật
        st.rerun()
