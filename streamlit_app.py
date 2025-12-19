import math
import threading
import time
import hashlib
import queue
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from settings import settings


# ================== CONFIG ==================
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "eval_results.csv"

PAGE_SIZE = 20

st.set_page_config(
    page_title="Voicemail Model Evaluation",
    layout="wide",
)


# ================== PERSIST ==================
def load_df():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(
        columns=["file", "file_hash", "actual", "predicted_old", "predicted_new"]
    )


def append_row(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(
        CSV_PATH,
        mode="a",
        header=not CSV_PATH.exists(),
        index=False,
    )


# ================== SESSION STATE ==================
def init_state():
    ss = st.session_state

    ss.setdefault("audio_cache", {})
    ss.setdefault("page", 1)
    ss.setdefault("actual_filter", "All")
    ss.setdefault("pred_filter", "All")

    ss.setdefault("df_eval", load_df())

    ss.setdefault("running", False)
    ss.setdefault("progress", 0.0)
    ss.setdefault("total_jobs", 0)
    ss.setdefault("done_jobs", 0)

    ss.setdefault("result_queue", queue.Queue())
    ss.setdefault("worker_thread", None)


init_state()


# ================== UTILS ==================
def norm_label(x: str) -> str:
    s = str(x).strip().lower().replace("-", "_")
    if "voicemail" in s and "non" not in s:
        return "voicemail"
    if "voicemail" in s and "non" in s:
        return "non_voicemail"
    return s


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def call_model(api_url: str, filename: str, file_bytes: bytes) -> str:
    files = {"file": (filename, file_bytes)}
    resp = requests.post(api_url, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()["label"]


def get_audio_bytes(filename: str):
    cache = st.session_state.audio_cache
    if filename in cache:
        return cache[filename]

    path = AUDIO_DIR / filename
    if path.exists():
        data = path.read_bytes()
        cache[filename] = data
        return data
    return None


def format_acc(correct: int, total: int) -> str:
    if total == 0:
        return "0.00% (0/0)"
    pct = correct / total * 100
    return f"{pct:.2f}% ({correct}/{total})"


# ================== BACKGROUND WORKER (PURE PYTHON) ==================
def worker(jobs, result_q):
    """
    ❗ KHÔNG dùng st.session_state trong hàm này
    """
    for job in jobs:
        try:
            old = call_model(
                settings.OLD_MODEL_API_URL,
                job["file"],
                job["bytes"],
            )
            new = call_model(
                settings.NEW_MODEL_API_URL,
                job["file"],
                job["bytes"],
            )

            result_q.put(
                {
                    "file": job["file"],
                    "file_hash": job["hash"],
                    "actual": job["actual"],
                    "predicted_old": old,
                    "predicted_new": new,
                }
            )
        except Exception as e:
            result_q.put({"error": str(e)})

    # báo hiệu đã xong
    result_q.put({"_done": True})


# ================== SYNC RESULT QUEUE (MAIN THREAD ONLY) ==================
while not st.session_state.result_queue.empty():
    item = st.session_state.result_queue.get()

    if "_done" in item:
        st.session_state.running = False
        continue

    if "error" in item:
        st.error(item["error"])
        continue

    append_row(item)
    st.session_state.df_eval.loc[len(st.session_state.df_eval)] = item

    st.session_state.done_jobs += 1
    st.session_state.progress = (
        st.session_state.done_jobs / st.session_state.total_jobs
    )


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

    df = st.session_state.df_eval.copy()

    if df.empty:
        st.info("Please run evaluation in the 'Evaluate' tab first.")
    else:
        df["_actual"] = df["actual"].apply(norm_label)
        df["_old"] = df["predicted_old"].apply(norm_label)
        df["_new"] = df["predicted_new"].apply(norm_label)

        df["_correct_old"] = df["_actual"] == df["_old"]
        df["_correct_new"] = df["_actual"] == df["_new"]

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
        disabled=st.session_state.running,
    )

    actual_label = st.selectbox(
        "Actual label (ground truth)",
        ["voicemail", "non_voicemail"],
        disabled=st.session_state.running,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Run evaluation", disabled=st.session_state.running):
            jobs = []
            existing_hashes = set(st.session_state.df_eval["file_hash"])

            for up in uploaded_files or []:
                data = up.read()
                h = file_hash(data)

                if h in existing_hashes:
                    continue

                (AUDIO_DIR / up.name).write_bytes(data)

                jobs.append(
                    {
                        "file": up.name,
                        "bytes": data,
                        "hash": h,
                        "actual": actual_label,
                    }
                )

            if jobs:
                st.session_state.total_jobs = len(jobs)
                st.session_state.done_jobs = 0
                st.session_state.progress = 0
                st.session_state.running = True

                st.session_state.worker_thread = threading.Thread(
                    target=worker,
                    args=(jobs, st.session_state.result_queue),
                    daemon=True,
                )
                st.session_state.worker_thread.start()
            else:
                st.warning("No new files to evaluate.")

    with col2:
        if st.session_state.running:
            st.progress(st.session_state.progress)
            st.info(
                f"Processing {st.session_state.done_jobs}/{st.session_state.total_jobs}"
            )


# ================== AUTO RERUN (MUST BE LAST) ==================
if st.session_state.running:
    time.sleep(0.3)
    st.rerun()
