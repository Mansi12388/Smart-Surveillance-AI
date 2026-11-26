"""
Streamlit dashboard for Smart Surveillance AI.

1) Start the backend  :  uvicorn backend.main:app --port 8000
2) Start this app     :  streamlit run frontend/app.py
Both commands are run from the project root.
"""
import base64
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import API_URL  # noqa: E402

st.set_page_config(page_title="Smart Surveillance AI", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------ session state
st.session_state.setdefault("api_url", API_URL)
st.session_state.setdefault("safety_duration", 120.0)
st.session_state.setdefault("safety_people", 4)
st.session_state.setdefault("safety_mask_issues", 0)
st.session_state.setdefault("safety_motion", 3)
st.session_state.setdefault("incident_text", "")

INCIDENT_EXAMPLES = [
    "Person fell near the entrance while walking.",
    "A man ran quickly through the corridor.",
    "The elevator door is broken and a light is flickering.",
    "Routine patrol completed, everything looked normal.",
]


# ------------------------------------------------------------------ helpers
def api(method, path, timeout=120, **kwargs):
    """Call the backend. Returns (data, error_message); exactly one is None."""
    url = st.session_state["api_url"].rstrip("/") + path
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, ("Cannot reach the backend. Start it from the project root with:  "
                      "uvicorn backend.main:app --port 8000")
    except requests.exceptions.Timeout:
        return None, "The request timed out. The first call loads the model and can be slow - try again."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}"

    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("detail", response.text) if isinstance(body, dict) else response.text
        except ValueError:
            detail = response.text
        return None, f"Error {response.status_code}: {detail}"
    return response.json(), None


def show_probabilities(probabilities):
    st.bar_chart(pd.Series(probabilities, name="probability"))


def set_incident_example(text):
    st.session_state["incident_text"] = text


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛡️ Smart Surveillance AI")
    st.text_input("Backend URL", key="api_url")
    if st.button("Check backend"):
        health, err = api("GET", "/health", timeout=10)
        if err:
            st.error(err)
        else:
            st.success("Backend is running")
            for name, ok in health["models"].items():
                st.write(("✅ " if ok else "❌ ") + name)
    st.caption("A ❌ means the model file is missing - run the matching notebook (02-07).")

st.title("Smart Surveillance AI")
st.caption("Face-mask detection · activity recognition · incident text analysis · safety-event prioritisation")

tab_mask, tab_activity, tab_incident, tab_safety, tab_dash = st.tabs(
    ["😷 Face mask", "🏃 Activity", "📝 Incident text", "⚠️ Safety event", "📊 Dashboard"]
)

# --------------------------------------------------------------- face mask
with tab_mask:
    st.subheader("Face-mask detection")
    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    if source == "Upload":
        image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="mask_upload")
    else:
        image_file = st.camera_input("Take a photo")
    use_detector = st.checkbox(
        "Detect faces first (OpenCV)", value=True,
        help="Turn off for tightly cropped face images. If no face is found the whole image is classified.")

    if st.button("Analyze image", disabled=image_file is None, key="btn_mask"):
        with st.spinner("Analyzing..."):
            data, err = api(
                "POST", "/mask/predict", timeout=120,
                params={"use_face_detector": str(use_detector).lower(), "annotate": "true"},
                files={"file": (image_file.name or "image.jpg", image_file.getvalue(),
                                image_file.type or "image/jpeg")},
            )
        if err:
            st.error(err)
        else:
            st.session_state["mask_result"] = data
            # pre-fill the Safety tab with what was just measured
            st.session_state["safety_people"] = int(data["summary"]["people_count"])
            st.session_state["safety_mask_issues"] = int(data["summary"]["mask_issue_count"])

    result = st.session_state.get("mask_result")
    if result:
        left, right = st.columns(2)
        with left:
            if result.get("annotated_image"):
                st.image(base64.b64decode(result["annotated_image"]), caption="Result")
        with right:
            summary = result["summary"]
            c1, c2 = st.columns(2)
            c1.metric("People", summary["people_count"])
            c2.metric("Mask issues", summary["mask_issue_count"])
            st.write(f"Overall: **{result['label']}**")
            st.dataframe(pd.DataFrame([
                {"label": d["label"], "confidence": round(d["confidence"], 3), "face detected": d["face_detected"]}
                for d in result["detections"]
            ]))
            if not any(d["face_detected"] for d in result["detections"]):
                st.caption("No face was found by the detector, so the whole image was classified as one face.")
            st.caption("People and mask-issue counts were copied to the Safety tab.")

# ---------------------------------------------------------------- activity
with tab_activity:
    st.subheader("Human activity recognition")
    st.caption("Proof of concept: trained on only 4 sample videos (Walking, Sitting, Standing).")
    video_file = st.file_uploader("Choose a short video", type=["avi", "mp4", "mov", "mkv", "webm"],
                                  key="activity_upload")
    model_type = st.radio("Model", ["cnn_lstm", "bilstm"], horizontal=True)
    if video_file is not None:
        st.video(video_file)

    if st.button("Recognize activity", disabled=video_file is None, key="btn_activity"):
        with st.spinner("Extracting features and classifying (the first call is slow)..."):
            data, err = api(
                "POST", "/activity/predict", timeout=300,
                params={"model_type": model_type},
                files={"file": (video_file.name, video_file.getvalue(), video_file.type or "video/mp4")},
            )
        if err:
            st.error(err)
        else:
            st.session_state["activity_result"] = data

    result = st.session_state.get("activity_result")
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Activity", result["label"])
        c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        c3.metric("Windows analysed", result["num_windows"])
        show_probabilities(result["probabilities"])

# ---------------------------------------------------------------- incident
with tab_incident:
    st.subheader("Incident description analysis (BERT)")
    st.write("Try an example:")
    example_cols = st.columns(len(INCIDENT_EXAMPLES))
    for i, (col, example) in enumerate(zip(example_cols, INCIDENT_EXAMPLES)):
        col.button(example[:30] + "...", key=f"example_{i}", on_click=set_incident_example, args=(example,))

    incident_text = st.text_area("Incident description", key="incident_text", height=100, max_chars=500)

    if st.button("Classify incident", disabled=len(incident_text.strip()) < 3, key="btn_incident"):
        with st.spinner("Classifying..."):
            data, err = api("POST", "/incident/predict", timeout=120, json={"text": incident_text.strip()})
        if err:
            st.error(err)
        else:
            st.session_state["incident_result"] = data

    result = st.session_state.get("incident_result")
    if result:
        c1, c2 = st.columns(2)
        c1.metric("Category", result["label"])
        c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.info(result["summary"])
        show_probabilities(result["probabilities"])

# ------------------------------------------------------------------ safety
with tab_safety:
    st.subheader("Safety-event prioritisation (ANN)")
    st.caption("Classifies an event record into a predefined category. It is a prioritisation aid trained on "
               "synthetic data, not an assessment of any person.")
    c1, c2 = st.columns(2)
    c1.number_input("Duration of unusual activity (seconds)", min_value=0.0, step=10.0, key="safety_duration")
    c1.number_input("Number of people detected", min_value=0, step=1, key="safety_people")
    c2.number_input("Mask issues (no mask / incorrect)", min_value=0, step=1, key="safety_mask_issues")
    c2.number_input("Motion changes (activity frequency)", min_value=0, step=1, key="safety_motion")

    if st.button("Prioritise event", key="btn_safety"):
        payload = {
            "duration_seconds": float(st.session_state["safety_duration"]),
            "people_count": int(st.session_state["safety_people"]),
            "mask_issue_count": int(st.session_state["safety_mask_issues"]),
            "motion_change_count": int(st.session_state["safety_motion"]),
        }
        with st.spinner("Classifying..."):
            data, err = api("POST", "/safety/predict", timeout=120, json=payload)
        if err:
            st.error(err)
        else:
            st.session_state["safety_result"] = data

    result = st.session_state.get("safety_result")
    if result:
        message = f"Priority: {result['priority']}  |  category: {result['label']}  ({result['confidence'] * 100:.1f}%)"
        {"Low": st.success, "Medium": st.warning, "High": st.error}.get(result["priority"], st.info)(message)
        show_probabilities(result["probabilities"])

# --------------------------------------------------------------- dashboard
with tab_dash:
    st.subheader("Event log")
    stats, stats_err = api("GET", "/stats", timeout=15)
    events, events_err = api("GET", "/events", timeout=15, params={"limit": 100})

    if stats_err or events_err:
        st.info(stats_err or events_err)
    else:
        cols = st.columns(5)
        cols[0].metric("Total events", stats["total"])
        for col, module in zip(cols[1:], ["mask", "activity", "incident", "safety"]):
            col.metric(module.title(), stats["by_module"].get(module, 0))

        if events:
            st.write("Latest events")
            st.dataframe(pd.DataFrame(events)[["id", "created_at", "module", "label", "confidence", "input_text"]])
            st.write("Counts by module and label")
            st.dataframe(pd.DataFrame(stats["by_label"]))
        else:
            st.caption("No events yet - analyze something in the other tabs.")

        with st.expander("Danger zone"):
            confirm = st.checkbox("I want to delete ALL events")
            if st.button("Delete all events", disabled=not confirm):
                _, err = api("DELETE", "/events", timeout=15)
                if err:
                    st.error(err)
                else:
                    st.rerun()