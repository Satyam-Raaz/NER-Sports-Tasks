import html
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Certificate NER", page_icon="🏅", layout="wide")

COLORS = {
    "SPORTS_NAME": "#a5d8ff",
    "ORG_YEAR": "#ffe066",
    "WINNING_POSITION": "#b2f2bb",
}

SAMPLE = (
    "Raleigh Flynn, your outstanding performance in the Equestrian Dressage "
    "during the 2013 competition has earned you a well-deserved 1st Position. "
    "Your sportsmanship and zeal for the game are appreciated. Congratulations!"
)


def highlight(text: str, entities: list) -> str:
    """Return HTML with every entity highlighted."""
    out, last = [], 0
    for e in sorted(entities, key=lambda x: x["start"]):
        out.append(html.escape(text[last:e["start"]]))
        color = COLORS.get(e["label"], "#dee2e6")
        out.append(
            f'<mark style="background:{color};color:#111;padding:2px 6px;'
            f'border-radius:4px;">{html.escape(e["text"])} '
            f'<small><b>{e["label"]}</b></small></mark>'
        )
        last = e["end"]
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "FastAPI URL", os.getenv("API_URL", "http://127.0.0.1:8080")
    ).rstrip("/")
    st.caption("Must match APP_HOST / PORT of `python app.py`.")

st.title("🏅 Certificate NER")
st.write("Extract **sport name**, **year** and **winning position** from certificate text.")

tab_predict, tab_train = st.tabs(["🔎 Predict", "🏋️ Train"])

# ------------------------------------------------------------------ predict
with tab_predict:
    if "text" not in st.session_state:
        st.session_state["text"] = ""

    if st.button("Use sample text"):
        st.session_state["text"] = SAMPLE

    text = st.text_area(
        "Certificate text", key="text", height=180,
        placeholder="Paste the certificate text here...",
    )

    if st.button("Predict", type="primary"):
        if not text.strip():
            st.warning("Please enter some certificate text.")
        else:
            try:
                with st.spinner("Predicting..."):
                    r = requests.post(f"{api_url}/predict", json={"text": text}, timeout=60)
                if r.status_code != 200:
                    st.error(r.json().get("detail", r.text))
                else:
                    res = r.json()
                    g = res["grouped"]

                    st.subheader("Answer")
                    st.success(res["answer"])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sport", ", ".join(g.get("SPORTS_NAME", [])) or "—")
                    c2.metric("Year", ", ".join(g.get("ORG_YEAR", [])) or "—")
                    c3.metric("Position", ", ".join(g.get("WINNING_POSITION", [])) or "—")

                    st.subheader("Highlighted text")
                    st.markdown(highlight(res["text"], res["entities"]), unsafe_allow_html=True)

                    if res["entities"]:
                        st.subheader("Entities")
                        st.dataframe(
                            pd.DataFrame(res["entities"])[["text", "label", "start", "end"]],
                            width="stretch", hide_index=True,
                        )
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the API at {api_url}. Is `python app.py` running?")
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------------------------------------------------------ train
with tab_train:
    st.write("Runs the training pipeline on the server. This can take several minutes.")
    if st.button("Start training"):
        try:
            with st.spinner("Training... please wait"):
                r = requests.get(f"{api_url}/train", timeout=None)
            if "successful" in r.text.lower():
                st.success(r.text)
            else:
                st.error(r.text)
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the API at {api_url}. Is `python app.py` running?")
        except Exception as e:
            st.error(f"Error: {e}")