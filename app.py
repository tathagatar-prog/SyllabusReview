import io
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = Path("review_data")
SESSION_FILE = DATA_DIR / "session.json"
MAX_QUESTIONS = 120

CATEGORIES = [
    "Out of syllabus",
    "Wrong unit",
    "Ambiguous",
    "Incorrect",
    "Repetitive",
    "Other quality issue",
    "No issue",
]

st.set_page_config(page_title="University Question Review", page_icon="📘", layout="centered")
st.title("University Question Review")
st.caption("Paste the syllabus once. Then review questions one by one, save each decision, and continue.")


def fresh_state():
    return {
        "syllabus": "",
        "subject": "",
        "questions": [""] * MAX_QUESTIONS,
        "categories": [""] * MAX_QUESTIONS,
        "remarks": [""] * MAX_QUESTIONS,
        "reasons": [""] * MAX_QUESTIONS,
        "topics": [""] * MAX_QUESTIONS,
        "units": [""] * MAX_QUESTIONS,
        "scores": [None] * MAX_QUESTIONS,
        "analysis": [{} for _ in range(MAX_QUESTIONS)],
        "statuses": ["Pending"] * MAX_QUESTIONS,
    }


def load_state():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION_FILE.exists():
        try:
            saved = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            base = fresh_state()
            base.update(saved)
            defaults = fresh_state()
            for key in ("questions", "categories", "remarks", "reasons", "topics", "units", "scores", "statuses"):
                values = list(base.get(key, []) or [])
                default_values = defaults[key]
                # Always restore arrays to exactly 120 entries; old app versions may
                # have saved shorter arrays or used a different key for analysis.
                values = values[:MAX_QUESTIONS]
                values.extend(default_values[len(values):MAX_QUESTIONS])
                base[key] = values
            return base
        except Exception:
            return fresh_state()
    return fresh_state()


def save_state():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st.session_state.review, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SESSION_FILE)


@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def chunks_from_syllabus(text):
    lines = [re.sub(r"\s+", " ", line).strip(" \t•-") for line in text.splitlines()]
    lines = [line for line in lines if line]
    topics = []
    current_unit = "Unit not specified"
    for line in lines:
        if re.match(r"(?i)^(unit|module|part)\s*([0-9ivx]+)?", line):
            current_unit = line[:100]
            rest = re.sub(r"(?i)^(unit|module|part)\s*([0-9ivx]+)?\s*[:.\-–]?\s*", "", line).strip()
            if len(rest.split()) >= 2:
                topics.append({"unit": current_unit, "topic": rest})
            continue
        parts = re.split(r"(?<=[.;])\s+|[•▪]", line)
        for part in parts:
            part = part.strip(" .;:-")
            if len(part.split()) >= 2:
                topics.append({"unit": current_unit, "topic": part})
    if not topics and text.strip():
        topics = [{"unit": "Unit not specified", "topic": x.strip()} for x in re.split(r"[.;\n]+", text) if len(x.split()) >= 2]
    # remove duplicates
    out, seen = [], set()
    for item in topics:
        k = (item["unit"].lower(), item["topic"].lower())
        if k not in seen:
            out.append(item)
            seen.add(k)
    return out[:500]


def token_set(text):
    stop = {"what", "when", "where", "which", "who", "why", "how", "explain", "describe",
            "define", "discuss", "write", "state", "give", "list", "with", "from", "into",
            "that", "this", "the", "and", "for", "are", "was", "were", "using", "use"}
    return {w for w in re.findall(r"[a-z0-9+#.-]{2,}", text.lower()) if w not in stop}


def analyze_question(question, syllabus, topics, previous_questions):
    if not syllabus.strip() or not topics:
        return {
            "category": "Other quality issue",
            "topic": "",
            "unit": "",
            "score": 0.0,
            "reason": "Paste and save the syllabus first so the question can be compared with it.",
        }

    encoder = load_model()
    topic_texts = [f"{t['unit']}. {t['topic']}" for t in topics]
    topic_embeddings = encoder.encode(topic_texts, normalize_embeddings=True, show_progress_bar=False)
    q_embedding = encoder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    similarities = topic_embeddings @ q_embedding
    best_i = int(similarities.argmax())
    score = float(similarities[best_i])
    best = topics[best_i]

    duplicate_question, duplicate_score = "", 0.0
    nonempty = [q for q in previous_questions if q.strip()]
    if nonempty:
        prev_embeddings = encoder.encode(nonempty, normalize_embeddings=True, show_progress_bar=False)
        sims = prev_embeddings @ q_embedding
        j = int(sims.argmax())
        duplicate_score = float(sims[j])
        duplicate_question = nonempty[j]

    low = question.lower()
    if duplicate_score >= 0.88:
        category = "Repetitive"
        reason = f"This question is highly similar to an earlier question (semantic similarity {duplicate_score:.2f})."
    elif score < 0.30:
        category = "Out of syllabus"
        reason = f"No strong semantic match was found in the pasted syllabus (best match score {score:.2f}). Please verify before finalizing."
    elif any(x in low for x in ["ambiguous", "unclear wording", "not clear", "multiple interpretations"]):
        category = "Ambiguous"
        reason = "The wording contains a possible clarity cue. Confirm whether students could interpret it in more than one way."
    elif any(x in low for x in ["incorrect", "wrong answer", "erroneous", "contains an error"]):
        category = "Incorrect"
        reason = "The text contains a possible error cue. Verify the technical content or model answer before finalizing."
    else:
        category = "No issue"
        reason = "A syllabus topic appears semantically relevant. Check the unit mapping and technical correctness manually."

    return {
        "category": category,
        "topic": best["topic"],
        "unit": best["unit"],
        "score": round(score, 3),
        "duplicate": duplicate_question,
        "duplicate_score": round(duplicate_score, 3),
        "reason": reason,
    }


if "review" not in st.session_state:
    st.session_state.review = load_state()
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
data = st.session_state.review
# Defensive normalization for any session loaded from an earlier version.
for _key, _default in fresh_state().items():
    if isinstance(_default, list):
        _values = list(data.get(_key, []) or [])
        _values = _values[:MAX_QUESTIONS]
        _values.extend(_default[len(_values):MAX_QUESTIONS])
        data[_key] = _values

with st.sidebar:
    st.subheader("Review progress")
    done = sum(s == "Reviewed" for s in data["statuses"])
    st.metric("Saved questions", f"{done}/{MAX_QUESTIONS}")
    if st.button("Start new subject / clear all", use_container_width=True):
        st.session_state.review = fresh_state()
        st.session_state.q_index = 0
        for key in list(st.session_state.keys()):
            if key.startswith(("question_input_", "remark_input_", "category_input_")):
                del st.session_state[key]
        save_state()
        st.rerun()

st.subheader("1. Paste syllabus")
st.text_input("Subject name (optional)", key="subject_entry", value=data["subject"])
syllabus_widget = st.text_area(
    "Complete syllabus",
    value=data["syllabus"],
    height=220,
    key="syllabus_entry",
    placeholder="Paste the entire syllabus here, including unit numbers, topics and subtopics...",
)
if st.button("Save syllabus", type="primary"):
    if not syllabus_widget.strip():
        st.error("Paste the syllabus into the box first.")
    else:
        data["subject"] = st.session_state.subject_entry.strip()
        data["syllabus"] = syllabus_widget.strip()
        data["topics"] = []  # topics are derived from the saved syllabus at analysis time
        save_state()
        st.success("Syllabus saved. You do not need to paste it again for this review.")
        st.rerun()

if data["syllabus"].strip():
    st.success("Syllabus saved ✓")
else:
    st.info("Paste and save the syllabus before reviewing questions.")

st.divider()
st.subheader("2. Enter questions one by one")
idx = min(st.session_state.q_index, MAX_QUESTIONS - 1)
st.progress((idx + 1) / MAX_QUESTIONS, text=f"Question slot {idx + 1} of {MAX_QUESTIONS}")
nav_a, nav_b, nav_c = st.columns([1, 2, 1])
with nav_a:
    if st.button("← Previous", disabled=idx == 0, use_container_width=True):
        st.session_state.q_index -= 1
        st.rerun()
with nav_b:
    jump = st.selectbox("Go to question number", list(range(1, MAX_QUESTIONS + 1)), index=idx, key="jump_to_question")
    if jump - 1 != idx:
        st.session_state.q_index = jump - 1
        st.rerun()
with nav_c:
    if st.button("Next →", disabled=idx >= MAX_QUESTIONS - 1, use_container_width=True):
        st.session_state.q_index += 1
        st.rerun()

question_key = f"question_input_{idx}"
if question_key not in st.session_state:
    st.session_state[question_key] = data["questions"][idx]
question = st.text_area(
    f"Question {idx + 1}",
    key=question_key,
    height=120,
    placeholder="Paste or type only this question here...",
)
old_result = data.get("reasons", [""] * MAX_QUESTIONS)[idx]
analyze_btn = st.button("Analyze this question", type="primary", disabled=not data["syllabus"].strip() or not question.strip())

if analyze_btn:
    topics = chunks_from_syllabus(data["syllabus"])
    earlier_questions = [q for i, q in enumerate(data["questions"]) if i != idx and q.strip()]
    with st.spinner("Comparing question meaning with the syllabus and previously saved questions..."):
        try:
            result = analyze_question(question.strip(), data["syllabus"], topics, earlier_questions)
            data["questions"][idx] = question.strip()
            data["categories"][idx] = result["category"]
            data["remarks"][idx] = result["category"]
            data["reasons"][idx] = result["reason"]
            data["topics"][idx] = result.get("topic", "")
            data["units"][idx] = result.get("unit", "")
            data["scores"][idx] = result.get("score")
            data["statuses"][idx] = "Pending"  # becomes Reviewed only after explicit save
            data.setdefault("analysis", [{} for _ in range(MAX_QUESTIONS)])
            data["analysis"][idx] = result
            save_state()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load/run the semantic model: {exc}. Check internet access and requirements.txt.")

result = {}
if data.get("analysis") and len(data["analysis"]) > idx:
    result = data["analysis"][idx] or {}
if result and data["questions"][idx]:
    st.markdown("#### AI suggestion")
    st.markdown(f"**Suggested keyword:** `{result.get('category', 'Other quality issue')}`")
    st.write(f"**Best matching syllabus topic:** {result.get('topic') or 'No clear match'}")
    st.write(f"**Unit/heading:** {result.get('unit') or 'Not identified'}")
    st.write(f"**Semantic similarity:** {result.get('score', 0):.3f}")
    st.write(f"**Reason:** {result.get('reason', '')}")
    if result.get("duplicate"):
        st.write(f"**Similar earlier question:** {result['duplicate']}")
        st.write(f"**Duplicate similarity:** {result.get('duplicate_score', 0):.3f}")

    st.markdown("#### Confirm and save")
    category_key = f"category_input_{idx}"
    if category_key not in st.session_state:
        st.session_state[category_key] = data["categories"][idx]
    chosen = st.selectbox("Remark keyword", CATEGORIES, key=category_key)
    remark_key = f"remark_input_{idx}"
    if remark_key not in st.session_state:
        st.session_state[remark_key] = chosen
    faculty_remark = st.text_input("Faculty remark (keyword only)", key=remark_key)
    st.caption("The app suggests one keyword. Confirm or change it after checking the match and reason.")
    save_a, save_b = st.columns([2, 1])
    with save_a:
        save_one = st.button("Save this question", type="primary", use_container_width=True)
    with save_b:
        save_next = st.button("Save and next", use_container_width=True, disabled=idx >= MAX_QUESTIONS - 1)
    if save_one or save_next:
        data["questions"][idx] = question.strip()
        data["categories"][idx] = chosen
        data["remarks"][idx] = faculty_remark.strip() or chosen
        data["statuses"][idx] = "Reviewed"
        save_state()
        if save_next:
            st.session_state.q_index = min(MAX_QUESTIONS - 1, idx + 1)
        st.success(f"Question {idx + 1} saved.")
        st.rerun()

st.divider()
st.subheader("3. Saved questions and Excel export")
rows = []
for i in range(MAX_QUESTIONS):
    if data["questions"][i].strip():
        rows.append({
            "Question No.": i + 1,
            "Question": data["questions"][i],
            "Suggested Topic": data["topics"][i],
            "Syllabus Unit/Heading": data["units"][i],
            "Similarity Score": data["scores"][i] if data["scores"][i] is not None else "",
            "Remark Category": data["categories"][i],
            "Faculty Remark": data["remarks"][i],
            "Reason": data["reasons"][i],
            "Status": data["statuses"][i],
        })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Review")
        pd.DataFrame(chunks_from_syllabus(data["syllabus"])).to_excel(writer, index=False, sheet_name="Syllabus Topics")
    st.download_button(
        "Download saved review as Excel",
        data=buf.getvalue(),
        file_name="university_question_review.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.caption("Saved questions will appear here. You can enter up to 120 questions one by one.")

st.caption("Progress is saved in review_data/session.json in the app runtime. On Streamlit Community Cloud, local files may be lost after a restart or redeployment; download Excel backups regularly.")
