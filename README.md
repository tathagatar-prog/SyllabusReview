# University Question Bank Review

A Streamlit app for reviewing university questions against an uploaded/pasted syllabus using sentence-transformer semantic embeddings.

## Features
- Save syllabus once per review session.
- Parse questions from pasted text or PDF, DOCX, TXT, and Markdown.
- Compare each question semantically against extracted syllabus topic segments.
- Suggest one keyword: Out of syllabus, Wrong unit, Ambiguous, Incorrect, Repetitive, Other quality issue, or No issue.
- Show best matching topic, unit/heading, similarity score, and likely duplicate question.
- Save review progress locally in `review_data/session.json`.
- Export question reviews and extracted syllabus topics to Excel.

## Local run

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. The first run can take a few minutes. No API key is required. The app does not run a generative LLM; it uses semantic embeddings and explicit rule checks.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository/branch, and set the main file path to `app.py`.
5. Deploy. On first analysis, the app will download the sentence-transformer model; allow the initial startup to complete.

### Cloud resource note
The sentence-transformers/PyTorch stack is larger than a basic Streamlit app. If deployment runs out of memory, try a smaller embedding backend or a hosted inference endpoint. A hosted endpoint may require an account/API token and will transmit question text to that service.

## Important limitations
- Similarity scores rank semantic relatedness; they do not prove syllabus inclusion or exclusion. Thresholds should be calibrated against representative MAKAUT questions and syllabi.
- Unit mapping is inferred from the heading/topic segment where the text parser found a unit/module label. If the syllabus formatting is irregular, inspect the extracted topic sheet.
- This version cannot reliably prove technical incorrectness or ambiguity from question text alone. It flags explicit cues; faculty must verify those categories.
- Local `review_data/session.json` is not guaranteed to persist on Streamlit Community Cloud across restarts/redeployments. Download the Excel file as a backup. For durable multi-user use, add a persistent database (e.g. managed PostgreSQL or Supabase).
