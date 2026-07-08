# Resume RAG LLM 🤖

A **Retrieval-Augmented Generation (RAG)** system that answers detailed questions about individual candidates from a directory of resumes. Built with **LlamaIndex**, **Chroma DB**, **Qwen LLMs via OpenRouter**, and a premium **Streamlit** interface.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Single-Candidate QA** | Deep-dive into any candidate's skills, experience, education, or projects |
| ✏️ **Spelling Correction** | "pawan" → "pavanteja kamma", "yasasvi" → "yasaswi kotha" |
| 🚫 **Multi-Resume Blocking** | Comparison, ranking, and aggregation queries are automatically rejected |
| 💬 **Chat History** | Full conversation history with per-message metadata |
| 📊 **Live Stats** | Query counts, correction alerts, and blocked-query counters |
| 🎨 **Premium Dark UI** | Glassmorphism design with smooth animations |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    app.py  (Streamlit UI)                   │
│   Chat window · Stats · Spelling alerts · Blocked alerts    │
└────────────────────────┬────────────────────────────────────┘
                         │ query(str)
┌────────────────────────▼────────────────────────────────────┐
│              src/retriever.py  (FilteredQueryEngine)        │
│  1. QueryClassifier  →  block comparison/multi queries      │
│  2. FuzzyNameMatcher →  resolve candidate name (+ typos)    │
│  3. LlamaIndex + ChromaDB  →  filtered semantic retrieval   │
└─────────────────────────────────────────────────────────────┘
              │                    │                    │
   src/classifier.py      src/matcher.py        src/indexer.py
   LLM / heuristic        rapidfuzz             Build Chroma index
   single vs multi        fuzzy matching         on first run
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Orchestration** | [LlamaIndex ≥ 0.10](https://www.llamaindex.ai/) |
| **LLM** | `qwen/qwen-2.5-72b-instruct` via [OpenRouter](https://openrouter.ai/) |
| **Embeddings** | `qwen/qwen3-embedding-8b` via OpenRouter |
| **Vector DB** | [Chroma DB](https://docs.trychroma.com/) (local, file-based) |
| **PDF Parser** | `pypdf` |
| **DOCX Parser** | `python-docx` |
| **Fuzzy Matching** | `rapidfuzz` |

---

## Project Structure

```
resume-rag-team3/
├── app.py                   # Streamlit dashboard (Person 3)
├── requirements.txt         # Python dependencies
├── .env.template            # API key template
├── implementation_plan.md   # Team design document
│
├── Resumes/                 # Drop PDF/DOCX resumes here
│   ├── ASHOK_Reddy_RESUME - M Ashok reddy.pdf
│   └── ...
│
├── src/
│   ├── config.py            # Config loader (.env + paths)
│   ├── parser.py            # PDF/DOCX parser + name extractor
│   ├── indexer.py           # Build/refresh Chroma DB index
│   ├── matcher.py           # Fuzzy name matching
│   ├── classifier.py        # LLM query classifier
│   └── retriever.py         # Filtered LlamaIndex query engine
│
└── tests/
    ├── test_retrieval.py    # Person 2 retrieval tests
    └── test_rag.py          # Person 3 — end-to-end RAG tests
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) account with an API key

### 1 — Clone the repository

```bash
git clone https://github.com/mmkyadav/Resume-RAG-LLM.git
cd Resume-RAG-LLM
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment

Copy the template and add your API key:

```bash
cp .env.template .env
```

Edit `.env`:

```dotenv
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx
# Optional overrides:
# LLM_MODEL=qwen/qwen-2.5-72b-instruct
# EMBEDDING_MODEL=qwen/qwen3-embedding-8b
```

### 4 — Add resumes

Place your PDF or DOCX resume files in the `Resumes/` directory.

**Required filename format:**
```
<Description> - <Candidate Full Name>.<ext>
```

Examples:
```
ASHOK_Reddy_RESUME - M Ashok reddy.pdf
PAVAN_KAMMA_CV - pavanteja kamma.pdf
Yasaswi_Profile - yasaswi kotha.docx
```

### 5 — Build the vector index

Run the indexer **once** (or whenever resumes change):

```bash
python src/indexer.py
```

This will parse all resumes, generate embeddings via OpenRouter, and store them in `chroma_db/`.

### 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Asking about a candidate

Type a question mentioning a candidate by name:

```
What are Ashok's Python skills?
Tell me about pavanteja's education
Summarize yasasvi's projects
What is Trinadh's contact email?
```

### Spelling corrections

The system automatically corrects common spelling variants and shortforms:

| Input | Resolved to |
|---|---|
| `pawan` | `pavanteja kamma` |
| `yasasvi` | `yasaswi kotha` |
| `trinad` | `Trinadh Kumar Reddi` |
| `krish` | `Mungara Muddu Krishna yadav` |
| `shirly` | `KANDRU SHIRLEY KATHERINE` |

A yellow alert banner will inform you of any applied correction.

### Blocked queries (by design)

The following query types are **rejected** with an explanatory message:

- **Comparison:** `"Compare Ashok and Pawan"`
- **Ranking:** `"Who is the best candidate for Java?"`
- **Listing/Aggregation:** `"List all candidates with React experience"`
- **Shortlisting:** `"Shortlist top 3 for data science"`
- **No name mentioned:** `"What is Python?"`

---

## Running Tests

### End-to-end RAG tests (Person 3)

```bash
pytest tests/test_rag.py -v
```

### Retrieval pipeline tests (Person 2)

```bash
pytest tests/test_retrieval.py -v
```

### All tests

```bash
pytest tests/ -v
```

> **Note:** Tests that require the `Resumes/` directory to be populated will be automatically **skipped** if no resume files are present, and will run in full when the directory contains resume files.

---

## Team & Branch Strategy

| Person | Branch | Responsibility |
|---|---|---|
| **Person 1** | `dev-person-1-ingestion` | Config, parsers, Chroma DB indexer |
| **Person 2** | `dev-person-2-retrieval` | Fuzzy matcher, classifier, filtered retriever |
| **Person 3** | `dev-person-3-frontend` | Streamlit UI, chat integration, tests, README |

All branches merge into `main` sequentially.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `OPENROUTER_API_KEY is not set` | Create `.env` and add your key |
| `Resumes directory does not exist` | Create `Resumes/` and add files |
| `Chroma DB not found / empty` | Run `python src/indexer.py` first |
| `Mock Mode` responses | API key missing/invalid — add valid key to `.env` |
| Import errors | Run `pip install -r requirements.txt` |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
