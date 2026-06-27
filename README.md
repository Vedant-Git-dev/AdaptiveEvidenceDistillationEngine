# AEDE 

### Adaptive Evidence Distillation Engine: The Context Optimization Engine that Respects Your Context Window.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

**AEDE** is a context optimization engine built on top of modern RAG and reasoning workflows designed to solve the "context bloat" problem. By transforming massive datasets into high-signal, distilled evidence, it saves up to **84% in token usage** while significantly reducing the amount of context sent to frontier models.

Stop throwing tokens into the void. Start reasoning with intent.

---

## Results at a Glance

- Up to 84% context reduction
- Tested across 50+ benchmark questions
- Tested on annual reports, research papers, agent conversations, and chat histories
- Adaptive routing with LangGraph
- Multi-model reasoning pipeline
- 90% answer quality retention evaluated by manual answer comparison against ground truth response across 50+ benchmark questions

## Performance Benchmarks

We tested AEDE against industry-standard document types. The results aren't just incremental; they're transformative.

| Context Type | Questions Tested | Reduction Range |
| :--- | :---: | :---: |
| **Annual Reports** | 20+ | **49% – 84%** |
| **Research Papers** | 10+ | **42% – 70%** |
| **Agent Conversations** | 10+ | **43% – 84%** |
| **Conversation History** | 10+ | **55% – 63%** |

*Benchmarks represent token savings compared to raw document injection into Gemini 3.1 Flash Lite without AEDE optimization.*

## Example Result

Question:
> Compare Tesla revenue across the latest two fiscal years.

Raw Context:
- 1,967 tokens

AEDE Optimized Context:
- 315 tokens

Token Reduction:
- 84%

Workflow:
retrieve → extract → analyze → compress → reason

---

## Key Pillars

### 1. Adaptive Evidence Distillation Engine (AEDE)
Traditional RAG retrieves relevant context and delegates all reasoning to the final model. AEDE **extracts, analyzes, and distills**. It identifies core concepts, pulls supporting evidence (claims + quotes), and merges redundancies before the final LLM even sees it.

### 2. Hybrid Intelligence Pipeline
AEDE orchestrates a multi-model dance for maximum efficiency:
- **Distillation Model (Llama 3.1 8B via Groq):** Handles extraction, analysis, and distillation in milliseconds.
- **Reasoning Model (Gemini 3.1 Flash Lite):** Performs the final, high-context reasoning on the optimized evidence.

### 3. Dynamic Workflow Routing
Powered by **LangGraph**, the system makes real-time decisions:
- `retrieve_more`: Not enough coverage? It goes back for more.
- `compress`: Too much noise? It aggressively de-duplicates.
- `direct_answer`: Simple query? It skips the heavy lifting to save time and cost.

### 4. Evidence-First Philosophy
If there's no quote, it didn't happen. AEDE preserves the lineage of every claim, ensuring that the final answer is grounded in verifiable facts from your own data.

---

## The Dashboard: Observability by Design

AEDE isn't just a CLI; it's a full-stack experience. The built-in dashboard provides:
- **Performance Sparklines:** Track token savings and latency in real-time.
- **Workflow Tracing:** See exactly which path the AEDE engine took (`retrieve_more` -> `compress` -> `reason`).
- **Collection Management:** Effortlessly ingest PDFs or paste text blobs into persistent ChromaDB collections.
- **Confidence Metrics:** Visual indicators of "Evidence Coverage" so you know how much to trust the answer.

---

## Tech Stack

- **Backend:** FastAPI, LangGraph, ChromaDB, Pydantic.
- **Frontend:** Next.js 14, Tailwind CSS, Framer Motion, Lucide.
- **AI Models:** Llama 3.1 8B (via Groq), Gemini 3.1 Flash Lite (via Google AI).
- **Embeddings:** Sentence-Transformers (MiniLM-L6-v2).

---

## Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/your-username/AEDE.git
cd AEDE
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:
```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

### 4. Run the Engines
**Start Backend:**
```bash
# From the backend directory
uvicorn api:app --reload --port 8000
```

**Start Frontend:**
```bash
# From the frontend directory
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the magic happen.

---

## Architecture: The AEDE Flow

1. **Extract Concepts:** Identify what the user is actually asking for.
2. **Focused Retrieval:** Pull the most relevant chunks from ChromaDB.
3. **Evidence Extraction:** Turn raw chunks into "Claims + Quotes".
4. **Coverage Analysis:** Check if we have enough information to answer.
5. **Workflow Compiler:** Decide whether to retrieve more, compress, or answer.
6. **Evidence Distillation:** Merge redundant facts into a clean context.
7. **Final Reasoning:** Answer the query using the optimized context.

<br>
<br>

```
                                Any Context Source
                          (PDFs, Agent Chats, Conversations)

                                        ↓

                                    Retrieve

                                        ↓

                                Evidence Extraction
                                        ↓

                                Evidence Analysis
                                        ↓

                              Coverage Evaluation
                                        ↓

                                  Adaptive Routing

                                ┌────────┬────────┐
                                │        │        │
                                ▼        ▼        ▼

                            Direct   Compress   Reason

                                └────────┴────────┘
                                         ↓

                                   Frontier Model
                                         ↓

                                       Answer

```

---

## 🤝 Contributing

We're building the future of context-efficient AI. If you have ideas for better distillation algorithms or smarter routing, we'd love your help!

1. Fork the Project
2. Create your Feature Branch 
3. Commit your Changes 
4. Push to the Branch 
5. Open a Pull Request

---

<br>
<p align="center">
  <strong>Reducing context. Preserving evidence. Improving reasoning.</strong>
</p>
