# Adaptive Evidence-Based Workflow Compiler (AEWC)

## Plan

---

## 1. Project Overview

**Type:** LangGraph-based multi-agent system
**Core Goal:** Compile the smallest reasoning workflow needed per query, optimizing for token efficiency, low latency, and answer quality.
**Target Users:** RAG system developers, LLM application engineers

---

## 2. Architecture

```
START
  │
  ▼
Workflow Compiler (router)
  │
  ├──────────────────────────────────────────────────────────────┐
  │  Path A: Simple (no reasoning needed)                        │
  │  Retrieve ──► Answer                                         │
  │                                                              │
  │  Path B: Compressed (evidence needs synthesis)               │
  │  Retrieve ──► Evidence Extract ──► Compression ──► Answer   │
  │                                                              │
  │  Path C: Contradictory (conflicts detected)                  │
  │  Retrieve ──► Evidence Extract ──► Contradiction ──►         │
  │               Negotiation ──► Compression ──► Answer         │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

```
/media/vedant/Storage/My Projects/Agentic AI/SelfTrainer/
├── src/
│   └── aewc/
│       ├── __init__.py
│       ├── graph.py              # Main LangGraph StateGraph
│       ├── state.py              # Shared state schema
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── compiler.py       # Workflow compiler/router
│       │   ├── retrieval.py      # FAISS/ChromaDB retrieval
│       │   ├── evidence.py       # Claim extraction agent
│       │   ├── contradiction.py   # Conflict detection
│       │   ├── negotiation.py     # Evidence filtering
│       │   ├── compression.py     # Context compression
│       │   └── answer.py         # Final answer generation
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── token_tracker.py  # Token counting/budgeting
│       │   └── metrics.py        # Evaluation metrics
│       └── config.py             # Settings
├── tests/
│   ├── __init__.py
│   ├── test_compiler.py
│   ├── test_evidence.py
│   ├── test_contradiction.py
│   └── test_integration.py
├── configs/
│   └── settings.yaml
├── data/
│   └── sample_docs/              # For testing
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
└── README.md
```

---

## 4. Implementation Phases

### Phase 1: Core Infrastructure
1. **Project setup** — pyproject.toml, virtualenv, directory structure
2. **State schema** — Define `AEWCState` with all fields: query, documents, claims, contradictions, workflow_path, token_budget, etc.
3. **Base graph** — Empty LangGraph that can route to 3 paths

### Phase 2: Single-Path Agents (simplest → most complex)

**Step 2.1: Retrieval Agent**
- Input: query
- Output: retrieved documents (chunks)
- Use: FAISS or ChromaDB (configurable)
- No reasoning—just retrieval

**Step 2.2: Evidence Extraction Agent**
- Input: documents
- Output: atomic claims
- Model: gemma2-9b or llama-3.3-70b (Groq free tier)
- Prompt: "Extract all atomic factual claims. One fact per claim."

**Step 2.3: Simple Answer Agent (Path A)**
- Input: documents
- Output: answer
- Used when query complexity = LOW

### Phase 3: Advanced Agents

**Step 3.1: Contradiction Agent**
- Input: claims
- Output: `{agreements: [...], contradictions: [...]}`
- Detect numeric conflicts, semantic opposites
- Use: pairwise claim comparison with LLM

**Step 3.2: Negotiation Agent**
- Input: contradictions, claims
- Output: `{accepted: [...], uncertain: [...], rejected: [...]}`
- Resolve conflicts, filter evidence

**Step 3.3: Compression Agent**
- Input: claims
- Output: synthesized findings (5-10x compression)
- Merge redundant claims into single statements

### Phase 4: Workflow Compiler

**Step 4.1: Query Classifier**
- Classify query complexity: LOW / MEDIUM / HIGH
- LOW → Path A
- MEDIUM → Path B
- HIGH → Path C
- Criteria:
  - LOW: single fact lookup, no comparisons
  - MEDIUM: multi-hop, synthesis needed
  - HIGH: contradictions likely, requires negotiation

**Step 4.2: Token Budget Assignment**
- Based on complexity
- LOW: 500 tokens budget
- MEDIUM: 2000 tokens budget
- HIGH: 5000 tokens budget
- Track usage at each step

### Phase 5: Metrics & Evaluation

**Step 5.1: Token Tracking**
- Count input/output tokens at each node
- Calculate savings vs baseline (full document send)

**Step 5.2: Latency Tracking**
- Time per node
- Total pipeline time

**Step 5.3: Answer Quality (LLM Judge)**
- Compare against baseline answer
- Score: completeness, faithfulness, correctness

**Step 5.4: Compression Ratio**
- input_tokens / output_tokens

---

## 5. LangGraph Implementation Details

### State Schema
```python
class AEWCState(TypedDict):
    query: str
    complexity: Literal["low", "medium", "high"]
    documents: list[str]
    claims: list[dict]
    agreements: list[dict]
    contradictions: list[dict]
    accepted_claims: list[dict]
    uncertain_claims: list[dict]
    rejected_claims: list[dict]
    compressed_evidence: list[str]
    answer: str
    workflow_path: list[str]  # ["retrieve", "evidence", "contradiction", ...]
    token_budget: int
    tokens_used: dict[str, int]
    latency_ms: dict[str, float]
```

### Conditional Routing
```python
def route_by_complexity(state: AEWCState) -> str:
    if state["complexity"] == "low":
        return "simple_answer"
    elif state["complexity"] == "medium":
        return "compressed_answer"
    else:
        return "negotiated_answer"
```

---

## 6. Model Configuration

| Agent | Model | Provider | Context |
|-------|-------|----------|---------|
| Compiler | llama-3.3-70b | Groq | Fast classification |
| Evidence Extract | llama-3.3-70b | Groq | Claim extraction |
| Contradiction | llama-3.3-70b | Groq | Conflict detection |
| Negotiation | llama-3.3-70b | Groq | Evidence filtering |
| Compression | llama-3.3-70b | Groq | Synthesis |
| Answer | llama-3.3-70b | Groq | Final answer |
| Judge | llama-3.3-70b | Groq | Evaluation only |

*All agents use Groq's free tier for initial development.*

---

## 7. Retrieval Setup

- **Engine:** FAISS (simpler) or ChromaDB (persistent)
- **Default:** In-memory FAISS for MVP
- **Index:** Encode documents with sentence-transformers (all-MiniLM-L6-v2)
- **Top-K:** Configurable, default 10 chunks

---

## 8. Dependencies

```
# Core
langgraph>=0.0.20
langchain>=0.1.0
langchain-groq>=0.0.2

# Retrieval
faiss-cpu OR chromadb
sentence-transformers

# Utilities
tiktoken  # token counting
pyyaml    # config
pydantic  # validation

# Testing
pytest
pytest-asyncio
```

---

## 9. Verification Plan

For each phase, verify:

| Phase | Verification |
|-------|--------------|
| Phase 1 | Graph compiles, routes to 3 paths |
| Phase 2.1 | Documents retrieved from sample DB |
| Phase 2.2 | Claims extracted from known document |
| Phase 2.3 | Simple answer matches baseline |
| Phase 3.1 | Contradictions detected in test data |
| Phase 3.2 | Negotiation filters conflicting claims |
| Phase 3.3 | 10x compression ratio achieved |
| Phase 4 | Correct path selected per query type |
| Phase 5 | Metrics printed: tokens, latency, quality |

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| Token savings vs baseline | >70% reduction |
| Latency improvement | >50% reduction |
| Compression ratio | >5x |
| Answer quality | ≥90% of baseline quality |

---

## 11. Open Questions

1. **Vector store choice:** FAISS (simpler) vs ChromaDB (persistent/searchable)?
2. **Claim extraction granularity:** One claim per sentence or more aggressive merging?
3. **LLM Judge approach:** Use separate model or self-contained evaluation?
4. **Persistence:** Database for metrics tracking? (Not needed for MVP)
5. **Batch evaluation:** Script to evaluate on multiple queries?

---

*Plan ready for review. To proceed, exit plan mode.*