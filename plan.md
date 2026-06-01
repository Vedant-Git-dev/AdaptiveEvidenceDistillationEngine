# Adaptive Evidence-Driven Extraction (AEDE) with Workflow Compiler

## Plan

> **Background:** This plan merges the original AEWC concept with AEDE (Adaptive Evidence-Driven Extraction), a research architecture for reducing LLM inference costs through intelligent evidence distillation. The original plan used query *complexity* for routing; the updated plan uses *evidence coverage* for routing.

---

## 1. Project Overview

**Type:** LangGraph-based multi-agent system  
**Core Goal:** Compile the smallest reasoning workflow needed per query, adapting dynamically based on evidence quality—not query complexity.  

**Key Innovation:** Instead of classifying queries upfront, we retrieve evidence first, then decide what to do based on what we found.

**Target Users:** RAG system developers, LLM application engineers seeking cost efficiency without quality loss.

---

## 2. Architecture

```
START
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NODE 1: Focused Retriever                                               │
│  - Pure retrieval, no LLM                                                │
│  - Start small: k=4 chunks                                               │
│  - Binary growth strategy: k=4 → 8 → 16 (not k+=4)                       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NODE 2: Evidence Extractor                                              │
│  - Model: Gemma 4 (cheap)                                                 │
│  - Input: documents + query                                              │
│  - Output: [{"claim": "...", "quote": "...", "chunk_id": N}, ...]        │
│  - Goal: Convert 5000 tokens → 100 facts                                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NODE 3: Evidence Quality Analyzer  ← THE BRAIN                          │
│  - Model: Gemma 4 (cheap)                                                 │
│  - Input: facts + query                                                   │
│  - Output: {                                                             │
│  │     answered_parts: [...],    ← What's covered                       │
│  │     missing_parts: [...],     ← What's missing (actionable!)          │
│  │     coverage: 0.65,           ← 0-1 score                            │
│  │     redundancy: 0.48,         ← How much overlap                     │
│  │     confidence: 0.74          ← Quality signal                       │
│  │   }                                                                        │
└────────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NODE 4: Workflow Compiler  ← THE DECISION ENGINE                       │
│  - Pure Python, no LLM                                                    │
│  - Uses: coverage, redundancy, missing_parts, missing_parts_core          │
│  - missing_parts_core = missing_parts ∧ query_core_concepts               │
└────────────────────────────┬────────────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│  PATH A: Retrieve More   │   │  PATH B: Compress           │
│                          │   │                             │
│  Binary growth:         │   │  Input: facts               │
│  → k=4→8→16→MAX        │   │  Output: compressed_facts    │
│  Until coverage>=0.8    │   │  Target: 10x reduction       │
│  OR missing_parts_core=∅ │   │                             │
│                         │   │  Then → Final Reasoner       │
└─────────────────────────┘   └─────────────────────────────┘
          │
          └─────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NODE 7: Final Reasoner                                                   │
│  - Model: Gemini 2.5 Pro (expensive, but minimal input)                  │
│  - Input: query + compressed_evidence (300-800 tokens target)            │
│  - Output: Final answer                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                              END
```

### Why This Beats Traditional RAG

| Aspect | Traditional RAG | AEDE |
|--------|---------------|------|
| Initial retrieval | k=20 (explodes context) | k=4 (starts small) |
| Evidence processing | Raw chunks | Distilled facts |
| Context to final LLM | 5000-15000 tokens | 300-800 tokens |
| Workflow | Fixed | Adapts to evidence |
| Decision basis | Query type | Evidence quality |

---

## 3. File Structure

```
SelfTrainer/
├── src/
│   └── aede/
│       ├── __init__.py
│       ├── graph.py              # Main LangGraph StateGraph
│       ├── state.py              # Shared state schema
│       ├── config.py             # Settings & model config
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── retrieval.py      # ChromaDB retrieval (k=4 initial)
│       │   ├── extractor.py      # Evidence extraction (Gemma 4)
│       │   ├── analyzer.py        # Quality analysis (Gemma 4)
│       │   ├── compiler.py       # Workflow compiler (pure Python)
│       │   ├── compressor.py     # Evidence compression (Gemma 4)
│       │   ├── retriever_more.py  # Incremental retrieval
│       │   └── reasoner.py       # Final answer (Gemini 2.5 Pro)
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── token_tracker.py   # Token counting
│       │   ├── concepts.py        # Query core concept extraction
│       │   └── metrics.py         # Evaluation metrics
│       └── prompts/
│           ├── extractor_prompt.py
│           ├── analyzer_prompt.py
│           └── compressor_prompt.py
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_extractor.py
│   ├── test_analyzer.py
│   ├── test_compiler.py
│   └── test_integration.py
├── configs/
│   └── settings.yaml
├── data/
│   └── sample_docs/              # For testing
├── evaluation/
│   ├── benchmark_runner.py       # Run on benchmark datasets
│   ├── judge.py                  # LLM-as-judge evaluator
│   └── metrics_tracker.py        # Track coverage_history etc.
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
└── README.md
```

---

## 4. Core State Schema

```python
from typing import TypedDict, Literal

class AEDEState(TypedDict):
    # Inputs
    query: str
    query_core_concepts: list[str]  # Extracted from query analysis

    # Retrieval
    current_top_k: int
    documents: list[str]

    # Evidence pipeline
    facts: list[dict]              # [{"claim": "...", "quote": "...", "chunk_id": N}]
    compressed_evidence: list[str]

    # Quality signals (from Node 3)
    answered_parts: list[str]
    missing_parts: list[str]
    missing_parts_core: list[str]  # missing_parts ∩ query_core_concepts
    coverage: float                 # 0.0 - 1.0
    redundancy: float               # 0.0 - 1.0
    confidence: float               # 0.0 - 1.0

    # Decision tracking
    workflow_path: list[str]       # ["retrieve", "extract", "analyze", "retrieve_more", ...]
    coverage_history: list[float]  # [0.31, 0.57, 0.81, 0.90]
    token_usage: dict[str, int]     # {"gemma_input": 5000, "gemma_output": 200, ...}

    # Outputs
    answer: str
    max_retrieval_reached: bool     # Flag when k=MAX and coverage < 0.8
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure

**Step 1.1: Project setup**
- pyproject.toml with dependencies
- Virtual environment setup
- Directory structure creation

**Step 1.2: State schema & graph skeleton**
- Define `AEDEState` with all fields
- Create empty LangGraph with all nodes connected
- Verify routing works

**Step 1.3: Configuration**
- Model settings (Gemma 4 via API, Gemini 2.5 Pro via GEMINI_API_KEY)
- Retrieval settings (ChromaDB, bge-large-en-v1.5 embeddings)
- Thresholds (coverage_target=0.8, redundancy_threshold=0.4, max_k=32)

### Phase 2: Retrieval Pipeline

**Step 2.1: Focused Retriever (Node 1)**
```python
def focused_retriever(state: AEDEState) -> AEDEState:
    """Pure retrieval, no LLM. Starts with k=4."""
    return {
        "documents": retriever.similarity_search(state["query"], k=4),
        "current_top_k": 4
    }
```

**Step 2.2: Evidence Extractor (Node 2)**
- Model: Gemma 4
- Prompt extracts atomic facts with source attribution
- Output schema validation

**Step 2.3: Incremental Retriever (Node 5)**
```python
def retrieve_more(state: AEDEState) -> AEDEState:
    """Binary growth, not linear. k=4→8→16→MAX."""
    if state["current_top_k"] == 4:
        new_k = 8
    elif state["current_top_k"] == 8:
        new_k = 16
    else:
        new_k = 32  # MAX

    return {
        "documents": retriever.similarity_search(state["query"], k=new_k),
        "current_top_k": new_k
    }
```

### Phase 3: Evidence Quality Analysis

**Step 3.1: Evidence Quality Analyzer (Node 3)**
- Model: Gemma 4
- Input: facts + query + query_core_concepts
- Output: answered_parts, missing_parts, missing_parts_core, coverage, redundancy, confidence

**Step 3.2: Query Concept Extractor (utility)**
- Extract core concepts from query (e.g., "why did revenue grow" → ["revenue", "growth", "cause"])
- Use simple keyword extraction or Gemma 4 (single call, cached)

### Phase 4: Workflow Compiler

**Step 4.1: Compiler Logic (Node 4)**

This is the updated decision matrix—fixed from the original:

```python
from typing import Literal

Decision = Literal["retrieve_more", "compress", "answer", "max_retrieval_reached"]

def compiler_decision(state: AEDEState) -> Decision:
    """
    Fixed decision logic that uses ALL available signals.

    Priority:
    1. If we've hit max retrieval and still missing core concepts → answer anyway
    2. If coverage < 0.8 AND missing_parts_core is non-empty → retrieve more
    3. If redundancy > 0.4 → compress (deduplicate)
    4. If confidence < 0.5 → retrieve more (low confidence signal)
    5. Else → answer
    """
    coverage = state["coverage"]
    redundancy = state["redundancy"]
    confidence = state["confidence"]
    missing_parts_core = state.get("missing_parts_core", [])
    max_reached = state.get("max_retrieval_reached", False)

    # Case 1: Give up and answer if we've exhausted retrieval
    if max_reached and len(missing_parts_core) == 0:
        return "answer"

    # Case 2: Need more evidence for core concepts
    if coverage < 0.8 or len(missing_parts_core) > 0:
        # Check if we can still retrieve
        if state["current_top_k"] >= 32:
            return "max_retrieval_reached"
        return "retrieve_more"

    # Case 3: Too much redundancy, compress
    if redundancy > 0.4:
        return "compress"

    # Case 4: Low confidence needs more evidence
    if confidence < 0.5:
        if state["current_top_k"] >= 32:
            return "max_retrieval_reached"
        return "retrieve_more"

    # Case 5: Ready to answer
    return "answer"
```

**Why this is better than the original:**
- Uses `missing_parts_core` not just coverage
- Uses `confidence` signal
- Handles max retrieval edge case
- Prioritizes core concepts over general coverage

### Phase 5: Compression & Reasoner

**Step 5.1: Evidence Compressor (Node 6)**
- Model: Gemma 4
- Target: 10x reduction (120 facts → 12 facts)
- Merge redundant claims, preserve unique information

**Step 5.2: Final Reasoner (Node 7)**
- Model: Gemini 2.5 Pro
- Input: query + compressed_evidence
- Target output: 300-800 tokens (not 5000-15000)

### Phase 6: Evaluation

**Step 6.1: Metrics Collection**

| Metric | Tracking |
|--------|----------|
| `gemini_reduction_percent` | (1 - gemini_input_tokens / baseline_tokens) * 100 |
| `coverage_history` | List[float] tracking incremental coverage |
| `compression_ratio` | input_facts / compressed_facts |
| `workflow_path` | List[str] of nodes visited |
| `iterations_to_answer` | Count of retrieve→analyze loops |

**Step 6.2: Quality Evaluation**
- Baseline: Traditional RAG (k=20, direct to Gemini)
- AEDE: Our pipeline
- Comparison: LLM-as-judge (another Gemma/Groq call)

**Step 6.3: Benchmark Suite**
- Use RAGAS metrics: faithfulness, answer_relevancy, context_relevancy
- Custom metrics: token_savings, workflow_efficiency

---

## 6. Model Configuration

| Node | Model | Purpose | Context Limit |
|------|-------|---------|---------------|
| Compiler | (none) | Pure Python | N/A |
| Extractor | **Gemma 4** | Claim extraction | Medium |
| Analyzer | **Gemma 4** | Quality analysis | Medium |
| Compressor | **Gemma 4** | Evidence synthesis | Medium |
| Reasoner | **Gemini 2.5 Pro** | Final answer | Minimal (300-800 tokens in) |
| Judge | **Gemma 4** | Quality evaluation | Medium |

We use Gemma for:
- High-volume operations (extraction, compression, analysis)
- Where it can reliably handle structured output

We use Gemini for:
- Final answer only (where quality matters most)
- With minimal input (thanks to compression)

---

## 7. Retrieval Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Engine | ChromaDB | Persistent, good for iteration |
| Embeddings | bge-large-en-v1.5 | Strong semantic match |
| Initial k | 4 | Minimize initial context |
| Max k | 32 | Upper bound before forcing answer |
| Growth | Binary | k=4→8→16→32, not k+=4 |

---

## 8. Key Improvements Over Original Plan

| Original Plan | Updated Plan (AEDE) |
|--------------|---------------------|
| Query complexity classification upfront | Evidence-based routing (discover, then decide) |
| k=10 initial retrieval | k=4 initial retrieval |
| Complexity paths (A/B/C) | Coverage paths (retrieve/compress/answer) |
| Compiler ignored missing_parts | Compiler uses missing_parts_core |
| No confidence signal | Confidence affects routing |
| Linear growth (k+=4) | Binary growth (4→8→16→32) |

---

## 9. Verification Plan

For each phase, verify:

| Phase | Verification Criteria |
|-------|----------------------|
| Phase 1 | Graph compiles, all nodes reachable, state flows correctly |
| Phase 2.1 | k=4 retrieval returns relevant documents |
| Phase 2.2 | Facts include claim, quote, chunk_id |
| Phase 2.3 | Binary growth: k=4→8→16→32 |
| Phase 3.1 | Analyzer returns answered_parts, missing_parts with content |
| Phase 4 | Compiler routes correctly based on all signals |
| Phase 5.1 | 10x compression ratio achieved on test data |
| Phase 5.2 | Final answer < 800 tokens |
| Phase 6 | Metrics stored: coverage_history, workflow_path, token_usage |

---

## 10. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Token reduction (Gemini) | >70% | (1 - gemini_input / baseline) * 100 |
| Compression ratio | >5x | facts_count / compressed_count |
| Quality preservation | ≥90% | LLM judge score vs baseline |
| Workflow efficiency | <4 iterations | count of retrieve→analyze loops |
| Coverage convergence | ≤3 retrievals | coverage_history length |

---

## 11. Open Questions (Updated)

1. ~~FAISS vs ChromaDB~~ → **ChromaDB** (persistent, good for iteration)
2. ~~Claim granularity~~ → **Atomic claims**, let compressor merge later
3. ~~LLM Judge approach~~ → **Gemma 4 as judge** (cheaper than Gemini for evaluation)
4. **NEW: Gemma 4 reliability** → Need to validate claim extraction quality before full pipeline
5. **NEW: Confidence calibration** → What confidence score maps to reliable answers?
6. **NEW: missing_parts_core extraction** → Simple keyword overlap vs semantic match?

---

## 12. Testing Strategy

```python
# Quick validation test
def test_aede_quality_vs_baseline():
    """Compare AEDE answer quality against naive RAG."""
    query = "Why did revenue grow in Q3?"

    # Baseline: traditional RAG
    baseline_answer = traditional_rag(query)

    # AEDE: adaptive pipeline
    aede_answer = aede_pipeline(query)

    # Judge
    score = judge.evaluate(aede_answer, baseline_answer, query)
    assert score >= 0.9  # AEDE should be ≥90% of baseline
```

---

*Plan updated to incorporate AEDE architecture. Key change: evidence-based routing instead of complexity-based routing.*
