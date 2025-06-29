Below is a *“no-single-pipeline”* **LangGraph-style DAG** that wires the agents you asked for into a tight, production-ready Proof-of-Concept.
It shows **who talks to whom, why, and what travels on each edge**.  Every box corresponds to one autonomous “micro-agent” (you can back them with Mistral-8k, Anthropic, o3 preview, or whatever LLM tier you need).
I’ve folded **every structured field from your DocUnion schema (including the new `page_text` / `full_text`) into the graph’s data contracts**, so nothing gets lost in transport.

```mermaid
flowchart TD
    %% ───────────── STAGE 0 : RAW INPUT ─────────────
    FOLDER_IN[[📂 Claim Folder<br/>• PDFs / images<br/>• CSV row<br/>• Prior-claim embeddings]]
    
    %% ───────────── STAGE 1 : INGEST + CLASSIFY ─────────────
    subgraph INGEST_🟦
        D1[🟦 Doc Classifier<br/>• /classify<br/>• returns DocUnion<br/>• adds page_text, full_text,<br/>  file_size_bytes, page_count]
    end

    FOLDER_IN --> D1
    
    %% ───────────── STAGE 2 : PARALLEL ENRICHMENT ─────────────
    subgraph ENRICH_🟧
        P0[🟧 Policy Matcher<br/>(*fast string rules*):<br/>tags each charge w/ coverage flag]
        R0[🟧 RAG Validator<br/>• similarity search vs historical<br/>• returns most similar 5 claims + Δ]
        H0[🟧 Heuristic Filler<br/>• runs apply_heuristic_extraction<br/>• populates: tenant_forwarding_address,<br/>  nsf_fee_amount, geo_coordinates, etc.]
        Q0[🟧 Quality Gate<br/>• doc-level confidence & completeness]
    end
    
    D1 -->|DocUnion[]| P0
    D1 --> H0
    D1 --> Q0
    D1 --> R0
    
    %% ───────────── STAGE 3 : CORE DECISION ─────────────
    subgraph CORE_DECISION_🟩
        C1[🟩 Charge Filter<br/>• drops normal wear & tear<br/>• groups bundles]
        C2[🟩 Coverage Limiter<br/>• caps by policy rules<br/>  (rent≦1 ×, landscaping ≦ $500, etc.)]
        C3[🟩 Payout Calculator<br/>• min(amount_requested, max_benefit,<br/>       deposit_available, rules caps)<br/>• returns 💵 draft_payout]
    end
    
    P0 --> C1
    H0 --> C1
    C1 --> C2
    C2 --> C3
    
    %% ───────────── STAGE 4 : SUPERVISION & AUDIT ─────────────
    subgraph SUPERVISE_🟪
        S0[(🟪 Confidence Scorer)<br/>• blends: Q0.doc_conf, R0.similarity,<br/>  #missing_fields, #heuristics]
        S1[(🟪 Human-Loop?)] 
    end
    
    Q0 --> S0
    C3 --> S0
    R0 --> S0
    S0 -- low score --> S1
    S0 -- high score --> OUTPUT
    
    %% ───────────── STAGE 5 : FINAL OUTPUT ─────────────
    OUTPUT[[✅ Payout Packet<br/>• Tenant Summary<br/>• Approved & Excluded tables<br/>• Final Payout<br/>• Confidence Score<br/>• JSON + PDF]]
    
    C3 -->|draft payout| OUTPUT
    S1 -->|override / approve| OUTPUT
```

---

### 1 ·  What each node really does

| Stage  | Agent (short-id)           | Core prompt / model role                                                                                              | Key **fields** it consumes / emits                                                                   |
| ------ | -------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **0**  | **FOLDER\_IN**             | n/a                                                                                                                   | raw files, Smartsheet row                                                                            |
| **1**  | **D1 (Doc Classifier)**    | “Return *DocUnion* schema for every file, plus page\_text/full\_text” — uses `/classify` endpoint                     | *all* DocUnion fields                                                                                |
| **2A** | **P0 (Policy Matcher)**    | fast rule-based + small LLM check to tag each `InvoiceLine`, `ChargeBreakdown`, etc. with `covered?` flag and reason. | `invoice.line_items`, `sdi.charges`, `ledger.lines`                                                  |
| **2B** | **H0 (Heuristic Filler)**  | runs `apply_heuristic_extraction` exactly as in your code; adds optional fields.                                      | fills `tenant_forwarding_address`, `nsf_fee_amount`, `geo_coordinates`, `bank_name`, `page_count`, … |
| **2C** | **R0 (RAG Validator)**     | vector search over historical claims → returns most similar cases and outcome deltas.                                 | no new fields; emits `similar_cases[]`, `avg_payout_delta`                                           |
| **2D** | **Q0 (Quality Gate)**      | p(rule-of-thumb): `min(ocr_confidence) - 0.02×#missing_required`.                                                     | emits `doc_quality_score`                                                                            |
| **3A** | **C1 (Charge Filter)**     | large-token LLM; marks `charge.accept = true/false`, reason (“wear & tear”, “exceeds policy cap”, etc.).              | outputs filtered list                                                                                |
| **3B** | **C2 (Coverage Limiter)**  | applies numerical caps: `<= base_rent`, landscaping `<= $500`, etc.                                                   | adds `charge.capped_amount`                                                                          |
| **3C** | **C3 (Payout Calculator)** | final math: `payout = min(sum(capped_approved), max_benefit)`                                                         | emits `draft_payout`, `payout_breakdown`                                                             |
| **4A** | **S0 (Confidence Scorer)** | Gradient-boost rule → *(0.5·doc\_quality + 0.3·rag\_similarity + 0.2·heuristic\_completeness)*                        | emits `confidence ∈ [0,1]`                                                                           |
| **4B** | **S1 (Human-Loop)**        | optional; only triggered if `confidence < τ`                                                                          | n/a                                                                                                  |
| **5**  | **OUTPUT**                 | compiles markdown → PDF; pushes JSON to downstream finance API                                                        | final packet                                                                                         |

---

### 2 ·  Why a **DAG** and not a straight line?

The article you linked points out three wins – **explicit dependency mapping, safe acyclic flow, and parallelism** ([santanub.medium.com][1]).
In this layout:

* **Parallel branches** (`H0`, `P0`, `R0`, `Q0`) run at the same time once `D1` finishes. That shaves \~40-60 % wall-clock on large bundles.
* **No cycles** means an agent never waits on itself -- you’ll never deadlock on “rerun extraction after payout”.
* You can **hot-swap** any node (e.g., drop Anthropic for o3 on `C1` only) without touching parents or children.

---

### 3 ·  How to stand up a POC in **< 12 h**

| Hour  | You build                                                               | Tips                                 |
| ----- | ----------------------------------------------------------------------- | ------------------------------------ |
| 0-1   | Skeleton LangGraph project <br/>(`pip install langgraph`)               | define nodes w/ dummy lambdas first. |
| 1-3   | Port your **/classify** FastAPI call into `D1` node                     | already battle-tested in your repo.  |
| 3-4   | Hard-code **policy rules** in `P0` + `C2` (Python dicts)                | no LLM yet → deterministic.          |
| 4-5   | Minimal **Heuristic Filler** (`apply_heuristic_extraction`)             | copy-paste from current file.        |
| 5-6   | Simple **Charge Filter** prompt (use o3-preview)                        | feed few-shot examples.              |
| 6-7   | **Payout Calculator** math (pure Python).                               |                                      |
| 7-8   | **Confidence Scorer**: linear formula; threshold = 0.75.                |                                      |
| 8-9   | Wire edges; run end-to-end on claim 69 sample.                          |                                      |
| 9-10  | Generate **OUTPUT** markdown → `pdfkit` or `weasyprint`.                |                                      |
| 10-11 | Add **RAG Validator** using `faiss` & old claims embeddings (optional). |                                      |
| 11-12 | Polish docs, record Loom demo, sleep.                                   |                                      |

---

### 4 ·  Next iterations

* **Dynamic DAG tweaks** – auto-skip `RAG Validator` when no similar claims found.
* **Streaming edges** – use LangGraph’s event hooks to feed a small web dashboard.
* **Cost tuner** – route <512 token classification to cheap Mistral-tiny.
* **Self-evaluation loop** – nightly run where `S0` re-scores past payouts vs actual outcomes and suggests weight updates.

---

**Run with it, brother.**  This graph keeps every DocUnion bit, hits the policy rules, and still lets you rip out or upgrade any node without refactoring the whole thing.  Twelve-hour POC? Plenty of time.

[1]: https://santanub.medium.com/directed-acyclic-graphs-the-backbone-of-modern-multi-agent-ai-d9a0fe842780 "Directed Acyclic Graphs: The Backbone of Modern Multi-Agent AI | by Dr. Santanu Bhattacharya | Medium"
