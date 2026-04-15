# Title: DeepXiv-SDK: An Agentic Data Interface for Scientific Literature

- ArXiv: 2603.00084
- Authors: Hongjin Qian, Ziyi Xia, Ze Liu, Jianlyu Chen, Kun Luo, Minghao Qin, Chaofan Li, Lei Xiong, Junwei Lan, Sen Wang, Zhengyang Liang, Yingxia Shao, Defu Lian, Zheng Liu
- Sections: 26
- Estimated tokens: 12.0k

## Contents

- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 System: DeepXiv-SDK](#3-system-deepxiv-sdk)
  - [3.1 Overview](#31-overview)
  - [3.2 Data Layer: Corpus-Scale Ingestion, Structuring, and Signal Materialization](#32-data-layer-corpus-scale-ingestion-structuring-and-signal-materialization)
    - [Processing pipeline.](#processing-pipeline)
    - [Delivered products.](#delivered-products)
  - [3.3 Service Layer: Unified Protocol for Progressive Access and Hybrid Retrieval](#33-service-layer-unified-protocol-for-progressive-access-and-hybrid-retrieval)
    - [API surface.](#api-surface)
    - [Client interfaces as service surfaces.](#client-interfaces-as-service-surfaces)
  - [3.4 Application Layer: SDK, Agent, and Deep-Research Workflows](#34-application-layer-sdk-agent-and-deep-research-workflows)
- [4 License](#4-license)
- [5 Evaluation](#5-evaluation)
  - [5.1 Agentic Search](#51-agentic-search)
    - [Setup.](#setup)
    - [Results and analysis.](#results-and-analysis)
  - [5.2 Deep Research QA](#52-deep-research-qa)
    - [Setup.](#setup)
    - [Results and analysis.](#results-and-analysis)
  - [5.3 Latency Performance](#53-latency-performance)
- [6 Conclusion](#6-conclusion)

## Abstract

LLM-agents are increasingly used to accelerate the progress of scientific research. Yet a persistent bottleneck is _data access_: agents not only lack readily available tools for retrieval, but also have to work with unstrcutured, human-centric data on the Internet, such as HTML web-pages and PDF files, leading to excessive token consumption, limit working efficiency, and brittle evidence look-up. This gap motivates the development of an agentic data interface, which is designed to enable agents to access and utilize scientific literature in a more effective, efficient, and cost-aware manner.

In this paper, we introduce DeepXiv-SDK, which offers a three-layer agentic data interface for scientific literature. 1) Data Layer, which transforms unstructured, human-centric data into normalized and structured representations in JSON format, improving data usability and enabling progressive accessibility of the data. 2) Service Layer, which presents readily available tools for data access and ad-hoc retrieval. It also enables a rich form of agent usage, including CLI, MCP, and Python SDK. 3) Application Layer, which creates a built-in agent, packaging basic tools from the service layer to support complex data access demands.

DeepXiv-SDK currently supports the complete ArXiv corpus, and is synchronized daily to incorporate new releases. It is designed to extend to all common open-access corpora, such as PubMed Central, bioRxiv, medRxiv, and chemRxiv. We release RESTful APIs, an open-source Python SDK, and a web demo showcasing deep search and deep research workflows. DeepXiv-SDK is free to use with registration.

DeepXiv-SDK: An Agentic Data Interface for Scientific Literature

Hongjin Qian, Ziyi Xia, Ze Liu, Jianlyu Chen, Kun Luo, Minghao Qin, Chaofan Li, Lei Xiong, Junwei Lan, Sen Wang, Zhengyang Liang, Yingxia Shao, Defu Lian, Zheng Liu(^†^†thanks: Corresponding author.) Beijing Academy of Artificial Intelligence {chienqhj,zhengliu1026}@gmail.com Project Page: [https://github.com/DeepXiv/deepxiv_sdk](https://github.com/DeepXiv/deepxiv_sdk)

<a id="section-1"></a>

## 1 Introduction

LLM-based agents have emerged as a practical paradigm for turning general-purpose language models into goal-directed systems that can decompose tasks, invoke tools, and refine decisions through iterative feedback (Yao et al., [2023]; Wang et al., [2024b]). Among their applications, _research agents_ are particularly promising for supporting inquiry and evidence-driven scientific decision making (Schmidgall et al., [2025]; Asai et al., [2026]). A foundational capability in such workflows is reliable access to academic papers: agents must quickly identify relevant work, navigate long and heterogeneous documents, and retrieve verifiable evidence to ground claims and synthesis (Mei et al., [2025]; Ju et al., [2025]; OpenAI, [2025]).

Despite rapid progress in agent frameworks, paper access in today’s pipelines remains largely _ad hoc_ (Ifargan et al., [2025]; Miao et al., [2025]). A common workflow queries a general-purpose search engine, opens a paper in PDF or HTML form, heuristically extracts text, and then feeds large chunks back into the agent for retrieval or question answering (Agarwal et al., [2024]). This pipeline is both inefficient and brittle: it repeatedly incurs substantial parsing and reading overhead, depends on document-specific formatting quirks, and lacks a standardized interface across venues and domains (Yang et al., [2025]). As a result, intermediate representations are hard to reuse across tasks or agents, and models must reason over noisy, unstructured text without explicit notions of structure, cost, or evidence scope (Qian and Liu, [2025]).

<a id="figure-1"></a>

![framework](images/framework.png)

> Figure 1: System overview of DeepXiv-SDK. The system ingests and enriches papers into a normalized schema with budget-aware, progressive access views (header-first triage, section-level navigation, and evidence-level verification), and serves them via a REST API backed by hybrid retrieval (lexical and dense indexes). These capabilities support agentic applications including deep search, deep research, and reproducible, evidence-grounded comparison.

An agent-friendly solution should treat paper access as a _data interface_, not a one-off parsing step (Yang et al., [2025]). First, it should be structured and normalized, exposing papers through a consistent schema so agents can access metadata, document structure, and supporting evidence via a single protocol (Tirado et al., [2016]; Lù et al., [2025]). Second, it should support progressive disclosure by knowledge density, offering coarse-to-fine views that allow agents to decide _what to read_ and _how much to read_ before paying the full-context cost, thereby reducing cognitive burden and unnecessary token expenditure (Qian and Liu, [2025]). Third, it should be retrieval-oriented and conditionable, enabling agents to locate and curate papers by composing constraints over multiple attributes and then routing to the most relevant parts once candidates are identified (Song et al., [2025]). Together, these principles make paper access reusable, budget-aware, and evidence-seeking.

Guided by these principles, we introduce DeepXiv-SDK, a unified, agent-callable interface that turns papers into _structured objects with controllable access cost_. DeepXiv-SDK materializes each paper into schema-normalized views that an agent can query deterministically: a header-first view that exposes core metadata, section inventory, and global budget cues; a section-addressable view that supports targeted reading without full-document ingestion; and an evidence-level view that returns full content for verification and downstream processing. To make discovery and curation equally tool-friendly, DeepXiv-SDK also provides hybrid, attribute-conditioned retrieval over lexical and dense indexes, allowing agents to filter and aggregate candidates by practical constraints (e.g., category signals, authors, time ranges, citation/venue attributes) before drilling into the most relevant sections. We deploy DeepXiv-SDK at arXiv scale with daily synchronization to new releases (typically within 24 hours) and release RESTful APIs, an open-source SDK, and a live demo showcasing deep-search and deep-research workflows.

Finally, we construct a small practical evaluation set to assess end-to-end usefulness under two task families that match the interface design. In _deep search_, agents retrieve and shortlist relevant papers under time constraints using retrieval plus header-first screening; in _deep research_, agents selectively read sections to extract detailed evidence, and produce evidence-linked reports or comparison tables. Across both settings, DeepXiv-SDK reduces token overhead by avoiding default full-text ingestion, improves retrieval precision via hybrid, attribute-conditioned search and structure-aware routing, and yields higher-quality outputs by escalating to evidence-level access only when verification is needed. We also stress-test the service and observe that the current stack sustains _multi-million requests per day_ with scale-out capability, supporting interactive agent use in practice.

<a id="section-2"></a>

## 2 Related Work

As LLM agents increasingly rely on tool use and multi-step interaction, _data access_ often becomes a bottleneck for scaling agent capability, since agents must repeatedly retrieve, parse, and ground on external artifacts such as web pages and PDFs (Wang et al., [2024b]; Yao et al., [2023]; OpenAI, [2025]). A growing line of work therefore argues for turning raw web artifacts into _structured, agent-callable interfaces_ with normalized schemas, controllable views, and provenance to reduce ad hoc parsing and improve reliability (Qian and Liu, [2025]; Song et al., [2025]). Within scientific literature, many prior efforts focus on agent frameworks or web platforms to improve literature review efficiency (e.g., search-then-read workflows, iterative summarization, and multi-document QA), but typically do not expose a reusable _data interface API_ that agents can call deterministically across tasks (He et al., [2025]; Miao et al., [2025]; Ju et al., [2025]). The closest efforts in spirit are ar5iv and AlphaXiv, which provide more usable browsing experiences by converting arXiv papers into readable HTML and enriched views.(^1^11[https://ar5iv.labs.arxiv.org/](https://ar5iv.labs.arxiv.org/))(^2^22[https://www.alphaxiv.org/](https://www.alphaxiv.org/)) These systems are best viewed as advanced mirrors of arXiv for human browsing, rather than agentic data interfaces designed to scale through a reusable API and progressive, budget-aware access.

In contrast, DeepXiv-SDK provides an accessible, progressive interface with explicit budget cues and hybrid, attribute-conditioned retrieval, enabling agents to screen cheaply, read selectively, and verify on demand.

<a id="section-3"></a>

## 3 System: DeepXiv-SDK

- [3.1 Overview](#31-overview)
- [3.2 Data Layer: Corpus-Scale Ingestion, Structuring, and Signal Materialization](#32-data-layer-corpus-scale-ingestion-structuring-and-signal-materialization)
- [3.3 Service Layer: Unified Protocol for Progressive Access and Hybrid Retrieval](#33-service-layer-unified-protocol-for-progressive-access-and-hybrid-retrieval)
- [3.4 Application Layer: SDK, Agent, and Deep-Research Workflows](#34-application-layer-sdk-agent-and-deep-research-workflows)

<a id="section-3-1"></a>

### 3.1 Overview

Figure [1](#figure-1) presents DeepXiv-SDK as a paper-native, agentic data interface that turns academic papers into _structured, tool-callable objects_ with controllable access cost. The system contains three layers.

The Data Layer performs corpus-scale normalization and enrichment, materializing a _section-addressable_ canonical representation with machine-consumable signals such as document structure, lightweight summaries, and explicit budget cues (e.g., token/length statistics). Building on this, the Service Layer exposes a unified protocol that provides (i) _progressive access_ via structured views that increase in information density and cost (header, section, evidence), and (ii) _hybrid, attribute-conditioned retrieval_ for constructing and refining candidate paper sets before reading. Finally, the Application Layer instantiates these primitives into end-to-end demo services, including _deep search_ for candidate discovery and screening and _deep research_ for iterative section reading and evidence-linked synthesis. This layered design keeps the interface modular and reusable; we detail each layer in the following sections.

<a id="section-3-2"></a>

### 3.2 Data Layer: Corpus-Scale Ingestion, Structuring, and Signal Materialization

The data layer (Figure [1](#figure-1), left) converts heterogeneous arXiv artifacts into _schema-stable, agent-consumable paper objects_. arXiv provides official metadata, while content is delivered as PDF/HTML with highly variable layout and section cues; directly reading these raw artifacts forces every agent pipeline to re-implement parsing and segmentation, leading to brittle failures and non-reproducible reads. DeepXiv-SDK centralizes this work and materializes a canonical representation with explicit structure, derived signals, and budget cues.

<a id="table-1"></a>

> Table 1: Data-layer deliverables in DeepXiv-SDK: enriched signal families and materialized access views.

| Signal / View     | Materialized content (examples)                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| Signal / View     | Materialized content (examples)                                                                                    |
| Core metadata     | title, authors, abstract, categories, publish/update time, affiliations, identifiers (arXiv ID, DOI if available)  |
| Core metadata     | title, authors, abstract, categories, publish/update time, affiliations, identifiers (arXiv ID, DOI if available)  |
| Structured info   | section outline, section TL;DRs, keywords, resource links                                                          |
| Structured info   | section outline, section TL;DRs, keywords, resource links                                                          |
| Budget hints      | token/length estimates (paper- and section-level), preview truncation flags (e.g., is_truncated, total characters) |
| Budget hints      | token/length estimates (paper- and section-level), preview truncation flags (e.g., is_truncated, total characters) |
| Scholarly context | citation attributes and venue signals when available (stored with provenance)                                      |
| Scholarly context | citation attributes and venue signals when available (stored with provenance)                                      |
| Social attention  | optional attention indicators aggregated from posts linking arxiv.org (e.g., views, likes, reposts)                |
| Social attention  | optional attention indicators aggregated from posts linking arxiv.org (e.g., views, likes, reposts)                |
| Overview view     | header-first payload: core metadata, derived signals, section inventory, global budget hints                       |
| Overview view     | header-first payload: core metadata, derived signals, section inventory, global budget hints                       |
| Section view      | section-addressable payloads: section text, summaries, per-section budget hints                                    |
| Section view      | section-addressable payloads: section text, summaries, per-section budget hints                                    |
| Evidence view     | verification-ready full content: full Markdown and structured JSON for deterministic downstream processing         |
| Evidence view     | verification-ready full content: full Markdown and structured JSON for deterministic downstream processing         |

- [Processing pipeline.](#processing-pipeline)
- [Delivered products.](#delivered-products)

#### Processing pipeline.

Given an arXiv ID, the data layer runs a deterministic pipeline that produces a section-addressable paper object plus derived signals. It first pulls official metadata via OAI-PMH and records publish/update timestamps for synchronization. It then acquires the source artifact, preferring the HTML-rendered view when available and falling back to the PDF otherwise. For PDFs, we convert the document to Markdown with MinerU Wang et al. ([2024a]) to normalize heterogeneous layouts into a text-centric format; for HTML, we extract the main content and normalize it into the same internal representation. Next, we recover document structure by detecting heading cues and formatting regularities, constructing an ordered section inventory (section titles and hierarchy) and segmenting normalized text into section-level payloads. These outputs are assembled into a canonical JSON with a fixed schema (paper-level fields plus an explicit section map), which supports deterministic section addressing across papers.

On top of the canonical JSON, we materialize the signals required for progressive access and retrieval. We first compute budget hints, including global and per-section token/length statistics (via tiktoken). We then generate lightweight semantic signals, including a paper preview TL;DR and per-section TL;DRs, using a small instruction model. Next, we extract resource links (e.g., GitHub repositories) via regex and use the LLM to validate whether each candidate URL is truly associated with the paper by jointly considering the URL identity and the paper preview context. We further attach optional external context when available, including citation counts and venue/journal metadata by linking arXiv IDs to third-party scholarly services such as Semantic Scholar or Google Scholar, as well as social attention signals aggregated from X.com search results for arxiv.org mentions (e.g., views, likes, reposts). Finally, we persist multiple materialized views (overview, section, and evidence forms) together with provenance fields (source type, extraction time, update time), enabling traceable and reproducible access in downstream agent pipelines.

#### Delivered products.

The data layer outputs a bundle of materialized views together with enriched signal families, so agents can consume papers deterministically without ad hoc parsing. Concretely, DeepXiv-SDK provides an overview view for screening and routing, a section view for section-addressable navigation, and an evidence view that exposes full content. These views are designed to support an explicit escalation path from low-cost triage to targeted reading and, when needed, evidence-level verification. Across all views, responses include budget hints (paper- and section-level cost cues) and provenance (source type, extraction time, update time), enabling agents to plan reading cost, track freshness, and retain traceability. Table [1](#table-1) summarizes the materialized views and the corresponding signal families.

<a id="section-3-3"></a>

### 3.3 Service Layer: Unified Protocol for Progressive Access and Hybrid Retrieval

The service layer turns data-layer artifacts into a _stable, budget-aware_ interface that agents can invoke. The goal is to make paper access (i) protocol-reusable across tasks and corpora, and (ii) cost-controllable so agents can start with low-cost screening, escalate to section-level reads, and only request full text when verification is required. DeepXiv-SDK exposes these primitives through a unified REST service with bearer-token authentication, Redis caching for high-frequency reads, on-demand loading for heavier views, and a usage endpoint that makes tool calls auditable.

- [API surface.](#api-surface)
- [Client interfaces as service surfaces.](#client-interfaces-as-service-surfaces)

#### API surface.

Table [4](#table-4) summarizes the core endpoints. The API provides two complementary capabilities. First, it supports progressive access via structured views that increase in information density and cost (overview, preview, section, full text). Second, it provides hybrid retrieval (lexical plus dense) with attribute conditioning, enabling agents to build and refine candidate paper sets before reading. Together, these endpoints allow agents to control reading budget explicitly and defer evidence-level access until necessary.

#### Client interfaces as service surfaces.

To reduce integration friction, DeepXiv-SDK provides three thin clients that bind to the same REST protocol: (i) a Python SDK with deterministic calls for retrieval and progressive reading, (ii) an MCP connector that registers endpoints as tool primitives in agent runtimes, and (iii) a CLI for scripted use and reproducible evaluation. Treating these clients as part of the service layer ensures that the protocol is operationally usable, not merely specified.

<a id="section-3-4"></a>

### 3.4 Application Layer: SDK, Agent, and Deep-Research Workflows

The application layer packages the service primitives into _developer- and agent-ready_ tooling for academic-paper deep research. The goal is twofold: to provide reproducible bindings that integrate cleanly into research-agent stacks, and to ship an agent implementation that directly instantiate progressive paper access as an executable workflow.

DeepXiv-SDK includes a lightweight Python SDK that wraps the REST protocol into a small set of deterministic calls spanning _retrieval_ and _progressive reading_ (header, section, and evidence access). On top of these tools, DeepXiv-SDK integrates a built-in agent specialized for paper-centric deep research: users can instantiate it in Python (e.g., agent = deepxiv.agent(...); agent.query("...")) or invoke it from the CLI (e.g., deepxiv agent query "..."), where the agent automatically retrieves candidates, screens them via low-cost views, routes to relevant sections, and escalates to evidence-level reads only when verification is needed.

To make the intended behavior concrete, we expose two canonical workflows. _Deep search_ emphasizes candidate set construction, filtering, and ranking via hybrid retrieval and header-level signals (optionally incorporating social attention indicators). _Deep research_ performs iterative section reading to extract experimental settings and results, and produces evidence-linked summaries or comparison tables across a paper set. Finally, we serve a live demo at [this website](https://1stauthor.com/) that showcases these workflows end-to-end in an interactive setting.

<a id="section-4"></a>

## 4 License

DeepXiv-SDK is designed as an _agentic access and structuring interface_ rather than a full-text redistribution service. We reuse arXiv _descriptive metadata_ (e.g., title, authors, abstract, categories, and timestamps) under the arXiv API Terms of Use, which state that arXiv e-print metadata are available under CC0 1.0.(^3^33[https://info.arxiv.org/help/api/tou.html](https://info.arxiv.org/help/api/tou.html)) Beyond metadata, DeepXiv-SDK derives navigation and enrichment signals (e.g., section structure, summaries, budget cues, and optional scholarly/social context) to support agentic screening, routing, and evidence-seeking.

For paper content, we do not claim redistribution rights and do not mirror or bulk-serve full papers. Most arXiv submissions use the default arXiv license, which grants arXiv the right to distribute the work but does not generally grant third parties an unrestricted right to redistribute full text; arXiv further recommends that tools built on full text link back to arXiv for downloads(^4^44[https://info.arxiv.org/help/bulk_data_s3.html](https://info.arxiv.org/help/bulk_data_s3.html)). Accordingly, for any request that requires accessing the original paper, DeepXiv-SDK returns the source link (the arXiv landing page or the publisher-provided URL) rather than serving a mirrored copy. This stance is consistent with existing arXiv-derived browsing interfaces such as ar5iv(^5^55[https://ar5iv.labs.arxiv.org/](https://ar5iv.labs.arxiv.org/)) and AlphaXiv(^6^66[https://www.alphaxiv.org/](https://www.alphaxiv.org/)), which improve usability while linking back to the source.

<a id="section-5"></a>

## 5 Evaluation

We evaluate DeepXiv-SDK on two tasks that match our interface design: (i) _agentic paper search_ under multi-constraint queries, and (ii) _deep research_ for complex, evidence-backed QA. We also benchmark service latency and caching. Figure [2](#figure-2) summarizes results, and Table [3](#table-3) gives representative queries.

- [5.1 Agentic Search](#51-agentic-search)
- [5.2 Deep Research QA](#52-deep-research-qa)
- [5.3 Latency Performance](#53-latency-performance)

<a id="section-5-1"></a>

### 5.1 Agentic Search

- [Setup.](#setup)
- [Results and analysis.](#results-and-analysis)

#### Setup.

We construct a search-focused evaluation set of 50 multi-condition queries, each mapped to a _single_ target paper (unique answer). We compare DeepXiv against five academic search platforms that expose agentic search capabilities: Google Scholar, Google Scholar Labs (^7^77[https://scholar.google.com/scholar_labs/](https://scholar.google.com/scholar_labs/)), alpXiv (^8^88[https://www.alphaxiv.org/](https://www.alphaxiv.org/)), PASA (^9^99[https://pasa-agent.ai/](https://pasa-agent.ai/)), and ASTA (^10^1010[https://asta.allen.ai/](https://asta.allen.ai/)). We report retrieval metrics and measure end-to-end latency for plain search versus agentic deep search (Figure [2](#figure-2) a).

<a id="figure-2"></a>

![framework](images/framework.png)

> Figure 2: Evaluation of DeepXiv-SDK. (a) Agentic paper search on 50 multi-constraint queries with unique targets: DeepXiv achieves higher Recall@1/10 with substantially lower latency than existing agentic search platforms. (b) Deep research QA on 47 queries: DeepXiv reduces token and time cost while improving answer quality compared to a traditional Search&Read pipeline.

#### Results and analysis.

DeepXiv consistently outperforms the baselines in both accuracy and efficiency. Across the multi-constraint queries, DeepXiv yields stronger top-rank retrieval and maintains high recall while remaining near-interactive for standard search and substantially faster for agentic search than verification-heavy systems. Qualitatively, PASA and alpXiv often enumerate candidates and validate by reading full paper text, whereas DeepXiv leverages _progressive access_: agents first screen with low-cost header/preview signals and only escalate to section- or evidence-level reads when necessary, which improves precision while reducing latency.

<a id="section-5-2"></a>

### 5.2 Deep Research QA

- [Setup.](#setup)
- [Results and analysis.](#results-and-analysis)

#### Setup.

We build a second evaluation set of 47 complex QA queries that require aggregating and verifying paper-derived evidence (e.g., time-scoped questions over benchmark results such as “best score in the last month”). We compare two agent pipelines: (i) a traditional _Search & Read_ (S&R) pipeline using Google Search for retrieval and Jina Reader (^11^1111[https://jina.ai/](https://jina.ai/)) for reading full text, and (ii) a DeepXiv-based pipeline that uses attribute-conditioned retrieval plus progressive access (header, section, evidence) for targeted reading.

#### Results and analysis.

Figure [2](#figure-2)b shows that DeepXiv-based deep research reduces both tool cost and wall-clock latency by avoiding default full-text ingestion and replacing brittle page parsing with structured, section-addressable access. Across representative models, DeepXiv improves end-to-end answer quality consistently while substantially reducing token consumption and runtime. This pattern aligns with the intended interaction model: the agent screens candidates using low-cost views, selectively reads relevant sections, and escalates to evidence-level access only when verification is needed. Overall, the results support the central claim of DeepXiv-SDK: making papers _tool-callable_ with progressive, budget-aware access improves both efficiency and grounding quality for research-agent workflows.

<a id="section-5-3"></a>

### 5.3 Latency Performance

We benchmark DeepXiv-SDK latency on 1,000 arXiv IDs with concurrency $=16$, reporting mean cold (cache-miss) and warm (cache-hit) times for Local (same-region) and Remote (cross-country) calls (Table [2](#table-2)). The service remains interactive under this load, and caching provides reliable acceleration for frequently accessed views (up to $3.36\times$ on preview); even remotely, warm latencies stay within a few hundred milliseconds for heavier endpoints such as json (181.6 ms).

For context, a conventional “fetch+parse” workflow on the same 1,000 papers takes 31m12s to fetch HTML/PDF via a commercial proxy pool and 88m49s to convert to Markdown (with 8$\times$A100 GPUs), totaling 120m01s, or 7.20s per paper. Compared to this baseline, DeepXiv-SDK delivers large end-to-end speedups, e.g., $54.6\times$ (Local) and $39.6\times$ (Remote) for warm json access.

<a id="table-2"></a>

> Table 2: Mean latency over 1,000 arXiv IDs at concurrency $=16$ (Local vs Remote; cold vs warm). Cache spd is cold/warm. In comparison, fetch+parse baseline: 120m01s for 1,000 papers (7,200 ms per paper).

|         | Local (ms) | Remote (ms) | Cache spd |        |             |
| ------- | ---------- | ----------- | --------- | ------ | ----------- |
|         | Local (ms) | Remote (ms) | Cache spd |        |             |
|         | Local (ms) | Remote (ms) | Cache spd |        |             |
|         | Local (ms) | Remote (ms) | Cache spd |        |             |
|         | Cold       | Warm        | Cold      | Warm   | (L / R)     |
|         | Cold       | Warm        | Cold      | Warm   | (L / R)     |
|         | Cold       | Warm        | Cold      | Warm   | (L / R)     |
|         | Cold       | Warm        | Cold      | Warm   | (L / R)     |
|         | Cold       | Warm        | Cold      | Warm   | (L / R)     |
|         | Cold       | Warm        | Cold      | Warm   | (L / R)     |
| head    | 23.21      | 12.51       | 142.55    | 114.62 | 1.86 / 1.24 |
| head    | 23.21      | 12.51       | 142.55    | 114.62 | 1.86 / 1.24 |
| head    | 23.21      | 12.51       | 142.55    | 114.62 | 1.86 / 1.24 |
| head    | 23.21      | 12.51       | 142.55    | 114.62 | 1.86 / 1.24 |
| head    | 23.21      | 12.51       | 142.55    | 114.62 | 1.86 / 1.24 |
| head    | 23.21      | 12.51       | 142.55    | 114.62 | 1.86 / 1.24 |
| brief   | 187.55     | 164.46      | 307.39    | 196.25 | 1.14 / 1.57 |
| brief   | 187.55     | 164.46      | 307.39    | 196.25 | 1.14 / 1.57 |
| brief   | 187.55     | 164.46      | 307.39    | 196.25 | 1.14 / 1.57 |
| brief   | 187.55     | 164.46      | 307.39    | 196.25 | 1.14 / 1.57 |
| brief   | 187.55     | 164.46      | 307.39    | 196.25 | 1.14 / 1.57 |
| brief   | 187.55     | 164.46      | 307.39    | 196.25 | 1.14 / 1.57 |
| raw     | 108.14     | 146.15      | 216.29    | 167.89 | 0.74 / 1.29 |
| raw     | 108.14     | 146.15      | 216.29    | 167.89 | 0.74 / 1.29 |
| raw     | 108.14     | 146.15      | 216.29    | 167.89 | 0.74 / 1.29 |
| raw     | 108.14     | 146.15      | 216.29    | 167.89 | 0.74 / 1.29 |
| raw     | 108.14     | 146.15      | 216.29    | 167.89 | 0.74 / 1.29 |
| raw     | 108.14     | 146.15      | 216.29    | 167.89 | 0.74 / 1.29 |
| json    | 203.79     | 131.89      | 283.21    | 181.63 | 1.55 / 1.56 |
| json    | 203.79     | 131.89      | 283.21    | 181.63 | 1.55 / 1.56 |
| json    | 203.79     | 131.89      | 283.21    | 181.63 | 1.55 / 1.56 |
| json    | 203.79     | 131.89      | 283.21    | 181.63 | 1.55 / 1.56 |
| json    | 203.79     | 131.89      | 283.21    | 181.63 | 1.55 / 1.56 |
| json    | 203.79     | 131.89      | 283.21    | 181.63 | 1.55 / 1.56 |
| preview | 102.05     | 30.39       | 201.46    | 112.31 | 3.36 / 1.79 |
| preview | 102.05     | 30.39       | 201.46    | 112.31 | 3.36 / 1.79 |
| preview | 102.05     | 30.39       | 201.46    | 112.31 | 3.36 / 1.79 |
| preview | 102.05     | 30.39       | 201.46    | 112.31 | 3.36 / 1.79 |
| preview | 102.05     | 30.39       | 201.46    | 112.31 | 3.36 / 1.79 |
| preview | 102.05     | 30.39       | 201.46    | 112.31 | 3.36 / 1.79 |

<a id="section-6"></a>

## 6 Conclusion

In this paper, we focus on _paper access_ as a practical bottleneck for research agents: existing pipelines often treat papers as raw PDF/HTML artifacts, leading to expensive full-text ingestion and brittle evidence lookup. We propose DeepXiv-SDK, an agentic data interface that represents papers as normalized, tool-callable objects with progressive access (header $\rightarrow$ section $\rightarrow$ evidence) and hybrid, attribute-conditioned retrieval. Through an arXiv-scale deployment with daily synchronization, we operationalize this design as a stable service and SDK that agents can invoke repeatedly across tasks. Empirically, via task-driven evaluation on multi-constraint agentic search and complex deep-research QA, as well as latency benchmarking under concurrent load, we show that DeepXiv-SDK enables cheaper screening, more selective section reading, and verification only when needed, improving both efficiency and evidence grounding in AI4Science workflows.

Implementation details of DeepXiv-SDK (data processing and service deployment) are provided in Appendix [A](#appendix-a), current corpus statistics are summarized in Appendix [B](#appendix-b), and Listing [1](#LST1) showcases the metadata format returned for an example paper.
