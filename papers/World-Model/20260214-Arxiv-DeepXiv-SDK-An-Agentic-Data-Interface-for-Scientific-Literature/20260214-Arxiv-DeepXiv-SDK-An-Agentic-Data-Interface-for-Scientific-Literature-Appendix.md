<a id="appendix-a"></a>

## Appendix A Implementation and Deployment

DeepXiv-SDK consists of an arXiv-scale processing pipeline that normalizes papers into section-addressable objects with enriched signals, and a serving stack that exposes progressive access and hybrid retrieval through a cached REST interface.

- [A.1 ArXiv-Scale Processing Pipeline](#a1-arxiv-scale-processing-pipeline)
- [A.2 Serving, Indexing, and T+0 Updates](#a2-serving-indexing-and-t0-updates)

### A.1 ArXiv-Scale Processing Pipeline

We obtain the full arXiv metadata stream via the OAI-PMH interface, keyed by arXiv ID. For the initial full crawl, we download PDFs for all papers and convert them to Markdown using MinerU Wang et al. ([2024a]), which normalizes heterogeneous layouts into a text-centric representation. This PDF$\rightarrow$MD stage runs on 8 nodes with 8$\times$H100 GPUs each and takes $\sim$72 hours. From Markdown, we recover section structure using formatting regularities and materialize a canonical JSON per paper, where each section is an explicit field to support deterministic section-level access.

On top of the structured JSON, we run lightweight LLM-based enrichment with Qwen3-4B-Instruct-2507. Given the first 2048 tokens and the metadata author list, the model infers author–affiliation relations, produces a one-sentence TL;DR, and extracts up to five keywords. We further extract GitHub repository links with regex and filter common non-paper-specific repositories (e.g., pytorch, vllm, sklearn); additionally, the model validates whether each extracted GitHub URL is _paper-specific_ by cross-checking the repository identity (e.g., owner/name) against the paper context. For budget-aware access, we compute total and per-section token counts using tiktoken and generate a TL;DR for each section. The enrichment stage runs on the same 8$\times$(8$\times$H100) cluster and takes $\sim$100 hours for the full corpus.

<a id="table-3"></a>

> Table 3: Example evaluation items: question-to-paper identification with arXiv IDs as gold answers.

| Question                                                                                                                                                                                                                                    | Answer (arXiv ID) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| Question                                                                                                                                                                                                                                    | Answer (arXiv ID) |
| Which paper proposes an imagination-driven agent that balances rescuing others and minimizing environmental harm in conflicting grid-world tasks, outperforming both standard and empathy-based baselines across diverse scenario variants? | 2501.00320        |
| Which paper proposes an imagination-driven agent that balances rescuing others and minimizing environmental harm in conflicting grid-world tasks, outperforming both standard and empathy-based baselines across diverse scenario variants? | 2501.00320        |
| Which paper builds a course-specific chatbot using course documents to guide a conversational model, outperforming a general model on database questions and policies, citing sources, with no instructor training?                         | 2401.00052        |
| Which paper builds a course-specific chatbot using course documents to guide a conversational model, outperforming a general model on database questions and policies, citing sources, with no instructor training?                         | 2401.00052        |
| Which paper develops a general algorithm for constructing minimal diagonally concave functions on strips with arbitrary boundary data, including cases with horizontal herringbone foliations and fissures?                                 | 2401.00053        |
| Which paper develops a general algorithm for constructing minimal diagonally concave functions on strips with arbitrary boundary data, including cases with horizontal herringbone foliations and fissures?                                 | 2401.00053        |

### A.2 Serving, Indexing, and T+0 Updates

Processed JSON and Markdown are stored in object storage, while metadata and extracted attributes are stored in PostgreSQL. We build Elasticsearch indexes over attributes and content surrogates, and compute dense embeddings with BGE-m3 on a compact representation (title + abstract + section TL;DRs) to support hybrid retrieval without indexing full text. The service is deployed on a 64-core/256GB node (with a hot-standby replica), using Gunicorn for the API, Caddy for reverse proxying and load balancing, and Redis for caching high-frequency endpoints.

For incremental maintenance, DeepXiv-SDK performs weekday scheduled syncs (Mon–Fri) by pulling OAI-PMH listing deltas and processing new/updated IDs. Since many arXiv entries provide HTML pages, we adopt an HTML-first strategy for parsing, falling back to PDF when HTML parsing fails; the downstream structuring and enrichment steps remain identical. This yields T+0 synchronization with new releases and keeps retrieval and progressive-access views fresh. Finally, the pipeline is designed to extend to other open-access corpora (e.g., PMC) by swapping only the ingestion connector, while reusing the same normalization schema, enrichment, indexing, and serving protocol.

<a id="appendix-b"></a>

## Appendix B Statistics

As of now, DeepXiv-SDK indexes 2,949,129 arXiv papers; among them, 2,712,378 are successfully parsed into an explicit section structure with per-section TL;DRs, while the remaining papers typically lack usable section cues (no section markers, weak/ambiguous structure, or content dominated by figures), yet still support the other API views and retrieval features. Across the corpus, we extract 219,717 GitHub URLs; a manual audit of 200 sampled links found no mismatches. For social attention, we query the X.com search API daily for arxiv.org mentions and aggregate exposure signals, yielding trending data for 448,825 papers to date. Finally, we enrich papers with citation counts and venue/journal metadata by linking arXiv IDs to Semantic Scholar records.

<a id="table-4"></a>

> Table 4: DeepXiv-SDK API surface. Progressive access controls cost, while hybrid retrieval curates sets before evidence-level inspection. Table [1](#LST1) showcases a header response. We detail documentation at [this page](https://data.rag.ac.cn/api/docs).

| Capability                   | Endpoint (template)                                  | Returns (role)                                                                                  |
| ---------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Capability                   | Endpoint (template)                                  | Returns (role)                                                                                  |
| Capability                   | Endpoint (template)                                  | Returns (role)                                                                                  |
| Progressive access (arXiv)   |                                                      |                                                                                                 |
| Quick Brief                  | /arxiv (type=brief, id={id})                         | Brief metadata.                                                                                 |
| Quick Brief                  | /arxiv (type=brief, id={id})                         | Brief metadata.                                                                                 |
| Quick Brief                  | /arxiv (type=brief, id={id})                         | Brief metadata.                                                                                 |
| Overview (header-first)      | /arxiv (type=head, id={id})                          | Metadata + section inventory + global budget hints.                                             |
| Overview (header-first)      | /arxiv (type=head, id={id})                          | Metadata + section inventory + global budget hints.                                             |
| Overview (header-first)      | /arxiv (type=head, id={id})                          | Metadata + section inventory + global budget hints.                                             |
| Preview                      | /arxiv (type=preview, id={id})                       | Fixed-length prefix (10k chars) + truncation flags.                                             |
| Preview                      | /arxiv (type=preview, id={id})                       | Fixed-length prefix (10k chars) + truncation flags.                                             |
| Preview                      | /arxiv (type=preview, id={id})                       | Fixed-length prefix (10k chars) + truncation flags.                                             |
| Section read                 | /arxiv (type=section, id={id}, section={sec})        | Named section payload + local budget cues.                                                      |
| Section read                 | /arxiv (type=section, id={id}, section={sec})        | Named section payload + local budget cues.                                                      |
| Section read                 | /arxiv (type=section, id={id}, section={sec})        | Named section payload + local budget cues.                                                      |
| Full text (Markdown)         | /arxiv (type=raw, id={id})                           | Full Markdown for verification.                                                                 |
| Full text (Markdown)         | /arxiv (type=raw, id={id})                           | Full Markdown for verification.                                                                 |
| Full text (Markdown)         | /arxiv (type=raw, id={id})                           | Full Markdown for verification.                                                                 |
| Full object (JSON)           | /arxiv (type=json, id={id})                          | Full structured JSON for deterministic processing.                                              |
| Full object (JSON)           | /arxiv (type=json, id={id})                          | Full structured JSON for deterministic processing.                                              |
| Full object (JSON)           | /arxiv (type=json, id={id})                          | Full structured JSON for deterministic processing.                                              |
| Retrieval and utilities      |                                                      |                                                                                                 |
| Hybrid retrieval & filtering | /arxiv (type=retrieve, q={…})                        | BM25 / vector / hybrid search with attribute filters and pagination.                            |
| Hybrid retrieval & filtering | /arxiv (type=retrieve, q={…})                        | BM25 / vector / hybrid search with attribute filters and pagination.                            |
| Hybrid retrieval & filtering | /arxiv (type=retrieve, q={…})                        | BM25 / vector / hybrid search with attribute filters and pagination.                            |
| Social attention (X)         | /arxiv/trending_signal (id={id})                     | Optional exposure stats: total_views, total_likes, total_reposts.                               |
| Social attention (X)         | /arxiv/trending_signal (id={id})                     | Optional exposure stats: total_views, total_likes, total_reposts.                               |
| Social attention (X)         | /arxiv/trending_signal (id={id})                     | Optional exposure stats: total_views, total_likes, total_reposts.                               |
| Usage & quota audit          | /stats/usage (days={n})                              | Usage summary for quota-aware tool planning.                                                    |
| Usage & quota audit          | /stats/usage (days={n})                              | Usage summary for quota-aware tool planning.                                                    |
| Usage & quota audit          | /stats/usage (days={n})                              | Usage summary for quota-aware tool planning.                                                    |
| Corpus extensibility (PMC)   |                                                      |                                                                                                 |
| PMC basic access (current)   | /pmc (type=head, id={id}), /pmc (type=json, id={id}) | Currently serves basic metainfo and full JSON; section parsing/TL;DRs are not yet materialized. |
| PMC basic access (current)   | /pmc (type=head, id={id}), /pmc (type=json, id={id}) | Currently serves basic metainfo and full JSON; section parsing/TL;DRs are not yet materialized. |
| PMC basic access (current)   | /pmc (type=head, id={id}), /pmc (type=json, id={id}) | Currently serves basic metainfo and full JSON; section parsing/TL;DRs are not yet materialized. |

> Table: Listing 1: Example metadata (JSON), accessible at [https://data.rag.ac.cn/arxiv/?arxiv_id=2409.05591&type=head](https://data.rag.ac.cn/arxiv/?arxiv_id=2409.05591&type=head). This ID is token-free; other requests require an API token.
