# Notebook E results

The current run selects **47 unique URLs** for scraping, linked to **12 of the 21 search items**. Nine items have no sufficiently supported match in the current metadata. No candidate page was fetched: the output is a scrape queue, not a set of verified answers.

## Start here

- `candidate_scraping_list.csv`: the final queue, ordered by scrape priority. Includes English destination text, matched item IDs and themes, scores, reasons, source examples, and `scrape_status=not_started`.
- `scraping_urls.txt`: the same unique URLs, one per line.
- `question_coverage.csv`: candidate coverage and explicit gaps for all 21 items.
- `ranking_coverage.png`: selected matches by theme and unique URLs by domain.

## How the list was made

English destination titles and link labels receive most of the lexical weight; website sections receive less. Source-page topics, Dutch URL words and discovery frequency do not contribute to relevance scores.

The pipeline combines controlled phrase matching, query expansion, TF-IDF with word/bigram cosine similarity, sentence embeddings, reciprocal rank fusion, and cross-encoder reranking. Clustering describes the candidate set; it does not decide relevance. The shortlist prefers information/service pages, limits repeated domains per item, and deprioritises near-identical metadata. Explicit assistant metadata reviews remove misleading matches and recover a few useful directories.

The nine gaps are item IDs **1, 3, 4, 5, 6, 10, 11, 12 and 21**: referral process, changing doctors, all waiting-time questions, all second-opinion questions, and English-speaking providers. Relevant content may exist within pages whose link metadata does not reveal it. Selected pages may also cover only part of a question.

## Inspect the decisions

- `candidate_url_audit.csv`: every original input URL, filtering status/reason, and final selection status. A blank score means it was not in the retrieved pair pool, not that it was assigned a relevance score of zero.
- `question_url_scores.csv`: the retrieved question–URL pool, component scores, matching terms, ranks, review notes, and inclusion/defer reasons.
- `ranking_review_sample.csv`: top matches, selected pairs and sampled rejected candidates. Human-label fields are intentionally blank.
- `method_comparison.csv` and `method_check_details.csv`: results on 11 assistant-authored metadata development contrasts. This is not an independent benchmark and does not establish real-world precision or recall.
- `selection_sensitivity.csv`: shortlist sizes under nearby semantic cutoffs.
- `candidate_topic_clusters.csv`: exploratory clusters, English terms and selection counts.
- `ranking_manifest.json`: input hashes, model revisions, dependency versions and settings.

## Reproduce

Run `new/notebooks/E_candidate_url_filtering_and_scoring.ipynb` from its own directory after installing `requirements.txt`. On a fresh machine, run `new/scripts/download_ranking_models.py` first to download the pinned public models. The notebook itself runs offline; it never visits the candidate URLs. Model files and calculated embeddings are cached under the ignored `new/.cache/ranking` directory.

The project's existing Python runs as Intel/x86 on macOS. NumPy 1.26.4, PyTorch 2.2.2 and compatible Transformers versions are pinned for that environment; `pip check` passed. The source questions and notebooks A–D were not changed.

The thresholds are practical starting settings, not calibrated probabilities or a proven optimum. The English metadata does not establish the language, current availability, authority, or actual contents of the destination page. All selected rows record that page content and page language have not been checked.
