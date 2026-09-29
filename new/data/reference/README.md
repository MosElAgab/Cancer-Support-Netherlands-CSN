# Search references for notebook E

These small, editable files explain the choices behind URL ranking. They are separate from the source questions extracted in notebook D.

- `community_query_profiles.csv`: assistant-authored English expansions of each search item. The original questions stay in the interim dataset.
- `community_search_vocabulary.csv`: English phrases, synonyms and variants. Specific phrases have weight 3; supporting words have weight 1. These are search aids, not additional source facts.
- `ranking_models.json`: exact revisions of the two public local models. `new/scripts/download_ranking_models.py` downloads only those model files.
- `selection_policy.json`: initial evidence thresholds, shortlist limits and diversity settings. These settings are heuristic, not calibrated probabilities or a proven optimum.
- `metadata_review_decisions.csv`: assistant decisions based on the existing English destination metadata. Exclusions remove misleading matches. Explicit inclusions recover relevant directories or decision-support resources missed by automatic thresholds. Page content was not inspected.
- `metadata_retrieval_checks.csv`: a small assistant-authored development set contrasting useful topic metadata with misleading alternatives. It is not independent human-labelled ground truth. Use the exported review sample for a later human assessment.

References tied to `item_id` must be reviewed if notebook D's questions change. Review decisions are tied to an item and an exact URL; they do not automatically apply to newly discovered pages.
