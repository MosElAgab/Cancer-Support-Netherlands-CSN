"""Local, English-metadata retrieval for notebook E. Never fetches candidate pages."""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, unquote
import hashlib
import html
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(value):
    if pd.isna(value):
        return ""
    value = html.unescape(str(value)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", value)).strip()


def unique_text(values):
    return ". ".join(dict.fromkeys(text for value in values if (text := clean_text(value))))


def canonical_url(url):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return url
    # Remove only recognised tracking parameters; preserve functional query parameters.
    params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
              if not k.casefold().startswith('utm_') and k.casefold() not in {'gclid', 'fbclid'}]
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path or '/',
                       urlencode(params), ''))


def prepare_candidates(urls, enriched):
    data = urls.drop(columns=['url_text'], errors='ignore').copy()
    labels = enriched.groupby('full_url')['link_text_en'].agg(
        lambda values: list(dict.fromkeys(v for v in values.dropna().astype(str) if v.strip())))
    generic = {'read more', 'more', 'click here', 'read', 'image', 'link', 'here', 'download'}
    def useful_labels(url):
        values = labels.get(url, [])
        return [v for v in values if clean_text(v) not in generic]
    data['link_labels_en'] = data['full_url'].map(useful_labels)
    data['title_text'] = data['full_url_slug_en'].map(clean_text)
    data['anchor_text'] = data['link_labels_en'].map(unique_text)
    data['section_text'] = data['url_section_en'].map(clean_text)
    # Title first; all unique English link labels are retained, with section separate below.
    data['destination_text'] = data.apply(
        lambda row: unique_text([row['title_text'], *row['link_labels_en'], row['section_text']]), axis=1)
    data['canonical_url'] = data['full_url'].map(canonical_url)
    parts = data['full_url'].map(urlsplit)
    data['url_scheme'] = parts.map(lambda p: p.scheme.casefold())
    data['destination_domain'] = parts.map(lambda p: p.hostname or '')
    data['is_pdf_url'] = parts.map(lambda p: unquote(p.path).casefold().endswith('.pdf'))
    data['english_field_warning'] = data['destination_text'].map(
        lambda t: 'possible untranslated fragment' if re.search(
            r'\b(zorgverzekering|ziekenhuis|verzekerde|kankernl|verwijsgids|huisarts|vergoeding)\b', t)
        else '')
    exclude = {
        'Privacy/cookie page': {'privacy', 'privacy policy', 'privacy statement', 'privacy and cookies',
                               'cookies', 'cookie policy'},
        'Login page': {'login', 'log in', 'sign in', 'signin'},
        'Share/navigation endpoint': {'share via email', 'share on facebook', 'share on twitter',
                                      'rss', 'print', 'sitemap'},
    }
    review = {
        'Job/volunteer listing': {'jobs', 'careers', 'vacancies', 'vacancies and calls'},
        'Disclaimer': {'disclaimer'},
        'Login help': {'login help'},
    }
    def decision(row):
        if row['url_scheme'] not in {'http', 'https'} or not row['destination_domain']:
            return 'exclude', 'Not a web page link', 'scheme: '+row['url_scheme']
        for status, rules in [('exclude', exclude), ('review', review)]:
            for reason, phrases in rules.items():
                # Exact English page labels only; broad sections never cause exclusion.
                if row['title_text'] in phrases:
                    return status, reason, 'full_url_slug_en: '+row['title_text']
        if re.search(r'\blogin required\b', row['anchor_text']):
            return 'review', 'Link label indicates login required', 'English link labels'
        if not re.search(r'[a-z]{3}', row['title_text']+' '+row['anchor_text']):
            return 'review', 'No useful destination label', 'title and link labels'
        return 'keep', 'Available for relevance scoring', ''
    decisions = data.apply(decision, axis=1, result_type='expand')
    decisions.columns = ['filter_status', 'filter_reason', 'filter_evidence']
    data[decisions.columns] = decisions
    def kind(row):
        text = row['section_text']
        if row['is_pdf_url']: return 'pdf'
        if text in {"others' experiences", 'blogs', 'discussion groups'}: return 'peer_experience'
        if text in {'tidings','news','topical'}: return 'news'
        if text in {'nkr','about nkr','scientific publications','research','work agenda'}: return 'research_or_registry'
        return 'information_or_service'
    data['content_kind'] = data.apply(kind, axis=1)
    return data


def phrase_scores(queries, candidates, vocabulary):
    scores = np.zeros((len(queries), len(candidates)), dtype=np.float32)
    specific = np.zeros_like(scores, dtype=np.int32)
    evidence = {}
    texts = (candidates['title_text']+'. '+candidates['anchor_text']).tolist()
    for qi, item in enumerate(queries.itertuples()):
        terms = vocabulary.loc[vocabulary.item_id.eq(item.item_id)]
        patterns = [(row.phrase, row.weight, row.specificity,
                     re.compile(r'(?<!\w)'+re.escape(clean_text(row.phrase))+r'(?:s)?(?!\w)'))
                    for row in terms.itertuples()]
        for di, text in enumerate(texts):
            matches = [(p, w, s) for p, w, s, regex in patterns if regex.search(text)]
            if matches:
                scores[qi, di] = sum(w for _, w, _ in matches)
                specific[qi, di] = sum(s == 'specific' for _, _, s in matches)
                evidence[(qi, di)] = '; '.join(p for p, _, _ in matches)
    # Cap repeated related phrases so long metadata cannot dominate indefinitely.
    return np.minimum(scores/9.0, 1.0), specific, evidence


def lexical_scores(queries, candidates, field_weights):
    matrices = {}
    explain = {}
    total = np.zeros((len(queries), len(candidates)))
    # Fit IDF on candidate texts only. Queries are transformed using the same vocabulary.
    for field, weight in field_weights.items():
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True,
                                    strip_accents='unicode', token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b')
        documents = vectorizer.fit_transform(candidates[field].fillna(''))
        query_vectors = vectorizer.transform(queries['retrieval_query'])
        values = cosine_similarity(query_vectors, documents)
        matrices[field] = values
        explain[field] = (vectorizer, query_vectors, documents)
        total += weight*values
    return total, matrices, explain


def overlap_terms(qi, di, explain, limit=8):
    contributions = {}
    for field, (vectorizer, query_vectors, documents) in explain.items():
        products = query_vectors[qi].multiply(documents[di]).tocoo()
        names = vectorizer.get_feature_names_out()
        for index, value in zip(products.col, products.data):
            term = names[index]
            contributions[term] = contributions.get(term, 0)+float(value)
    return '; '.join(sorted(contributions, key=contributions.get, reverse=True)[:limit])


def cache_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def embedding_scores(queries, candidates, model_spec, cache_dir):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    import torch
    torch.set_num_threads(4)
    cache_dir = Path(cache_dir)
    key = cache_key([model_spec, candidates.destination_text.tolist(), queries.retrieval_query.tolist()])
    cache = cache_dir/f'embeddings-{key}.npz'
    if cache.exists():
        saved = np.load(cache)
        docs, query, truncated = saved['documents'], saved['queries'], saved['truncated']
    else:
        path = snapshot_download(model_spec['model_id'], revision=model_spec['revision'],
                                 cache_dir=cache_dir/'models', local_files_only=True)
        model = SentenceTransformer(path, device='cpu')
        model.max_seq_length = 256
        texts = candidates.destination_text.tolist()
        truncated = np.array([len(model.tokenizer.encode(t, truncation=False)) > 256 for t in texts])
        docs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        query = model.encode(queries.retrieval_query.tolist(), normalize_embeddings=True,
                             show_progress_bar=False)
        np.savez_compressed(cache, documents=docs, queries=query, truncated=truncated)
    return query@docs.T, docs, truncated


def ranked_positions(scores):
    # Stable tie breaking follows canonical URL order established in the notebook.
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.empty_like(order)
    for qi in range(len(scores)):
        ranks[qi, order[qi]] = np.arange(1, scores.shape[1]+1)
    return ranks


def retrieve_pairs(queries, candidates, lexical, semantic, phrase, specific, evidence,
                   explain, retrieval_depth=40, rrf_k=60):
    ranks = {name: ranked_positions(scores) for name, scores in
             [('tfidf', lexical), ('semantic', semantic), ('phrase', phrase)]}
    # Reciprocal rank fusion combines ranks, without treating unlike raw scores as equivalent.
    fused = (1/(rrf_k+ranks['tfidf'])*(lexical > 0)
             + 1/(rrf_k+ranks['semantic'])*(semantic > 0)
             + 0.5/(rrf_k+ranks['phrase'])*(phrase > 0))
    rows = []
    for qi, item in enumerate(queries.itertuples()):
        pool = ((ranks['tfidf'][qi] <= retrieval_depth) & (lexical[qi] > 0)
                | (ranks['semantic'][qi] <= retrieval_depth) & (semantic[qi] > 0)
                | (ranks['phrase'][qi] <= retrieval_depth) & (phrase[qi] > 0))
        for di in np.flatnonzero(pool):
            row = candidates.iloc[di]
            rows.append(dict(item_id=item.item_id, item_text=item.item_text,
                section_heading=item.section_heading, retrieval_query=item.retrieval_query,
                reranker_query=item.retrieval_query,
                candidate_index=int(di), full_url=row.full_url, canonical_url=row.canonical_url,
                destination_domain=row.destination_domain, destination_text=row.destination_text,
                filter_status=row.filter_status, content_kind=row.content_kind,
                tfidf_score=float(lexical[qi,di]), semantic_score=float(semantic[qi,di]),
                phrase_score=float(phrase[qi,di]), specific_phrase_count=int(specific[qi,di]),
                matched_phrases=evidence.get((qi,di),''), shared_terms=overlap_terms(qi,di,explain),
                tfidf_rank=int(ranks['tfidf'][qi,di]), semantic_rank=int(ranks['semantic'][qi,di]),
                phrase_rank=int(ranks['phrase'][qi,di]), fusion_score=float(fused[qi,di])))
    pairs = pd.DataFrame(rows)
    pairs['fusion_rank'] = pairs.groupby('item_id')['fusion_score'].rank(method='first',ascending=False).astype(int)
    return pairs, ranks


def rerank_pairs(pairs, model_spec, cache_dir):
    from sentence_transformers import CrossEncoder
    from huggingface_hub import snapshot_download
    cache_dir = Path(cache_dir)
    inputs = list(zip(pairs.reranker_query, pairs.destination_text))
    key = cache_key([model_spec, inputs])
    cache = cache_dir/f'reranking-{key}.npy'
    if cache.exists():
        values = np.load(cache)
    else:
        path = snapshot_download(model_spec['model_id'], revision=model_spec['revision'],
                                 cache_dir=cache_dir/'models', local_files_only=True)
        model = CrossEncoder(path, device='cpu', max_length=384)
        values = model.predict(inputs, batch_size=32, show_progress_bar=False)
        np.save(cache, values)
    result = pairs.copy()
    result['reranker_score'] = values
    # The reranker produces raw relevance logits, not calibrated probabilities.
    result = result.sort_values(['item_id','reranker_score','fusion_score','canonical_url'],
                                ascending=[True,False,False,True]).reset_index(drop=True)
    result['reranker_rank'] = result.groupby('item_id').cumcount()+1
    result['final_score'] = result['fusion_score'] + 0.5/(60+result['reranker_rank'])
    result = result.sort_values(['item_id','final_score','reranker_score','canonical_url'],
                                ascending=[True,False,False,True]).reset_index(drop=True)
    result['final_rank'] = result.groupby('item_id').cumcount()+1
    return result


def select_pairs(pairs, document_embeddings, policy, overrides=None):
    result = pairs.copy()
    result['eligible'] = (
        result.filter_status.eq('keep')
        & result.reranker_score.ge(policy['min_reranker_score'])
        & result.semantic_score.ge(policy['min_semantic_score'])
        & ((result.specific_phrase_count.gt(0) & result.tfidf_score.ge(policy['min_tfidf_with_phrase']))
           | (result.semantic_score.ge(policy['semantic_without_phrase'])
              & result.tfidf_score.ge(policy['min_tfidf_without_phrase'])))
    )
    result['selection_status'] = 'not_selected'
    result['selection_reason'] = 'Below initial relevance evidence thresholds'
    result.loc[result.filter_status.ne('keep'), 'selection_reason'] = 'Page needs review before selection'
    result['review_decision'] = ''
    result['review_note'] = ''
    if overrides is not None and len(overrides):
        review_map = {(int(r.item_id), r.full_url): (r.decision, r.reason) for r in overrides.itertuples()}
        for index, row in result.iterrows():
            if (row.item_id, row.full_url) in review_map:
                decision, reason = review_map[(row.item_id, row.full_url)]
                result.loc[index, ['review_decision','review_note']] = [decision, reason]
                if decision == 'include' and row.filter_status == 'keep':
                    result.loc[index, 'eligible'] = True
                if decision == 'exclude':
                    result.loc[index, 'eligible'] = False
                    result.loc[index, 'selection_reason'] = 'Metadata review: '+reason
    result['review_priority'] = np.where(result['review_decision'].eq('include'), 0, 1)
    result['content_priority'] = result['content_kind'].map({'information_or_service':0, 'pdf':0, 'peer_experience':1, 'news':2, 'research_or_registry':2})
    for _, group in result.groupby('item_id', sort=False):
        group = group.sort_values(['review_priority','content_priority','final_rank','canonical_url'])
        selected = []
        domain_counts = {}
        for index, row in group.iterrows():
            if not row.eligible:
                continue
            if len(selected) >= policy['max_urls_per_item']:
                result.loc[index, 'selection_reason'] = 'Per-item shortlist limit'
                continue
            if domain_counts.get(row.destination_domain, 0) >= policy['max_urls_per_domain_per_item']:
                result.loc[index, 'selection_reason'] = 'Domain diversity limit'
                continue
            near_duplicate = False
            for previous in selected:
                other = result.loc[previous]
                similarity = float(document_embeddings[int(row.candidate_index)] @
                                   document_embeddings[int(other.candidate_index)])
                if (row.destination_domain == other.destination_domain
                        and similarity >= policy['near_duplicate_similarity']):
                    result.loc[index, 'selection_reason'] = 'Very similar metadata to '+other.full_url
                    near_duplicate = True
                    break
            if near_duplicate:
                continue
            selected.append(index)
            domain_counts[row.destination_domain] = domain_counts.get(row.destination_domain,0)+1
            result.loc[index, 'selection_status'] = 'selected'
            if row.review_decision == 'include':
                result.loc[index, 'selection_reason'] = 'Metadata review: '+row.review_note
                continue
            result.loc[index, 'selection_reason'] = (
                'Strong phrase and model evidence' if row.specific_phrase_count > 0
                else 'Semantic and lexical agreement')
    return result


def coverage_report(queries, decisions):
    rows = []
    for query in queries.itertuples():
        group = decisions.loc[decisions.item_id.eq(query.item_id)]
        selected = group.loc[group.selection_status.eq('selected')]
        rows.append(dict(item_id=query.item_id, item_text=query.item_text, section_heading=query.section_heading,
            selected_urls=len(selected), selected_domains=selected.destination_domain.nunique(),
            best_retrieved_reranker_score=float(group.reranker_score.max()),
            coverage_status='candidate_sources_found' if len(selected) else 'gap_in_current_metadata',
            limitation='Candidate metadata only; page content has not been checked'))
    return pd.DataFrame(rows)
