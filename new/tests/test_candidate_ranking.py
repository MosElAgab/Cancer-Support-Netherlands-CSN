"""Regression checks for filtering and evidence boundaries; no model download needed."""
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.candidate_ranking import prepare_candidates, phrase_scores, canonical_url


class CandidateRankingTests(unittest.TestCase):
    def prepare(self, title, url='https://example.org/page', label=None):
        urls = pd.DataFrame([dict(full_url=url,full_url_slug_en=title,url_section_en='information',
            url_text='chemotherapy treatment cancer insurance',source_page_slug_examples='cancer treatment')])
        enriched = pd.DataFrame([dict(full_url=url,link_text_en=label or title)])
        return prepare_candidates(urls,enriched).iloc[0]

    def test_source_topics_do_not_leak_into_destination(self):
        row = self.prepare('privacy policy')
        self.assertEqual(row.filter_status, 'exclude')
        self.assertNotIn('chemotherapy', row.destination_text)
        self.assertNotIn('url_text', row.index)
        self.assertEqual(row.source_page_slug_examples, 'cancer treatment')

    def test_content_substrings_do_not_trigger_exclusions(self):
        for title in ['patient privacy rights', 'contact eczema', 'cancer and work', 'login help for patients']:
            self.assertEqual(self.prepare(title).filter_status,'keep')

    def test_non_web_links_excluded_and_pdfs_kept(self):
        self.assertEqual(self.prepare('Email','mailto:public@example.org').filter_status,'exclude')
        pdf = self.prepare('nutrition advice','https://example.org/guide.pdf')
        self.assertEqual(pdf.filter_status,'keep')
        self.assertTrue(pdf.is_pdf_url)

    def test_all_english_anchor_labels_are_preserved(self):
        urls=pd.DataFrame([dict(full_url='https://example.org/page',full_url_slug_en='guide',url_section_en='care')])
        links=pd.DataFrame([dict(full_url='https://example.org/page',link_text_en=f'label {i}') for i in range(8)])
        row=prepare_candidates(urls,links).iloc[0]
        self.assertIn('label 7',row.anchor_text)

    def test_word_boundaries_and_plural_phrases(self):
        queries=pd.DataFrame([{'item_id':1}])
        candidates=pd.DataFrame({'title_text':['prolonged symptoms','second opinions'], 'anchor_text':['','']})
        vocab=pd.DataFrame([{'item_id':1,'phrase':'long','weight':1,'specificity':'supporting'},
                            {'item_id':1,'phrase':'second opinion','weight':3,'specificity':'specific'}])
        scores,counts,evidence=phrase_scores(queries,candidates,vocab)
        self.assertEqual(scores[0,0],0)
        self.assertEqual(counts[0,1],1)
        self.assertEqual(evidence[(0,1)],'second opinion')

    def test_canonicalisation_preserves_functional_queries(self):
        self.assertEqual(canonical_url('https://EXAMPLE.org/page?id=2&utm_source=test#intro'),
                         'https://example.org/page?id=2')
        self.assertNotEqual(canonical_url('https://example.org/page?id=2'),
                            canonical_url('https://example.org/page?id=3'))



class SelectionReviewTests(unittest.TestCase):
    def test_review_cannot_override_page_exclusions(self):
        import numpy as np
        from src.candidate_ranking import select_pairs
        policy = dict(min_reranker_score=-8, min_semantic_score=.35,
                      min_tfidf_with_phrase=.05, semantic_without_phrase=.55,
                      min_tfidf_without_phrase=.1, max_urls_per_item=8,
                      max_urls_per_domain_per_item=3, near_duplicate_similarity=.98)
        rows = []
        for index, status in enumerate(['keep', 'review', 'exclude']):
            rows.append(dict(item_id=1,full_url=f'https://example.org/{index}',
                canonical_url=f'https://example.org/{index}', candidate_index=index,
                filter_status=status,reranker_score=-20,semantic_score=.1,tfidf_score=.01,
                specific_phrase_count=0,content_kind='information_or_service',
                final_rank=index+1,destination_domain='example.org'))
        pairs=pd.DataFrame(rows)
        overrides=pd.DataFrame([dict(item_id=1,full_url=r['full_url'],decision='include',reason='metadata example') for r in rows])
        result=select_pairs(pairs,np.eye(3),policy,overrides)
        self.assertEqual(result.loc[result.selection_status.eq('selected'),'full_url'].tolist(),['https://example.org/0'])
        overrides.loc[0,'decision']='exclude'
        result=select_pairs(pairs,np.eye(3),policy,overrides)
        self.assertFalse(result.selection_status.eq('selected').any())


if __name__ == '__main__':
    unittest.main()
