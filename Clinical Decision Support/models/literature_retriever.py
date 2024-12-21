from typing import Any, List, Dict, Optional
import numpy as np
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger('literature_retriever')

class LiteratureRetriever:
    """
    Manages retrieval and ranking of relevant medical literature.
    
    Features:
    - Semantic search for relevant papers
    - Citation tracking
    - Recency weighting
    - Evidence level classification
    """
    
    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model
        self.evidence_levels = {
            'systematic_review': 1,
            'rct': 2,
            'cohort_study': 3,
            'case_control': 4,
            'case_series': 5
        }
    
    def rank_literature(
        self,
        documents: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        min_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank medical literature based on relevance, recency, and evidence level.
        
        Args:
            documents: List of medical literature documents
            query_embedding: Embedded query vector
            min_year: Minimum publication year to consider
            
        Returns:
            List of ranked documents
        """
        current_year = datetime.now().year
        ranked_docs = []
        
        for doc in documents:
            # Calculate semantic similarity
            doc_embedding = self.embedding_model.encode(doc['abstract'])
            similarity = np.dot(query_embedding, doc_embedding)
            
            # Calculate recency score
            years_old = current_year - doc['publication_year']
            recency_score = 1 / (1 + 0.1 * years_old)  # Decay factor of 0.1
            
            # Get evidence level score
            evidence_score = 1 / self.evidence_levels.get(doc['study_type'], 6)
            
            # Calculate final score
            final_score = (
                0.5 * similarity +  # 50% weight on relevance
                0.3 * recency_score +  # 30% weight on recency
                0.2 * evidence_score  # 20% weight on evidence level
            )
            
            if min_year is None or doc['publication_year'] >= min_year:
                ranked_docs.append({
                    'document': doc,
                    'score': final_score,
                    'relevance': similarity,
                    'recency': recency_score,
                    'evidence_level': evidence_score
                })
        
        # Sort by final score
        ranked_docs.sort(key=lambda x: x['score'], reverse=True)
        return ranked_docs