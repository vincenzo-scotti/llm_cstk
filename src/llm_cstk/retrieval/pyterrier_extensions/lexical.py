import pyterrier as pt

from .document import PTDocManager
from .ranker import _PTRanker
from .utils import *
from .utils import _Singleton


class LexicalPTRanker(_PTRanker, _Singleton):
    def __init__(
            self, ranking_model: Optional[LexicalSearch], reranking_model: Optional[LexicalSearch], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Scoring approaches
        self.ranking_model: Optional[LexicalSearch] = ranking_model
        self.reranking_model: Optional[LexicalSearch] = reranking_model
    
    def _get_model(
            self, model: LexicalSearch, corpus: str, metadata: bool = False
    ) -> pt.Transformer:
        # Get model id
        model_id: str = str(np.uint(hash(f'{model}_{corpus}')))
        # Check whether the model is not already in cache
        if model_id not in self._model_cache:
            # Load index
            pt_indexer: pt.Indexer = pt.DFIndexer(self.get_index_dir_path(corpus))
            pt_index_ref = pt.IndexRef.of(pt_indexer.path)
            pt_index: pt.IndexRef = pt.IndexFactory.of(pt_index_ref)
            # Build retriever model
            pt_transformer: pt.Transformer = pt.BatchRetrieve(
                pt_index,  wmodel=LEXICAL_SEARCH_MAPPING[model], metadata=METADATA if metadata else None
            )
            # Cache model
            self._model_cache[model_id] = pt_transformer

        return self._model_cache[model_id]

    def get_ranking_model(
            self, corpus: str, reranking: bool = False, cutoff: bool = True, metadata: bool = False
    ) -> pt.Transformer:
        if cutoff:
            return self._get_model(
                self.ranking_model, corpus, metadata=metadata or reranking
            ) % (self.ranking_cutoff if not reranking else self.reranking_cutoff)
        else:
            return self._get_model(self.ranking_model, corpus, metadata=metadata or reranking)

    def get_reranking_model(self, corpus: str, metadata: bool = False) -> pt.Transformer:
        return self._get_model(self.reranking_model, corpus, metadata=metadata)

    def get_scoring_model(self, corpus: str, txt_col: str = TEXT, ranking: bool = True):
        # Get model id
        model = self.ranking_model if ranking else self.reranking_model
        model_id: str = str(np.uint(hash(f'{model}_{corpus}')))
        # Check whether the model is not already in cache
        if model_id not in self._scoring_model_cache:
            # Load index
            pt_indexer: pt.Indexer = pt.DFIndexer(self.get_index_dir_path(corpus))
            pt_index_ref = pt.IndexRef.of(pt_indexer.path)
            pt_index: pt.IndexRef = pt.IndexFactory.of(pt_index_ref)
            # Build retriever model
            pt_transformer: pt.Transformer = pt.text.scorer(
                body_attr=txt_col,
                wmodel=LEXICAL_SEARCH_MAPPING[model],
                background_index=pt_index
            )
            # Cache model
            self._scoring_model_cache[model_id] = pt_transformer

        return self._scoring_model_cache[model_id]

    def get_index_path(self, corpus: str) -> str:
        return PTDocManager.get_lexical_index_path(self.data_dir_path, corpus)
