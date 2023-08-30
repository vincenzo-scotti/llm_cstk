import pyterrier as pt

from .semantic import SemanticPTRanker
from .lexical import LexicalPTRanker
from .query import *
from .document import reset_text_col
from .utils import *
from .utils import _Singleton


__all__ = ['PTTransformerFactory']


class PTTransformerFactory(_Singleton):
    SUBMODULES: List[str] = ['semantic', 'lexical']

    def __init__(self, **kwargs):
        super().__init__()
        # Prepare kwargs
        self._submodules_params: Dict[str, Dict] = {key: dict() for key in self.SUBMODULES}
        for param_id, param_val in kwargs.items():
            if "__" not in param_id:
                raise ValueError(
                    f"Parameters for the submodules must be passed in the form 'submodule__parameter', "
                    f"received parameter with name '{param_id}'."
                )
            module, param = param_id.split("__", 1)
            self._submodules_params[module][param] = param_val
        #
        self._semantic_ranker: SemanticPTRanker = SemanticPTRanker.load(**self._submodules_params['semantic'])
        self._lexical_ranker: LexicalPTRanker = LexicalPTRanker.load(**self._submodules_params['lexical'])

    @classmethod
    def load(cls, *args, **kwargs):
        for submodule in cls.SUBMODULES:
            if submodule in kwargs:
                configs = kwargs.pop(submodule)
                for k, v in configs.items():
                    kwargs[f'{submodule}__{k}'] = v
        return super().load(*args, **kwargs)

    def query_cleaner(self, scoring: Scoring):
        if scoring == 'semantic':
            return None
        elif scoring == 'lexical':
            return pt.apply.query(
                lambda q: str().join(char if char.isalnum() else " " for char in q["query"])
            )
        else:
            raise ValueError(
                f"Unknown scoring method for (re)ranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def raw_doc_loader(
            self,
            corpus: str,
            size: Optional[int],
            stride: Optional[int],
            scoring: Scoring
    ) -> Optional[pt.Transformer]:
        if scoring == 'semantic':
            if self._semantic_ranker.index_exists(corpus, True, size, stride):
                return None
            else:
                return pt.text.get_text(
                    pt.IndexRef.of(pt.DFIndexer(self._lexical_ranker.get_index_path(corpus)).path)
                )
        elif scoring == 'lexical':
            if self._lexical_ranker.index_exists(corpus, True, size, stride):
                return None
            else:
                return pt.text.get_text(
                    pt.IndexRef.of(pt.DFIndexer(self._lexical_ranker.get_index_path(corpus)).path)
                )
        else:
            raise ValueError(
                f"Unknown scoring method for (re)ranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_chunks_generator(
            self,
            corpus: str,
            size: Optional[int],
            stride: Optional[int],
            scoring: Scoring,
            snippet: bool = False,
    ) -> Optional[pt.Transformer]:
        if scoring == 'semantic':
            if self._semantic_ranker.index_exists(corpus, True, size, stride):
                return None
            else:
                return pt.apply.generic(reset_text_col) >> pt.text.sliding(
                    text_attr=BODY if snippet else TEXT,
                    length=size,
                    stride=stride,
                    title_attr=None if snippet else TITLE
                )
        elif scoring == 'lexical':
            if self._lexical_ranker.index_exists(corpus, True, size, stride):
                return None
            else:
                return pt.apply.generic(reset_text_col) >> pt.text.sliding(
                    text_attr=BODY if snippet else TEXT,
                    length=size,
                    stride=stride,
                    title_attr=None if snippet else TITLE
                )
        else:
            raise ValueError(
                f"Unknown scoring method for (re)ranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_ranker(
            self,
            scoring: Scoring,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None,
            reranking: bool = False,
            cutoff: bool = True
    ) -> pt.Transformer:
        if scoring == 'semantic':
            return self._semantic_ranker.get_ranking_model(
                corpus,
                chunk_doc=chunk_doc,
                chunk_size=chunk_size,
                chunk_stride=chunk_stride,
                reranking=reranking,
                cutoff=cutoff
            )
        elif scoring == 'lexical':
            return self._lexical_ranker.get_ranking_model(corpus, reranking=reranking, cutoff=cutoff)
        else:
            raise ValueError(
                f"Unknown scoring method for ranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_reranker(
            self,
            scoring: Scoring,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None
    ) -> pt.Transformer:
        if scoring == 'semantic':
            return self._semantic_ranker.get_reranking_model(
                corpus, chunk_doc=chunk_doc, chunk_size=chunk_size, chunk_stride=chunk_stride
            )
        elif scoring == 'lexical':
            return self._lexical_ranker.get_reranking_model(corpus)
        else:
            raise ValueError(
                f"Unknown scoring method for reranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_scorer(
            self,
            scoring: Scoring,
            corpus: str,
            ranking: bool = True
    ) -> pt.Transformer:
        if scoring == 'semantic':
            return self._semantic_ranker.get_scoring_model(corpus, txt_col=BODY, ranking=ranking)
        elif scoring == 'lexical':
            return self._lexical_ranker.get_scoring_model(corpus, txt_col=BODY, ranking=ranking)
        else:
            raise ValueError(
                f"Unknown scoring method for (re)ranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_chunks_aggregator(self, method: DocAggregation) -> pt.Transformer:
        if method == 'max':
            return pt.text.max_passage()
        elif method == 'mean':
            return pt.text.mean_passage()
        else:
            raise ValueError(
                f"Unknown document chunks aggregation method: \'{method}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in DocAggregation)}"
            )

    def query_chunks_aggregator(self, method: QueryAggregation) -> pt.Transformer:
        if method == 'max':
            return pt.apply.generic(max_query_passage)
        elif method == 'mean':
            return pt.apply.generic(min_query_passage)
        else:
            raise ValueError(
                f"Unknown query chunks aggregation method: \'{method}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in QueryAggregation)}"
            )
