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
            corpus: str
    ) -> Optional[pt.Transformer]:
        if self._lexical_ranker.index_exists(corpus, True):
            pt_index_ref = pt.IndexRef.of(self._lexical_ranker.get_index_path(corpus))
            pt_index: pt.IndexRef = pt.IndexFactory.of(pt_index_ref)

            return pt.text.get_text(pt_index, metadata=[DOCNO] + METADATA)
        else:
            # TODO add manual document text loading
            raise NotImplementedError()

    def doc_chunks_generator(
            self,
            corpus: str,
            size: Optional[int],
            stride: Optional[int],
            scoring: Scoring,
            snippet: bool = False
    ) -> Optional[pt.Transformer]:
        if scoring == 'semantic':
            if self._semantic_ranker.index_exists(corpus, True) and size is None and stride is None:
                return None
            else:
                if size is not None and stride is not None:
                    return pt.apply.generic(reset_text_col) >> pt.text.sliding(
                        text_attr=TEXT,
                        length=size,
                        stride=stride,
                        prepend_attr=None if snippet else TITLE
                    )
                else:
                    return pt.apply.generic(reset_text_col) >> pt.text.sliding(
                        text_attr=TEXT, prepend_attr=None if snippet else TITLE
                    )
        elif scoring == 'lexical':
            if self._lexical_ranker.index_exists(corpus, True) and size is None and stride is None:
                return None
            else:
                if size is not None and stride is not None:
                    return pt.apply.generic(reset_text_col) >> pt.text.sliding(
                        text_attr=TEXT,
                        length=size,
                        stride=stride,
                        prepend_attr=None if snippet else TITLE
                    )
                else:
                    return pt.apply.generic(reset_text_col) >> pt.text.sliding(
                        text_attr=TEXT, prepend_attr=None if snippet else TITLE
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
            reranking: bool = False,
            cutoff: bool = True,
            metadata: bool = False,
            preranking: bool = False
    ) -> pt.Transformer:
        if scoring == 'semantic':
            if preranking:
                return self._semantic_ranker.get_ranking_model(
                    corpus,
                    chunk_doc=chunk_doc,
                    reranking=reranking,
                    cutoff=cutoff,
                    metadata=metadata
                ) % self._semantic_ranker.reranking_cutoff
            else:
                return self._semantic_ranker.get_ranking_model(
                    corpus,
                    chunk_doc=chunk_doc,
                    reranking=reranking,
                    cutoff=cutoff,
                    metadata=metadata
                )
        elif scoring == 'lexical':
            if preranking:
                return self._lexical_ranker.get_ranking_model(
                    corpus, chunk_doc=chunk_doc, reranking=reranking, cutoff=cutoff, metadata=True
                ) % self._lexical_ranker.reranking_cutoff
            else:
                return self._lexical_ranker.get_ranking_model(
                    corpus, chunk_doc=chunk_doc, reranking=reranking, cutoff=cutoff, metadata=metadata
                )
        else:
            raise ValueError(
                f"Unknown scoring method for ranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_reranker(
            self,
            scoring: Scoring,
            corpus: str,
            chunk_doc: bool = False
    ) -> pt.Transformer:
        if scoring == 'semantic':
            return self._semantic_ranker.get_reranking_model(corpus, chunk_doc=chunk_doc)
        elif scoring == 'lexical':
            return self._lexical_ranker.get_reranking_model(corpus, chunk_doc=chunk_doc)
        else:
            raise ValueError(
                f"Unknown scoring method for reranking: \'{scoring}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def doc_scorer(
            self,
            scoring: Scoring,
            corpus: Optional[str] = None,
            ranking: bool = True,
            txt_col: str = TEXT,
    ) -> pt.Transformer:
        if scoring == 'semantic':
            return self._semantic_ranker.get_scoring_model(corpus=corpus, txt_col=txt_col, ranking=ranking)
        elif scoring == 'lexical':
            return self._lexical_ranker.get_scoring_model(corpus, txt_col=txt_col, ranking=ranking)
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
            return pt.apply.generic(mean_query_passage)
        else:
            raise ValueError(
                f"Unknown query chunks aggregation method: \'{method}\', "
                f"accepted values are {', '.join(f'{repr(t)}' for t in QueryAggregation)}"
            )
