from functools import lru_cache

import pandas as pd
import pyterrier as pt

from .pyterrier_extensions import *
from .pyterrier_extensions.utils import *
from .pyterrier_extensions.utils import _Singleton


if not pt.started():
    pt.init()


# TODO create separate instances for full-text/passages/utterances


class DocRetriever(_Singleton):
    SUBMODULES: List[str] = ['doc_manager', 'transformer_factory']

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
        # Documents manager
        self._doc_manager: PTDocManager = PTDocManager.load(**self._submodules_params['doc_manager'])
        # Transformer
        self._transformer_factory: PTTransformerFactory = PTTransformerFactory.load(
            **self._submodules_params['transformer_factory']
        )

    @classmethod
    def load(cls, *args, **kwargs):
        for submodule in cls.SUBMODULES:
            if submodule in kwargs:
                configs = kwargs.pop(submodule)
                for k, v in configs.items():
                    kwargs[f'{submodule}__{k}'] = v
        return super().load(*args, **kwargs)

    def _build_scoring_pipeline(
            self,
            corpus: str,
            ranking: Scoring,
            reranking: Optional[Scoring],
            chunk_doc: bool,
            doc_chunk_size: Optional[int],
            doc_chunk_stride: Optional[int],
            doc_chunks_aggregation: Optional[DocAggregation],
            query_chunks_aggregation: Optional[QueryAggregation],
    ):
        # Query cleaning (optional)
        query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(ranking)
        # Doc chunking (optional)
        doc_chunks_generator: Optional[pt.Transformer] = self._transformer_factory.doc_chunks_generator(
            corpus, doc_chunk_size, doc_chunk_stride, ranking
        ) if chunk_doc and doc_chunk_size is not None and doc_chunk_stride is not None else None
        # Scorer
        doc_scorer: pt.Transformer = self._transformer_factory.doc_scorer(
            ranking if reranking is None else reranking, corpus=corpus, ranking=reranking is None
        )
        # Doc chunks aggregation (optional)
        doc_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.doc_chunks_aggregator(
            doc_chunks_aggregation
        ) if chunk_doc and doc_chunks_aggregation is not None else None
        # Query parts aggregation (optional)
        query_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.query_chunks_aggregator(
            query_chunks_aggregation
        ) if query_chunks_aggregation is not None else None
        # Compose pipeline
        pipeline: pt.Transformer = doc_scorer
        if doc_chunks_generator is not None:
            pipeline = doc_chunks_generator >> pipeline
        if query_cleaner is not None:
            pipeline = query_cleaner >> pipeline
        if doc_chunks_aggregator is not None:
            pipeline >>= doc_chunks_aggregator
        if query_chunks_aggregator is not None:
            pipeline >>= query_chunks_aggregator

        return pipeline

    def _build_search_pipeline(
            self,
            query: Union[str, List[str]],
            corpus: str,
            ranking: Scoring,
            reranking: Optional[Scoring],
            chunk_doc: bool,
            doc_chunk_size: Optional[int],
            doc_chunk_stride: Optional[int],
            doc_chunks_aggregation: Optional[DocAggregation],
            chunk_query: bool,
            query_chunk_size: Optional[int],
            query_chunk_stride: Optional[int],
            query_chunks_aggregation: Optional[QueryAggregation],
    ) -> pt.Transformer:
        # Query pre-processing (optional)
        query_pre_processor: Optional[pt.Transformer] = self._transformer_factory.query_pre_processor(
            query_chunk_size, query_chunk_stride
        ) if isinstance(query, list) or chunk_query else None
        # Query cleaning (optional)
        ranking_query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(ranking)
        reranking_query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(
            reranking
        ) if reranking is not None else None
        # Doc pre-ranking (optional)
        doc_pre_ranker: pt.Transformer = self._transformer_factory.doc_ranker(
            'lexical', corpus, metadata=True, preranking=True  # ranking, corpus, metadata=True  # TODO find better solution like averaging doc embeddings when registering them
        ) if chunk_doc and doc_chunk_size is not None and doc_chunk_stride is not None else None
        # Doc chunking (optional)
        doc_chunks_generator: Optional[pt.Transformer] = self._transformer_factory.doc_chunks_generator(
            corpus, doc_chunk_size, doc_chunk_stride, ranking
        ) if chunk_doc and doc_chunk_size is not None and doc_chunk_stride is not None else None
        # Scoring
        doc_ranker: pt.Transformer = self._transformer_factory.doc_ranker(
            ranking,
            corpus,
            chunk_doc=chunk_doc and doc_chunk_size is None and doc_chunk_stride is None,
            reranking=reranking is not None
        )
        # Reranking (optional)
        doc_reranker: Optional[pt.Transformer] = self._transformer_factory.doc_reranker(
            reranking, corpus, chunk_doc=chunk_doc and doc_chunk_size is None and doc_chunk_stride is None
        ) if reranking is not None else None
        # Doc chunks aggregation (optional)
        doc_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.doc_chunks_aggregator(
            doc_chunks_aggregation
        ) if chunk_doc and doc_chunks_aggregation is not None else None
        # Query parts aggregation (optional)
        query_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.query_chunks_aggregator(
            query_chunks_aggregation
        ) if query_chunks_aggregation is not None else None
        # Sorting optional
        sorter: Optional[pt.Transformer] = self._transformer_factory.get_sorter() if (
            (chunk_doc and doc_chunks_aggregation) or query_chunks_aggregation
        ) else None
        # Compose pipeline
        pipeline: pt.Transformer = doc_ranker
        if doc_chunks_generator is not None:
            pipeline = doc_chunks_generator >> pipeline
        if doc_pre_ranker is not None:
            pipeline = doc_pre_ranker >> pipeline
        if ranking_query_cleaner is not None:
            pipeline = ranking_query_cleaner >> pipeline
        if query_pre_processor is not None:
            pipeline = query_pre_processor >> pipeline
        if reranking_query_cleaner is not None:
            pipeline >>= reranking_query_cleaner
        if doc_reranker is not None:
            pipeline >>= doc_reranker
        if doc_chunks_aggregator is not None:
            pipeline >>= doc_chunks_aggregator
        if query_chunks_aggregator is not None:
            pipeline >>= query_chunks_aggregator
        if sorter is not None:
            pipeline >>= sorter

        return pipeline

    def _build_snippet_pipeline(
            self,
            search_results: pd.DataFrame,
            query: Union[str, List[str]],
            corpus: str,
            ranking: Scoring,
            reranking: Optional[Scoring],
            doc_chunk_size: Optional[int],
            doc_chunk_stride: Optional[int],
            chunk_query: bool,
            query_chunk_size: Optional[int],
            query_chunk_stride: Optional[int],
            query_chunks_aggregation: Optional[QueryAggregation],
            n_passages: int
    ) -> pt.Transformer:
        # Query pre-processing (optional)
        query_pre_processor: Optional[pt.Transformer] = self._transformer_factory.query_pre_processor(
            query_chunk_size, query_chunk_stride
        ) if isinstance(query, list) or chunk_query else None
        # Query cleaning (optional)
        scoring_query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(
            reranking
        ) if reranking is not None else self._transformer_factory.query_cleaner(ranking)
        # Doc loading (optional)
        raw_doc_loader: Optional[pt.Transformer] = self._transformer_factory.raw_doc_loader(
            corpus
        ) if BODY not in search_results.columns else None
        # Doc chunking
        doc_chunks_generator: pt.Transformer = self._transformer_factory.doc_chunks_generator(
            corpus, doc_chunk_size, doc_chunk_stride, reranking if reranking is not None else ranking, snippet=True
        )
        # Scorer
        doc_scorer: pt.Transformer = self._transformer_factory.doc_scorer(ranking, corpus, ranking=reranking is None)
        # Query parts aggregation (optional)
        query_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.query_chunks_aggregator(
            query_chunks_aggregation
        ) if query_chunks_aggregation is not None else None
        # Compose pipeline
        pipeline: pt.Transformer = doc_scorer
        if doc_chunks_generator is not None:
            pipeline = doc_chunks_generator >> pipeline
        if raw_doc_loader is not None:
            pipeline = raw_doc_loader >> pipeline
        if scoring_query_cleaner is not None:
            pipeline = scoring_query_cleaner >> pipeline
        if query_pre_processor is not None:
            pipeline = query_pre_processor >> pipeline
        if query_chunks_aggregator is not None:
            pipeline >>= query_chunks_aggregator

        return pt.text.snippets(pipeline, joinstr=SNIPPET_SEP, num_psgs=n_passages)

    @staticmethod
    def _post_process_search(query: Union[str, List[str]], results: pd.DataFrame):
        return results[[DOCNO, SCORE]].to_dict(orient='list') | {QUERY: query}

    @staticmethod
    def _post_process_snippet(query: Union[str, List[str]], results: pd.DataFrame):
        return results[[DOCNO, SCORE, SUMMARY]].to_dict(orient='list') | {QUERY: query}

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

    @lru_cache
    def score(
            self,
            query: Tuple[str],
            doc: Tuple[str],
            corpus: Optional[str] = None,
            scoring: Scoring = 'semantic',
            rescoring: Optional[Scoring] = None,
            chunk_doc: bool = False,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            doc_score_aggregation: Optional[DocAggregation] = None,
            chunk_query: bool = False,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = None
    ):  # TODO Update this method to use same approach of search
        query: List[str] = list(query)
        doc: List[str] = list(doc)
        # Prepare query
        if chunk_query:
            tokenised_query: List[List[str]] = [self._tokeniser(q) for q in query]
            query = [
                ' '.join(tokenised_q[idx:idx + query_chunk_size])
                for tokenised_q in tokenised_query
                for idx in range(0, len(tokenised_q), query_chunk_stride)
            ]
            doc = sum(([d] * len(tokenised_q) for tokenised_q, d in zip(tokenised_query, doc)), start=list())
        # Prepare data
        data_df: pd.DataFrame = pd.DataFrame(zip(query, doc), columns=[QUERY, TEXT])
        # Build scoring pipeline
        scoring_pipeline: pt.Transformer = self._build_scoring_pipeline(
            corpus,
            scoring,
            rescoring,
            chunk_doc,
            doc_chunk_size,
            doc_chunk_stride,
            doc_score_aggregation,
            query_score_aggregation
        )
        # Run search
        results: pd.DataFrame = scoring_pipeline.transform(data_df)

        return results

    @lru_cache
    def search(
            self,
            query: Union[str, Tuple[str]],
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            chunk_doc: bool = False,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            doc_score_aggregation: Optional[DocAggregation] = None,
            chunk_query: bool = False,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = None,
    ) -> pd.DataFrame:
        # Prepare query
        query: Union[str, List[str]] = list(query) if isinstance(query, tuple) else query
        # Build search pipeline
        search_pipeline: pt.Transformer = self._build_search_pipeline(
            query,
            corpus,
            ranking,
            reranking,
            chunk_doc,
            doc_chunk_size,
            doc_chunk_stride,
            doc_score_aggregation,
            chunk_query,
            query_chunk_size,
            query_chunk_stride,
            query_score_aggregation
        )
        # Run search
        results: pd.DataFrame = search_pipeline.transform(pd.DataFrame({QID: ['q'], QUERY: [query]}))
        # Post-process results
        results = self._post_process_search(query, results)

        return results

    @lru_cache
    def snippet(
            self,
            search_results: Tuple,
            query: Union[str, Tuple[str]],
            corpus: str,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            chunk_query: bool = False,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = None,
            n_passages: int = 1
    ) -> pd.DataFrame:
        # Prepare search results
        keys, values = search_results
        search_results: pd.DataFrame = pd.DataFrame(values, columns=keys)
        # Prepare query
        query: Union[str, List[str]] = list(query) if isinstance(query, tuple) else query
        # Build snippet generation pipeline
        snippet_pipeline: pt.Transformer = self._build_snippet_pipeline(
            search_results,
            query,
            corpus,
            ranking,
            reranking,
            doc_chunk_size,
            doc_chunk_stride,
            chunk_query,
            query_chunk_size,
            query_chunk_stride,
            query_score_aggregation,
            n_passages
        )
        #
        search_results[QID] = ['q'] * len(search_results)
        search_results[QUERY] = [query] * len(search_results)
        # Run search
        results: pd.DataFrame = snippet_pipeline.transform(search_results)
        # Post-process snippet results
        results = self._post_process_snippet(query, results)

        return results

    def corpus(
            self,
            corpus: str,
            docs: Optional[pd.DataFrame] = None,
            docs_chunked: Optional[pd.DataFrame] = None,
            chunking_configs: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
            transformer: Optional[Union[Dict, List[Dict]]] = None,
            large: bool = False,
            overwrite: bool = False
    ):
        raise NotImplementedError()

    def score_query_doc_pair(
            self,
            query: Union[str, List[str]],
            doc: Union[str, List[str]],
            corpus: Optional[str] = None,
            scoring: Scoring = 'semantic',
            rescoring: Optional[Scoring] = None,
            chunk_doc: bool = False,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            doc_score_aggregation: Optional[DocAggregation] = 'max',
            chunk_query: bool = False,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = 'mean'
    ):
        assert not chunk_doc or doc_score_aggregation is not None
        assert not chunk_query or query_score_aggregation is not None

        return self.score(
            tuple(query) if isinstance(query, list) else (query,),
            tuple(doc) if isinstance(doc, list) else (doc,),
            corpus=corpus,
            scoring=scoring,
            rescoring=rescoring,
            chunk_doc=chunk_doc,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride,
            doc_score_aggregation=doc_score_aggregation,
            chunk_query=chunk_query,
            query_chunk_size=query_chunk_size,
            query_chunk_stride=query_chunk_stride,
            query_score_aggregation=query_score_aggregation
        )

    def search_doc(
            self,
            query: str,
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            chunk: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None,
            score_aggregation: DocAggregation = 'max'
    ) -> pd.DataFrame:
        # Input sanity check
        assert not chunk or score_aggregation is not None
        # Run query on documents
        return self.search(
            query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            chunk_doc=chunk,
            doc_chunk_size=chunk_size,
            doc_chunk_stride=chunk_stride,
            doc_score_aggregation=score_aggregation
        )

    def search_doc_chunk(
            self,
            query: str,
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None
    ):
        # Run query on document chunks
        return self.search(
            query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            chunk_doc=True,
            doc_chunk_size=chunk_size,
            doc_chunk_stride=chunk_stride,
            doc_score_aggregation=None
        )

    def search_doc_long_query(
            self,
            query: Union[str, List[str]],
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            chunk: bool = False,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            doc_score_aggregation: DocAggregation = 'max',
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: QueryAggregation = 'mean'
    ):
        # Input sanity check
        assert not chunk or doc_score_aggregation is not None
        assert isinstance(query, str) or query_score_aggregation is not None

        # Run long query on documents
        return self.search(
            tuple(query) if isinstance(query, list) else query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            chunk_doc=chunk,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride,
            doc_score_aggregation=doc_score_aggregation,
            chunk_query=isinstance(query, str),
            query_chunk_size=query_chunk_size,
            query_chunk_stride=query_chunk_stride,
            query_score_aggregation=query_score_aggregation
        )

    def search_doc_chunk_long_query(
            self,
            query: Union[str, List[str]],
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = 'mean'
    ):
        # Input sanity check
        assert isinstance(query, str) or query_score_aggregation is not None
        # Run long query on document chunks
        return self.search(
            tuple(query) if isinstance(query, list) else query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            chunk_doc=True,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride,
            doc_score_aggregation=None,
            chunk_query=isinstance(query, str),
            query_chunk_size=query_chunk_size,
            query_chunk_stride=query_chunk_stride,
            query_score_aggregation=query_score_aggregation
        )

    def generate_snippet(
            self,
            search_results: Dict[str, List],
            query: str,
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            n_passages: int = 1
    ) -> pd.DataFrame:
        #
        return self.snippet(
            (tuple(search_results.keys()), tuple(zip(search_results.values()))),
            query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride,
            n_passages=n_passages
        )

    def generate_snippet_long_query(
            self,
            search_results: Dict[str, List],
            query: Union[str, List[str]],
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = 'mean',
            n_passages: int = 1
    ) -> pd.DataFrame:
        # Input sanity check
        assert not isinstance(query, str) or query_score_aggregation is not None
        #
        return self.snippet(
            (tuple(search_results.keys()), tuple(zip(search_results.values()))),
            tuple(query) if isinstance(query, list) else query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride,
            chunk_query=isinstance(query, str),
            query_chunk_size=query_chunk_size,
            query_chunk_stride=query_chunk_stride,
            query_score_aggregation=query_score_aggregation,
            n_passages=n_passages
        )

    def add_corpus(
            self,
            corpus: str,
            docs: pd.DataFrame,
            docs_chunked: Optional[pd.DataFrame] = None,
            overwrite: bool = False
    ):
        self._doc_manager.register_corpus(
            corpus, docs, docs_chunked=docs_chunked, overwrite=overwrite
        )

    def add_large_corpus(
            self,
            corpus: str,
            docs: pd.DataFrame,
            docs_chunked: Optional[pd.DataFrame] = None,
            overwrite: bool = False
    ):
        raise NotImplementedError()

    def index_corpus(
            self,
            corpus: str,
            transformer: Optional[Union[Dict, List[Dict]]] = None,
            overwrite: bool = False
    ):
        self._doc_manager.index_corpus(
            corpus,
            transformers=transformer if isinstance(transformer, list) or transformer is None else [transformer],
            overwrite=overwrite
        )

    def index_large_corpus(
            self,
            corpus: str,
            transformer: Optional[Union[str, List[str]]] = None,
            overwrite: bool = False
    ):
        raise NotImplementedError()
