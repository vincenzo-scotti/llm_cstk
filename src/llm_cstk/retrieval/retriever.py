from functools import lru_cache

import pandas as pd
import pyterrier as pt

from functools import partial
from nltk import word_tokenize

from typing import Callable, Optional, Tuple

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
        # Tokeniser
        self._tokeniser: Callable = partial(word_tokenize, preserve_line=False)

    @classmethod
    def load(cls, *args, **kwargs):
        for submodule in cls.SUBMODULES:
            if submodule in kwargs:
                configs = kwargs.pop(submodule)
                for k, v in configs.items():
                    kwargs[f'{submodule}__{k}'] = v
        return super().load(*args, **kwargs)

    def _build_search_pipeline(
            self,
            corpus: str,
            ranking: Scoring,
            reranking: Optional[Scoring],
            chunk_doc: bool,
            doc_chunk_size: Optional[int],
            doc_chunk_stride: Optional[int],
            doc_chunks_aggregation: Optional[DocAggregation],
            query_chunks_aggregation: Optional[QueryAggregation],
    ) -> pt.Transformer:
        # Query cleaning (optional)
        ranking_query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(ranking)
        reranking_query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(
            reranking
        ) if reranking is not None else None
        # Doc loading (optional)
        raw_doc_loader: Optional[pt.Transformer] = self._transformer_factory.raw_doc_loader(
            corpus, chunk_doc, doc_chunk_size, doc_chunk_stride
        )
        # Doc chunking (optional)
        doc_chunks_generator: Optional[pt.Transformer] = self._transformer_factory.doc_chunks_generator(
            corpus, doc_chunk_size, doc_chunk_stride
        ) if chunk_doc else None
        # Scoring
        doc_ranker: pt.Transformer = self._transformer_factory.doc_ranker(
            ranking,
            corpus,
            chunk_doc=chunk_doc,
            chunk_size=doc_chunk_size,
            chunk_stride=doc_chunk_stride,
            reranking=reranking is not None,
        )
        # Reranking (optional)
        doc_reranker: Optional[pt.Transformer] = self._transformer_factory.doc_reranker(
            reranking, corpus, chunk_doc=chunk_doc, chunk_size=doc_chunk_size, chunk_stride=doc_chunk_stride
        ) if reranking is not None else None
        # Doc chunks aggregation (optional)
        doc_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.doc_chunks_aggregator(
            doc_chunks_aggregation
        ) if chunk_doc and doc_chunks_aggregation is not None else None
        # Query parts aggregation (optional)
        query_chunks_aggregator: Optional[pt.Transformer] = self._transformer_factory.query_chunks_aggregator(
            query_chunks_aggregation
        ) if query_chunks_aggregation is not None else None
        # Compose pipeline
        pipeline: pt.Transformer = doc_ranker
        if doc_chunks_generator is not None:
            pipeline = doc_chunks_generator >> pipeline
        if raw_doc_loader is not None:
            pipeline = raw_doc_loader >> pipeline
        if ranking_query_cleaner is not None:
            pipeline = ranking_query_cleaner >> pipeline
        if reranking_query_cleaner is not None:
            pipeline >>= reranking_query_cleaner
        if doc_reranker is not None:
            pipeline >>= doc_reranker
        if doc_chunks_aggregator is not None:
            pipeline >>= doc_chunks_aggregator
        if query_chunks_aggregator is not None:
            pipeline >>= query_chunks_aggregator

        return pipeline

    def _build_snippet_pipeline(
            self,
            search_results: pd.DataFrame,
            corpus: str,
            ranking: Scoring,
            reranking: Optional[Scoring],
            doc_chunk_size: Optional[int],
            doc_chunk_stride: Optional[int],
            query_chunks_aggregation: Optional[QueryAggregation]
    ) -> pt.Transformer:
        # Query cleaning (optional)
        scoring_query_cleaner: Optional[pt.Transformer] = self._transformer_factory.query_cleaner(
            reranking
        ) if reranking is not None else self._transformer_factory.query_cleaner(ranking)
        # Doc loading (optional)
        raw_doc_loader: Optional[pt.Transformer] = self._transformer_factory.raw_doc_loader(
            corpus, True, doc_chunk_size, doc_chunk_stride
        ) if BODY not in search_results.columns else None
        # Doc chunking
        doc_chunks_generator: pt.Transformer = self._transformer_factory.doc_chunks_generator(
            corpus, doc_chunk_size, doc_chunk_stride, snippet=True
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
        if query_chunks_aggregator is not None:
            pipeline >>= query_chunks_aggregator

        return pt.text.snippets(pipeline, joinstr=SNIPPET_SEP)

    def _prepare_query(
            self,
            query: Union[str, List[str]],
            chunk_query: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None,
    ) -> pd.DataFrame:
        # TODO rework with query expansion
        # Prepare query in correct format
        if isinstance(query, str):
            if chunk_query:
                tokenised_query: List[str] = self._tokeniser(query)
                query = [
                    tokenised_query[idx*chunk_stride:idx*chunk_stride+chunk_size]
                    for idx in range(0, len(tokenised_query), chunk_stride)
                ]
            else:
                query = [query]
        if len(query) > 1:
            query = pd.DataFrame({QID: [f'q%p{i + 1}' for i, q in range(len(query))], QUERY: query})
        elif len(query) == 1:
            query = pd.DataFrame({QID: ['q'], QUERY: query})
        else:
            raise ValueError(f"Improper query format: \'{query}\'")

        return query

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

    @lru_cache
    def search(
            self,
            query: Union[str, List[str]],
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
        query: pd.DataFrame = self._prepare_query(query, chunk_query, query_chunk_size, query_chunk_stride)
        # Build search pipeline
        search_pipeline: pt.Transformer = self._build_search_pipeline(
            corpus,
            ranking,
            reranking,
            chunk_doc,
            doc_chunk_size,
            doc_chunk_stride,
            doc_score_aggregation,
            query_score_aggregation
        )
        # Run search
        results: pd.DataFrame = search_pipeline.transform(query)

        return results

    @lru_cache
    def snippet(
            self,
            search_results: pd.DataFrame,
            query: Union[str, List[str]],
            corpus: str,
            doc_chunk_size: int,
            doc_chunk_stride: int,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            chunk_query: bool = False,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = None,
    ) -> pd.DataFrame:
        # Prepare query
        query: pd.DataFrame = self._prepare_query(query, chunk_query, query_chunk_size, query_chunk_stride)
        #
        if QUERY in search_results.columns:
            search_results = search_results.drop([QID, QUERY])
        search_results = search_results.merge(query, how='cross')
        # Build snippet generation pipeline
        snippet_pipeline: pt.Transformer = self._build_snippet_pipeline(
            search_results,
            corpus,
            ranking,
            reranking,
            doc_chunk_size,
            doc_chunk_stride,
            query_score_aggregation
        )
        # Run search
        results: pd.DataFrame = snippet_pipeline.transform(search_results)

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
            chunk_doc: bool = False,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None,
            doc_score_aggregation: DocAggregation = 'max',
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: QueryAggregation = 'mean'
    ):
        # Input sanity check
        assert not chunk_doc or doc_score_aggregation is not None
        assert isinstance(query, str) or query_score_aggregation is not None
        # Run long query on documents
        return self.search(
            query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            chunk_doc=chunk_doc,
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
            query,
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
            search_results: pd.DataFrame,
            query: str,
            corpus: str,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            doc_chunk_size: Optional[int] = None,
            doc_chunk_stride: Optional[int] = None
    ) -> pd.DataFrame:
        #
        return self.snippet(
            search_results,
            query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride
        )

    def generate_snippet_long_query(
            self,
            search_results: pd.DataFrame,
            query: Union[str, List[str]],
            corpus: str,
            doc_chunk_size: int,
            doc_chunk_stride: int,
            ranking: Scoring = 'semantic',
            reranking: Optional[Scoring] = None,
            query_chunk_size: Optional[int] = None,
            query_chunk_stride: Optional[int] = None,
            query_score_aggregation: Optional[QueryAggregation] = None
    ) -> pd.DataFrame:
        # Input sanity check
        assert not isinstance(query, str) or query_score_aggregation is not None
        #
        return self.snippet(
            search_results,
            query,
            corpus,
            ranking=ranking,
            reranking=reranking,
            doc_chunk_size=doc_chunk_size,
            doc_chunk_stride=doc_chunk_stride,
            chunk_query=isinstance(query, str),
            query_chunk_size=query_chunk_size,
            query_chunk_stride=query_chunk_stride,
            query_score_aggregation=query_score_aggregation
        )

    def add_corpus(
            self,
            corpus: str,
            docs: pd.DataFrame,
            docs_chunked: Optional[pd.DataFrame] = None,
            chunking_configs: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
            overwrite: bool = False
    ):
        self._doc_manager.register_corpus(
            corpus, docs, docs_chunked=docs_chunked, chunking_configs=chunking_configs, overwrite=overwrite
        )

    def add_large_corpus(
            self,
            corpus: str,
            docs: pd.DataFrame,
            docs_chunked: Optional[pd.DataFrame] = None,
            chunking_configs: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
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
