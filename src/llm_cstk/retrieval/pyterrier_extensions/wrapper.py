import os

from functools import partial

import pickle
import bz2

import pandas as pd

from sentence_transformers import util

from pyterrier.model import coerce_queries_dataframe

from typing import Callable, Tuple, Optional

from .utils import *


class _SemanticPTTransformer(pt.Transformer):
    TRANSFORMER_TYPE: Optional[Union[Type[SentenceTransformer], Type[CrossEncoder]]] = None
    _transformer_encoder_cache: Dict[str, SentenceTransformer] = dict()

    def __init__(
            self,
            transformer: str,
            data_df: Optional[pd.DataFrame] = None,
            metadata: bool = False,
            device: Optional[torch.device] = None
    ):
        self.transformer: str = transformer
        self.data_df: Optional[pd.DataFrame] = data_df
        self.metadata: bool = metadata
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._transformer_encoder: SentenceTransformer = self._load_transformer_encoder()

    def _load_transformer_encoder(self) -> SentenceTransformer:
        if self.transformer not in self._transformer_encoder_cache:
            transformer_encoder = self.TRANSFORMER_TYPE(self.transformer, device=self.device.type)
            self._transformer_encoder_cache[self.transformer] = transformer_encoder

        return self._transformer_encoder_cache[self.transformer]

    def _prepare_data_search(self, queries: pd.DataFrame) -> Tuple:
        raise NotImplementedError()

    def _get_doc_ids(self, search_results: pd.DataFrame) -> List[str]:
        raise NotImplementedError()

    def _get_doc_idxs_rescore(self, input_results: pd.DataFrame) -> List[List[int]]:
        if DOCID in input_results.columns:
            return input_results.groupby(QID, sort=False)[DOCID].astype(int).apply(list).values.tolist()
        elif DOCNO in input_results.columns:
            if DOCID in self.data_df.columns:
                ...
            else:
                ...
        else:
            raise ValueError("Unable to recover document indices.")

    def _prepare_data_rescore(self, queries: pd.DataFrame, input_results: pd.DataFrame) -> Tuple:
        raise NotImplementedError()

    def _prepare_data(self, queries: pd.DataFrame, input_results: Optional[pd.DataFrame]) -> Tuple:
        if input_results is None:
            return self._prepare_data_search(queries)
        else:
            return self._prepare_data_rescore(queries, input_results)

    def _semantic_search(self, *args) -> pd.DataFrame:
        raise NotImplementedError()

    def _semantic_rescore(self, *args) -> pd.DataFrame:
        raise NotImplementedError()

    def _search(self, input_results: pd.DataFrame, *search_input) -> pd.DataFrame:
        if input_results is None:
            # Run classic search if no previous result is given
            return self._semantic_search(*search_input)
        else:
            # Re-rank existing results
            return self._semantic_rescore(*search_input)

    def transform(self, queries: Union[str, List[str], pd.DataFrame]) -> pd.DataFrame:
        # Adapted from https://pyterrier.readthedocs.io/en/latest/_modules/pyterrier/batchretrieve.html#BatchRetrieve
        # Prepare queries  #NOTE this is taken from BatchRetrieve
        if not isinstance(queries, pd.DataFrame):
            queries = coerce_queries_dataframe(queries)
        # Check if document IDs or scores are provided  #NOTE this is taken from BatchRetrieve
        docno_provided: bool = DOCNO in queries.columns
        docid_provided: bool = DOCID in queries.columns
        scores_provided: bool = SCORE in queries.columns
        # Isolate unique queries if the model is being used for rescoring  #NOTE this is adapted from BatchRetrieve
        input_results: Optional[pd.DataFrame] = None
        if docno_provided or docid_provided:
            input_results = queries
            queries = input_results[[QID, QUERY]].dropna(axis=0, subset=[QUERY]).drop_duplicates()
        # Enforce string type on query IDs  #NOTE this is adapted from BatchRetrieve
        if queries[QID].dtype != str:
            queries[QID] = queries[QID].astype(str)
        # Prepare data for search
        search_input = self._prepare_data(queries, input_results)
        # Run queries
        search_results: pd.DataFrame
        search_results = self._search(input_results, *search_input)
        # Add optional metadata
        if self.metadata:  # TODO check back this part
            search_results = search_results.merge(self.data_df[METADATA], on=[DOCID])
        # Add missing data from query
        query_output_columns = queries.columns[
            (queries.columns == QID) | (~queries.columns.isin(search_results.columns))
        ]
        search_results = search_results.merge(queries[query_output_columns], on=[QID])
        if DOCNO not in search_results.columns:
            if self.data_df is not None:
                search_results = search_results.merge(self.data_df[[DOCID, DOCNO]], on=[DOCID])
            else:
                search_results[DOCNO] = self._get_doc_ids(search_results)

        return search_results

    def _encoder_apply(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError

    def to_semantic_scorer(self, **kwargs) -> pt.Transformer:
        # Create re-scorer
        scorer = pt.apply.doc_score(partial(self._encoder_apply, **kwargs))

        return scorer


class BiEncoderPTTransformer(_SemanticPTTransformer):
    TRANSFORMER_TYPE = SentenceTransformer

    def __init__(
            self,
            *args,
            ann: Optional[ANNSearch] = None,
            ann_index_path: Optional[str] = None,
            pre_computed_embeddings_path: Optional[str] = None,
            tensors: bool = True,
            normalise: bool = True,
            top_k: Optional[int] = None,
            indexing_params: Optional[Dict] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Transformer
        self.ann: Optional[ANNSearch] = ann
        self.ann_index_path: Optional[str] = ann_index_path
        self._ann_index: Optional[ANNIndex] = None
        self.indexing_params: Optional[Dict] = indexing_params
        if self.indexing_params is None and self.ann is not None:
            self.indexing_params = ANN_SEARCH_INDEX_DEFAULT_PARAMETERS[self.ann]
        self.pre_computed_embeddings_path: Optional[str] = pre_computed_embeddings_path
        self._pre_computed_embeddings: Optional[EmbeddingsCache] = None
        # Vector space params
        self.tensors: bool = tensors
        self.normalise: bool = normalise
        self.top_k: Optional[int] = top_k
        #
        self._score_fn: Callable = util.dot_score if self.normalise else util.cos_sim
        self._pairwise_score_fn: Callable = util.pairwise_dot_score if self.normalise else util.pairwise_cos_sim
        self._stacking_fn: Callable = torch.hstack if self.tensors else np.hstack
        #
        if self.ann is not None and self.ann_index_path is not None and os.path.exists(self.ann_index_path):
            self.load_ann_index()
        if pre_computed_embeddings_path is not None and os.path.exists(self.pre_computed_embeddings_path):
            self.load_pre_computed_embeddings()

    def embed_text(self, text: Union[str, List[str]]) -> EmbeddingVector:
        if isinstance(text, str):
            return self.embed_text([text])[0]
        embeddings = self._transformer_encoder.encode(
            text,
            normalize_embeddings=self.normalise,
            convert_to_numpy=not self.tensors,
            convert_to_tensor=self.tensors,
            device=self.device.type
        )

        return embeddings

    def load_pre_computed_embeddings(self, path: Optional[str] = None) -> EmbeddingsCache:
        # Update attributes if required
        if path is not None:
            self.pre_computed_embeddings_path = path
        # Load embedding cache from binary file
        with bz2.BZ2File(self.pre_computed_embeddings_path, 'rb') as f:
            self._pre_computed_embeddings = pickle.load(f)

        return self._pre_computed_embeddings

    def save_pre_computed_embeddings(self, path: Optional[str] = None):
        # Update attributes if required
        if path is not None:
            self.pre_computed_embeddings_path = path
        # Serialise embedding cache in binary file
        with bz2.BZ2File(self.pre_computed_embeddings_path, 'wb') as f:
            pickle.dump(self._pre_computed_embeddings, f)

    def compute_doc_embeddings(self) -> EmbeddingsCache:
        docs: List[str] = self.data_df[TEXT].values.tolist()
        docnos: List[str] = self.data_df[DOCNO].values.tolist()
        embeddings: EmbeddingVector = self.embed_text(docs)
        self._pre_computed_embeddings = {DOCNO_BIENC_CACHE: docnos, EMBEDDINGS_BIENC_CACHE: embeddings}

        return self._pre_computed_embeddings

    def load_ann_index(self, ann: Optional[ANNSearch] = None, path: Optional[str] = None):
        # Update attributes if required
        if ann is not None:
            self.ann = ann
        if path is not None:
            self.ann_index_path = path
        # Load index depending on ANN indexing tool
        if self.ann == 'annoy':
            raise NotImplementedError()
        elif self.ann == 'faiss':
            raise NotImplementedError()
        elif self.ann == 'hnswlib':
            # Create the HNSWLIB index
            self._ann_index = hnswlib.Index(
                space=DOT_PROD_SPACE if self.normalise else COS_SIM_SPACE,
                dim=self._transformer_encoder.get_sentence_embedding_dimension(),
            )
            # Load from file
            self._ann_index.load_index(self.ann_index_path)
        elif self.ann is None:
            pass
        else:
            raise ValueError(
                f"Unknown approximated nearest neighbor search approach: \'{self.ann}\', "
                f"accepted values are `None` or {', '.join(f'{repr(t)}' for t in ANNSearch)}"
            )

    def save_ann_index(self, path: Optional[str] = None):
        # Update attributes if required
        if path is not None:
            self.ann_index_path = path
        # Save index depending on ANN indexing tool
        if self.ann == 'annoy':
            raise NotImplementedError()
        elif self.ann == 'faiss':
            raise NotImplementedError()
        elif self.ann == 'hnswlib':
            self._ann_index.save_index(self.ann_index_path)
        elif self.ann is None:
            pass
        else:
            raise ValueError(
                f"Unknown approximated nearest neighbor search approach: \'{self.ann}\', "
                f"accepted values are `None` or {', '.join(f'{repr(t)}' for t in ANNSearch)}"
            )

    def build_ann_index(self, ann: Optional[ANNSearch] = None, indexing_params: Optional[Dict] = None):
        # Update attributes if required
        if ann is not None:
            self.ann = ann
        if indexing_params is not None:
            self.indexing_params = indexing_params
        # Compute embeddings if not available
        if self._pre_computed_embeddings is None:
            self.compute_doc_embeddings()
        # Build index depending on ANN indexing tool
        if self.ann == 'annoy':
            raise NotImplementedError()
        elif self.ann == 'faiss':
            raise NotImplementedError()
        elif self.ann == 'hnswlib':
            # Create the HNSWLIB index
            self._ann_index = hnswlib.Index(
                space=DOT_PROD_SPACE if self.normalise else COS_SIM_SPACE,
                dim=self._transformer_encoder.get_sentence_embedding_dimension()
            )
            # Initialise index
            self._ann_index.init_index(
                len(self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE]), **self.indexing_params
            )
            # Train the index to find a suitable clustering
            self._ann_index.add_items(
                self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE],
                range(len(self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE]))
            )
        elif self.ann is None:
            pass
        else:
            raise ValueError(
                f"Unknown approximated nearest neighbor search approach: \'{self.ann}\', "
                f"accepted values are `None` or {', '.join(f'{repr(t)}' for t in ANNSearch)}"
            )

    def _get_doc_ids(self, search_results: pd.DataFrame) -> List[str]:
        idxs: List[int] = search_results[DOCID].astype(int).values.tolist()
        if self._pre_computed_embeddings is not None:
            ids: List[str] = [self._pre_computed_embeddings[DOCNO_BIENC_CACHE][idx] for idx in idxs]
        else:
            raise ValueError("Unable to recover `docno` information")

        return ids

    def _prepare_data_search(self, queries: pd.DataFrame) -> Tuple[List[str], EmbeddingVector]:
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Embed queries
        query_embeddings = self.embed_text(queries[QUERY].values.tolist())

        return query_ids, query_embeddings

    def _prepare_data_rescore(
            self, queries: pd.DataFrame, input_results: pd.DataFrame
    ) -> Tuple[List[str], EmbeddingVector, List[List[int]], EmbeddingVector]:
        # NOTE duplicates are removed by transform
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Embed queries
        query_embeddings = self.embed_text(queries[QUERY].values)
        # Get document ids
        doc_idxs: List[List[int]] = self._get_doc_idxs_rescore(input_results)
        # Embed documents (or retrieve cached embeddings)
        if self._pre_computed_embeddings is None and TEXT in input_results.columns:
            doc_embeddings = self.embed_text(input_results[TEXT].values)
        elif self.data_df is not None:
            if self._pre_computed_embeddings is None:
                self.compute_doc_embeddings()
            doc_embeddings = self._stacking_fn(
                [self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE][idxs] for idxs in doc_idxs]
            )
        else:
            raise ValueError('Missing document text')

        return query_ids, query_embeddings, doc_idxs, doc_embeddings

    def _ann_index_search(
            self, query_embeddings: EmbeddingVector
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self._ann_index, hnswlib.Index):
            # Run approximated search using HNSWLib
            idxs, distances = self._ann_index.knn_query(
                query_embeddings,
                k=self.top_k if self.top_k is not None else len(self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE]),  # NOTE: it is assuming embeddings exists if index exists
                filter=None
            )
            # Compute scores
            scores = 1 - distances
            ranks = np.argsort(-scores, axis=1)
        else:
            raise NotImplementedError()

        return idxs, scores, ranks

    def _semantic_search(self, query_ids: List[str], query_embeddings: EmbeddingVector) -> pd.DataFrame:
        # Depending on the approach, search through the collection
        if self._ann_index is not None:
            idxs, scores, ranks = self._ann_index_search(query_embeddings)
            # Put together results
            results: List[Dict] = [
                {QID: query_id, DOCID: idx, RANK: rank, SCORE: score}
                for query_id, idxs, ranks, scores in zip(query_ids, idxs, ranks, scores)
                for rank, (idx, score) in enumerate(zip(idxs[ranks], scores[ranks]))
            ]
        else:
            if self._pre_computed_embeddings is None:
                self.compute_doc_embeddings()
            # Plain search on entire collection embeddings
            hits = util.semantic_search(
                query_embeddings,
                self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE],
                score_function=self._score_fn,
                top_k=self.top_k if self.top_k is not None else len(self._pre_computed_embeddings[EMBEDDINGS_BIENC_CACHE])
            )
            # Put together results
            results: List[Dict] = [
                {QID: query_id, DOCID: hit['corpus_id'], RANK: rank, SCORE: hit['score']}
                for query_id, query_hits in zip(query_ids, hits)
                for rank, hit in enumerate(query_hits)
            ]
        # Store results in data frame
        results: pd.DataFrame = pd.DataFrame(results)

        return results

    def _semantic_rescore(
            self,
            query_ids: List[str],
            query_embeddings: EmbeddingVector,
            doc_ids: List[List[int]],
            doc_embeddings: List[EmbeddingVector]
    ) -> pd.DataFrame:
        # Create accumulator for results
        results: List[Dict] = []
        # Compute similarity between query-document pairs
        for query_id, query_embeds, idxs, doc_embeds in zip(query_ids, query_embeddings, doc_ids, doc_embeddings):
            # Compute new scores and new ranks
            scores: np.ndarray = self._score_fn(query_embeds, doc_embeds).squeeze()
            ranks: np.ndarray = np.argsort(-scores, axis=-1)
            # Add current results to accumulator
            idxs = np.array(idxs)
            results.extend([
                {QID: query_id, DOCID: idx, RANK: rank, SCORE: score}
                for rank, (idx, score) in enumerate(zip(idxs[ranks], scores[ranks]))
            ])
        # Store results in data frame
        results: pd.DataFrame = pd.DataFrame(results)

        return results

    def _encoder_apply(self, df: pd.DataFrame, txt_col: str = TEXT):
        # Embed queries
        query_embeddings = self.embed_text(df[QUERY].values.tolist())
        # Embed documents
        doc_embeddings = self.embed_text(df[txt_col].values.tolist())
        # Compute cosine similarity
        scores = self._pairwise_score_fn(query_embeddings, doc_embeddings)

        return scores


class CrossEncoderPTTransformer(_SemanticPTTransformer):
    TRANSFORMER_TYPE = CrossEncoder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_data_search(self, queries: pd.DataFrame) -> Tuple[List[str], List[str]]:
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Get query text
        queries = queries[QUERY].values.tolist()

        return query_ids, queries

    def _prepare_data_rescore(
            self, queries: pd.DataFrame, input_results: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[List[int]], List[List[str]]]:
        # NOTE duplicates are removed by transform
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Get query text
        queries = queries[QUERY].values.tolist()
        # Get document ids
        doc_idxs: List[List[int]] = self._get_doc_idxs_rescore(input_results)
        # Get documents text
        if TEXT in input_results.columns:
            # Get text from data frame
            docs: List[List[str]] = input_results.groupby(QID, sort=False)[TEXT].apply(list).values.tolist()
        # Use document text from stored documents
        elif self.data_df is not None:
            # Get text from indices
            docs = [self.data_df.iloc[idxs][TEXT].values.tolist() for idxs in doc_idxs]
        else:
            raise ValueError('Missing document text')

        return query_ids, queries, doc_idxs, docs

    def semantic_search(self, query_ids: List[str], queries: List[str]) -> pd.DataFrame:
        # Get documents
        docs: List[str] = self.data_df[TEXT].values.tolist()
        idxs: np.ndarray = self.data_df[DOCNO].values
        # Create accumulator for results
        results: List[Dict] = []
        # Iterate over queries
        for query_id, query in zip(query_ids, queries):
            scores = self._transformer_encoder.predict([[query, doc] for doc in docs])
            ranks: np.ndarray = np.argsort(-scores, axis=1)
            # Add current results to accumulator
            results.extend([
                {QID: query_id, DOCID: idx, RANK: rank, SCORE: score}
                for rank, (idx, score) in enumerate(zip(idxs[ranks], scores[ranks]))
            ])
        # Store results in data frame
        results: pd.DataFrame = pd.DataFrame(results)

        return results

    def _semantic_rescore(
            self, query_ids: List[str], queries: List[str], doc_ids: List[List[int]], documents: List[List[str]]
    ) -> pd.DataFrame:
        # Create accumulator for results
        results: List[Dict] = []
        # Compute similarity between query-document pairs
        for query_id, query, idxs, docs in zip(query_ids, queries, doc_ids, documents):
            # Compute new scores and new ranks
            scores: np.ndarray = self._transformer_encoder.predict([[query, doc] for doc in docs])
            ranks: np.ndarray = np.argsort(-scores, axis=1)
            # Add current results to accumulator
            idxs = np.array(idxs)
            results.extend([
                {QID: query_id, DOCID: idx, RANK: rank, SCORE: score}
                for rank, (idx, score) in enumerate(zip(idxs[ranks], scores[ranks]))
            ])
        # Store results in data frame
        results: pd.DataFrame = pd.DataFrame(results)

        return results

    def _encoder_apply(self, df: pd.DataFrame, txt_col: str = TEXT):
        # Queries
        queries = df[QUERY].values
        # Docs
        docs = df[txt_col].values
        # Compute score
        scores = self._transformer_encoder.predict(list([q, d] for q, d in zip(queries, docs)))

        return scores
