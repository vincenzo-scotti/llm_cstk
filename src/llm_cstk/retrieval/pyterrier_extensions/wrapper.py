import os

from functools import partial

import pickle
import bz2

import torch
from sentence_transformers import util

import pandas as pd
import pyterrier as pt
from pyterrier.model import coerce_queries_dataframe

from .utils import *


class _SemanticPTTransformer(pt.Transformer):
    TRANSFORMER_TYPE: Optional[Union[Type[SentenceTransformer], Type[CrossEncoder]]] = None
    _transformer_encoder_cache: Dict[str, Union[SentenceTransformer, CrossEncoder]] = dict()
    _data_df_cache: Dict[str, pd.DataFrame] = dict()
    _essential_data_df_cache: Dict[str, pd.DataFrame] = dict()

    def __init__(
            self,
            transformer: str,
            data_df_path: Optional[str] = None,
            metadata: bool = False,
            device: Optional[torch.device] = None,
            batch_size: int = 32,
            **semantic_model_kwargs
    ):
        self.transformer: str = transformer
        self.data_df_path: Optional[str] = data_df_path
        self.metadata: bool = metadata
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.semantic_model_kwargs: Dict = semantic_model_kwargs if semantic_model_kwargs is not None else dict()
        self.batch_size: int = batch_size

        self._transformer_encoder: SentenceTransformer = self._load_transformer_encoder()
        self._data_df: Optional[pd.DataFrame] = self._load_data_df(
            self.data_df_path
        ) if self.data_df_path is not None and self.metadata else None
        self._essential_data_df: Optional[pd.DataFrame] = self._load_essential_data_df(
            self.data_df_path
        ) if self.data_df_path is not None and not self.metadata else None

    def _load_transformer_encoder(self) -> SentenceTransformer:
        if self.transformer not in self._transformer_encoder_cache:
            transformer_encoder = self.TRANSFORMER_TYPE(
                self.transformer, device=self.device.type, **self.semantic_model_kwargs
            )
            self._transformer_encoder_cache[self.transformer] = transformer_encoder

        return self._transformer_encoder_cache[self.transformer]

    def _load_data_df(self, path: str) -> pd.DataFrame:
        if path not in self._data_df_cache:
            self._data_df_cache[path] = pd.read_csv(path, dtype=DTYPES)

        return self._data_df_cache[path]

    def _load_essential_data_df(self, path: str) -> pd.DataFrame:
        if path not in self._essential_data_df_cache:
            self._essential_data_df_cache[path] = pd.read_csv(
                path, dtype=DTYPES_ESSENTIAL, usecols=list(DTYPES_ESSENTIAL.keys())
            )

        return self._essential_data_df_cache[path]

    def _prepare_data_search(self, queries: pd.DataFrame) -> Tuple:
        raise NotImplementedError()

    def _get_doc_ids(self, idxs: List[int]) -> List[Tuple[str, str]]:
        # Get DataFrame with document info
        df: pd.DataFrame = self._essential_data_df if self._essential_data_df is not None else self._data_df

        return df.iloc[idxs][[DOCNO, DOCID]].values.tolist()

    def _get_doc_idxs(self, ids: List[str]) -> List[int]:
        # Get DataFrame with document info
        df: pd.DataFrame = self._essential_data_df if self._essential_data_df is not None else self._data_df
        #
        ids_mapping = {id_: i for i, id_ in enumerate(ids)}

        return sorted(df[df[DOCNO].isin(ids)].index.values.tolist(), key=lambda idx: ids_mapping[df.iloc[idx][DOCNO]])

    # Gather index of the document in the data frame
    def _get_doc_ids_rescore(self, input_results: pd.DataFrame) -> List[List[Tuple[str, str]]]:
        assert DOCNO in input_results.columns and DOCID in input_results.columns, "Unable to recover document identifiers."

        return [
            input_results[[DOCNO, DOCID]].values.tolist()
            for _, input_results_group in input_results.groupby(QID, sort=False)
        ]

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
        # Prepare queries  # NOTE this is taken from BatchRetrieve
        if not isinstance(queries, pd.DataFrame):
            queries = coerce_queries_dataframe(queries)
        # Check if document IDs or scores are provided  # NOTE this is taken from BatchRetrieve
        docno_provided: bool = DOCNO in queries.columns
        docid_provided: bool = DOCID in queries.columns
        scores_provided: bool = SCORE in queries.columns
        # Isolate unique queries if the model is being used for rescoring  # NOTE this is adapted from BatchRetrieve
        input_results: Optional[pd.DataFrame] = None
        if docno_provided or docid_provided:
            input_results = queries
            queries = input_results[[QID, QUERY]].dropna(axis=0, subset=[QUERY]).drop_duplicates()
        # Enforce string type on query IDs  # NOTE this is adapted from BatchRetrieve
        if queries[QID].dtype != str:
            queries[QID] = queries[QID].astype(str)
        # Prepare data for search
        search_input = self._prepare_data(queries, input_results)
        # Run queries
        search_results: pd.DataFrame = self._search(input_results, *search_input)
        # Add optional metadata
        if self.metadata and search_results[DOCNO].isin(self._data_df[DOCNO]).all():
            # NOTE this should cause problems when using metadata on custom chunks of the documents,
            #  however such case should not happen
            search_results = search_results.merge(self._data_df[[DOCNO] + METADATA], on=[DOCNO])
        elif self.metadata and input_results is not None:
            input_results_output_columns = input_results.columns[
                (input_results.columns == DOCNO) | (~input_results.columns.isin(search_results.columns))
            ]
            search_results = search_results.merge(input_results[input_results_output_columns], on=[DOCNO])
        # Add missing data from query
        query_output_columns = queries.columns[
            (queries.columns == QID) | (~queries.columns.isin(search_results.columns))
        ]
        search_results = search_results.merge(queries[query_output_columns], on=[QID])

        return search_results

    def _encoder_apply(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError()

    def to_semantic_scorer(self, **kwargs) -> pt.Transformer:
        # Create re-scorer
        scorer = pt.apply.doc_score(partial(self._encoder_apply, **kwargs), batch_size=self.batch_size)

        return scorer


class BiEncoderPTTransformer(_SemanticPTTransformer):
    TRANSFORMER_TYPE = SentenceTransformer
    _ann_index_cache: Dict[str, ANNIndex] = dict()
    _pre_computed_embeddings_cache: Dict[str, EmbeddingsCache] = dict()

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
        indexing_params = ANN_SEARCH_INDEX_DEFAULT_PARAMETERS.get(ann) if indexing_params is None else indexing_params
        super().__init__(*args, **kwargs)
        # Transformer
        self.ann: Optional[ANNSearch] = ann
        self.ann_index_path: Optional[str] = ann_index_path
        self.indexing_params: Optional[Dict] = indexing_params
        self.pre_computed_embeddings_path: Optional[str] = pre_computed_embeddings_path
        # Vector space params
        self.tensors: bool = tensors
        self.normalise: bool = normalise
        self.top_k: Optional[int] = top_k
        #
        self._score_fn: Callable = util.dot_score if self.normalise else util.cos_sim
        self._pairwise_score_fn: Callable = util.pairwise_dot_score if self.normalise else util.pairwise_cos_sim
        self._stacking_fn: Callable = torch.hstack if self.tensors else np.hstack
        #
        self._ann_index: Optional[ANNIndex] = None
        if self.ann is not None and self.ann_index_path is not None:
            self._ann_index: Optional[ANNIndex] = self._load_ann_index()
        self._pre_computed_embeddings: Optional[EmbeddingVector] = None
        if pre_computed_embeddings_path is not None:
            self._pre_computed_embeddings: Optional[EmbeddingVector] = self._load_pre_computed_embeddings()

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
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu()

        return embeddings

    def _load_pre_computed_embeddings(self) -> Optional[EmbeddingVector]:
        if self.pre_computed_embeddings_path is not None and os.path.exists(self.pre_computed_embeddings_path):
            if self.pre_computed_embeddings_path not in self._pre_computed_embeddings_cache:
                # Load embedding cache from binary file
                with bz2.BZ2File(self.pre_computed_embeddings_path, 'rb') as f:
                    self._pre_computed_embeddings_cache[self.pre_computed_embeddings_path] = pickle.load(f)
            return self._pre_computed_embeddings_cache[self.pre_computed_embeddings_path]
        else:
            return None

    def load_pre_computed_embeddings(self, path: Optional[str] = None) -> EmbeddingsCache:
        # Update attributes if required
        if path is not None:
            self.pre_computed_embeddings_path = path
        # Load embedding cache from binary file
        self._pre_computed_embeddings = self._load_pre_computed_embeddings()

        return self._pre_computed_embeddings

    def save_pre_computed_embeddings(self, path: str):
        # Serialise embedding cache in binary file
        with bz2.BZ2File(path, 'wb') as f:
            pickle.dump(self._pre_computed_embeddings, f)

    def compute_doc_embeddings(self) -> EmbeddingVector:
        docs: List[str] = self._data_df[TEXT].values.tolist()
        embeddings: EmbeddingVector = self.embed_text(docs)
        if isinstance(embeddings, torch.Tensor):
            embeddings.cpu()
        self._pre_computed_embeddings = embeddings

        return self._pre_computed_embeddings

    def _load_ann_index(self):
        if self.ann is not None and self.ann_index_path is not None and os.path.exists(self.ann_index_path):
            if self.ann_index_path not in self._ann_index_cache:
                # Load index depending on ANN indexing tool
                if self.ann == 'annoy':
                    raise NotImplementedError()
                elif self.ann == 'faiss':
                    raise NotImplementedError()
                elif self.ann == 'hnswlib':
                    # Create the HNSWLIB index
                    self._ann_index_cache[self.ann_index_path] = hnswlib.Index(
                        space=DOT_PROD_SPACE if self.normalise else COS_SIM_SPACE,
                        dim=self._transformer_encoder.get_sentence_embedding_dimension(),
                    )
                    # Load from file
                    self._ann_index_cache[self.ann_index_path].load_index(self.ann_index_path)
                else:
                    raise ValueError(
                        f"Unknown approximated nearest neighbor search approach: \'{self.ann}\', "
                        f"accepted values are `None` or {', '.join(f'{repr(t)}' for t in ANNSearch)}"
                    )
            return self._ann_index_cache[self.ann_index_path]
        else:
            return None

    def load_ann_index(self, ann: Optional[ANNSearch] = None, path: Optional[str] = None) -> Optional[ANNIndex]:
        # Update attributes if required
        if ann is not None:
            self.ann = ann
        if path is not None:
            self.ann_index_path = path
        # Load index depending on ANN indexing tool
        self._ann_index = self._load_ann_index()

        return self._ann_index

    def save_ann_index(self, path: str):
        # Save index depending on ANN indexing tool
        if self.ann == 'annoy':
            raise NotImplementedError()
        elif self.ann == 'faiss':
            raise NotImplementedError()
        elif self.ann == 'hnswlib':
            self._ann_index.save_index(path)
        elif self.ann is None:
            pass
        else:
            raise ValueError(
                f"Unknown approximated nearest neighbor search approach: \'{self.ann}\', "
                f"accepted values are `None` or {', '.join(f'{repr(t)}' for t in ANNSearch)}"
            )

    def _build_ann_index(self):
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
            self._ann_index.init_index(len(self._pre_computed_embeddings), **self.indexing_params)
            # Train the index to find a suitable clustering
            self._ann_index.add_items(self._pre_computed_embeddings, range(len(self._pre_computed_embeddings)))
        else:
            raise ValueError(
                f"Unknown approximated nearest neighbor search approach: \'{self.ann}\', "
                f"accepted values are `None` or {', '.join(f'{repr(t)}' for t in ANNSearch)}"
            )

        return self._ann_index

    def build_ann_index(self, ann: Optional[ANNSearch] = None, indexing_params: Optional[Dict] = None):
        # Update attributes if required
        if ann is not None:
            self.ann = ann
        if indexing_params is not None:
            self.indexing_params = indexing_params
        #
        self._build_ann_index()

    def _prepare_data_search(self, queries: pd.DataFrame) -> Tuple[List[str], EmbeddingVector]:
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Embed queries
        query_embeddings = self.embed_text(queries[QUERY].values.tolist())

        return query_ids, query_embeddings

    def _prepare_data_rescore(
            self, queries: pd.DataFrame, input_results: pd.DataFrame
    ) -> Tuple[List[str], EmbeddingVector, List[List[Tuple[str, str]]], List[EmbeddingVector]]:
        # NOTE duplicates are removed by transform
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Embed queries
        query_embeddings = self.embed_text(queries[QUERY].values)
        # Get document ids
        doc_ids: List[List[Tuple[str, str]]] = self._get_doc_ids_rescore(input_results)
        # Embed documents (or retrieve cached embeddings)
        df: pd.DataFrame = self._essential_data_df if self._essential_data_df is not None else self._data_df
        if TEXT in input_results.columns and not input_results[DOCNO].isin(df[DOCNO]).all():
            doc_embeddings = [
                self.embed_text(res_df[TEXT].values) for _, res_df in input_results.groupby(QID, sort=False)
            ]
        elif self._data_df is not None:
            if self._pre_computed_embeddings is None:
                self.compute_doc_embeddings()
            doc_embeddings = [
                self._pre_computed_embeddings[self._get_doc_idxs(res_df[DOCNO].values.tolist())]
                for _, res_df in input_results.groupby(QID, sort=False)
            ]
        else:
            raise ValueError('Missing document text')

        return query_ids, query_embeddings, doc_ids, doc_embeddings

    def _ann_index_search(
            self, query_embeddings: EmbeddingVector
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self._ann_index, hnswlib.Index):
            # Run approximated search using HNSWLib
            idxs, distances = self._ann_index.knn_query(
                query_embeddings,
                k=self.top_k if self.top_k is not None else len(self._pre_computed_embeddings),  # NOTE: it is assuming embeddings exists if index exists
                filter=None
            )
            # Compute scores
            scores = 1 - distances
            ordering = np.argsort(-scores, axis=1)
        else:
            raise NotImplementedError()

        return idxs, scores, ordering

    def _semantic_search(self, query_ids: List[str], query_embeddings: EmbeddingVector) -> pd.DataFrame:
        # Depending on the approach, search through the collection
        if self._ann_index is not None:
            idxs, scores, ordering = self._ann_index_search(query_embeddings)
            # Put together results
            results: List[Dict] = [
                {QID: query_id, DOCNO: docno, DOCID: docid, RANK: rank, SCORE: score}
                for query_id, idxs, ordering, scores in zip(query_ids, idxs, ordering, scores)
                for rank, ((docno, docid), score) in enumerate(zip(self._get_doc_ids(idxs[ordering]), scores[ordering]))
            ]
        else:
            if self._pre_computed_embeddings is None:
                self.compute_doc_embeddings()
            # Plain search on entire collection embeddings
            hits = util.semantic_search(
                query_embeddings,
                self._pre_computed_embeddings,
                score_function=self._score_fn,
                top_k=self.top_k if self.top_k is not None else len(self._pre_computed_embeddings)
            )
            idxs = [[hit['corpus_id'] for hit in q_hits] for q_hits in hits]
            scores = [[hit['score'] for hit in q_hits] for q_hits in hits]
            # Put together results
            results: List[Dict] = [
                {QID: query_id, DOCNO: docno, DOCID: docid, RANK: rank, SCORE: score}
                for query_id, idxs, scores in zip(query_ids, idxs, scores)
                for rank, ((docno, docid), score) in enumerate(zip(self._get_doc_ids(idxs), scores))
            ]
        # Store results in data frame
        results: pd.DataFrame = pd.DataFrame(results)

        return results

    def _semantic_rescore(
            self,
            query_ids: List[str],
            query_embeddings: EmbeddingVector,
            doc_ids: List[List[Tuple[str, str]]],
            doc_embeddings: List[EmbeddingVector]
    ) -> pd.DataFrame:
        # Create accumulator for results
        results: List[Dict] = []
        # Compute similarity between query-document pairs
        for query_id, query_embeds, ids, doc_embeds in zip(query_ids, query_embeddings, doc_ids, doc_embeddings):
            # Compute new scores and new ranks
            scores: np.ndarray = self._score_fn(query_embeds, doc_embeds).squeeze()
            ordering: np.ndarray = np.argsort(-scores, axis=-1)
            # Add current results to accumulator
            results.extend([
                {QID: query_id, DOCNO: docno, DOCID: docid, RANK: rank, SCORE: score}
                for rank, ((docno, docid), score) in enumerate(zip([ids[i] for i in ordering], scores[ordering]))
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
    def _prepare_data_search(self, queries: pd.DataFrame) -> Tuple[List[str], List[str]]:
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Get query text
        queries = queries[QUERY].values.tolist()

        return query_ids, queries

    def _prepare_data_rescore(
            self, queries: pd.DataFrame, input_results: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[List[Tuple[str, str]]], List[List[str]]]:
        # NOTE duplicates are removed by transform
        # Get query ids
        query_ids = queries[QID].values.tolist()
        # Get query text
        queries = queries[QUERY].values.tolist()
        # Get document ids
        doc_ids: List[List[Tuple[str, str]]] = self._get_doc_ids_rescore(input_results)
        # Get documents text
        if TEXT in input_results.columns:
            # Get text from data frame
            docs: List[List[str]] = input_results.groupby(QID, sort=False)[TEXT].apply(list).values.tolist()
        # Use document text from stored documents
        elif self._data_df is not None:
            # Get text from indices
            docs = [
                self._data_df.iloc[self._get_doc_idxs([docno for docno, _ in ids])][TEXT].values.tolist()
                for ids in doc_ids
            ]
        else:
            raise ValueError('Missing document text')

        return query_ids, queries, doc_ids, docs

    def semantic_search(self, query_ids: List[str], queries: List[str]) -> pd.DataFrame:
        # Get documents
        docs: List[str] = self._data_df[TEXT].values.tolist()
        ids: np.ndarray = self._data_df[[DOCNO, DOCID]].values
        # Create accumulator for results
        results: List[Dict] = []
        # Iterate over queries
        for query_id, query in zip(query_ids, queries):
            scores = self._transformer_encoder.predict([[query, doc] for doc in docs]).squeeze()
            if len(scores.shape) > 1:  # TODO fixme
                scores = scores[:, 0]
            ordering: np.ndarray = np.argsort(-scores, axis=-1)
            # Add current results to accumulator
            results.extend([
                {QID: query_id, DOCNO: docno, DOCID: docid, RANK: rank, SCORE: score}
                for rank, ((docno, docid), score) in enumerate(zip(ids[ordering], scores[ordering]))
            ])
        # Store results in data frame
        results: pd.DataFrame = pd.DataFrame(results)

        return results

    def _semantic_rescore(
            self, query_ids: List[str], queries: List[str], doc_ids: List[List[Tuple[str, str]]], documents: List[List[str]]
    ) -> pd.DataFrame:
        # Create accumulator for results
        results: List[Dict] = []
        # Compute similarity between query-document pairs
        for query_id, query, ids, docs in zip(query_ids, queries, doc_ids, documents):
            # Compute new scores and new ranks
            scores: np.ndarray = self._transformer_encoder.predict([[query, doc] for doc in docs]).squeeze()
            if len(scores.shape) > 1:  # TODO fixme
                scores = scores[:, 0]
            ordering: np.ndarray = np.argsort(-scores, axis=-1)
            # Add current results to accumulator
            # ids = np.array(self._get_doc_ids(idxs))
            results.extend([
                {QID: query_id, DOCNO: docno, DOCID: docid, RANK: rank, SCORE: score}
                for rank, ((docno, docid), score) in enumerate(zip([ids[i] for i in ordering], scores[ordering]))
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
        scores = self._transformer_encoder.predict(list([q, d] for q, d in zip(queries, docs))).squeeze()
        if len(scores.shape) > 1:  # TODO fixme
            scores = scores[:, 0]

        return scores
