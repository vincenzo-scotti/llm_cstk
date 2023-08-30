import os

import pandas as pd
import pyterrier as pt

from .document import PTDocManager
from .ranker import _PTRanker
from .wrapper import BiEncoderPTTransformer, CrossEncoderPTTransformer
from .utils import *
from .utils import _Singleton


class SemanticPTRanker(_PTRanker, _Singleton):
    def __init__(
            self,
            ranking_model: Optional[SemanticSearch],
            reranking_model: Optional[SemanticSearch],
            *args,
            ranking_params: Optional[Dict] = None,
            reranking_params: Optional[Dict] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Scoring approaches
        self.ranking_model: Optional[SemanticSearch] = ranking_model
        self.ranking_params: Optional[Dict] = ranking_params
        self.reranking_model: Optional[SemanticSearch] = reranking_model
        self.reranking_params: Optional[Dict] = reranking_params
        # Caches
        self._data_df_cache: Dict[str, pd.DataFrame] = dict()
        self._essential_data_df_cache: Dict[str, pd.DataFrame] = dict()

    def _load_data_df(self, path: str) -> pd.DataFrame:
        if path not in self._data_df_cache:
            self._data_df_cache[path] = pd.read_csv(path, dtype=DTYPES)

        return self._data_df_cache[path]

    def _load_essential_data_df(self, path: str) -> pd.DataFrame:
        if path not in self._data_df_cache:
            self._essential_data_df_cache[path] = pd.read_csv(
                path, dtype=DTYPES_ESSENTIAL, usecols=list(DTYPES_ESSENTIAL.keys())
            )

        return self._essential_data_df_cache[path]

    def _get_model(
            self,
            model: SemanticSearch,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None,
            ranking: bool = True,
            metadata: bool = False
    ) -> pt.Transformer:
        # Get model id
        transformer: str = (self.ranking_params if ranking else self.reranking_params)[TRANSFORMER_PARAM]
        model_id: str = str(np.uint(hash(f'{model}_{transformer}_{corpus}_{chunk_doc}_{chunk_size}_{chunk_stride}')))
        # Check whether the model is not already in cache
        if model_id not in self._model_cache:
            if model == 'bienc':
                # Load data if required
                data_df = None
                if metadata or not(
                    self.index_exists(corpus, chunk_doc, chunk_size, chunk_stride, ranking=ranking) or
                    self.pre_computed_embeddings_exists(corpus, chunk_doc, chunk_size, chunk_stride, ranking=ranking)
                ):
                    data_df = self._load_data_df(self.get_corpus_path(
                        corpus,
                        chunk_doc=chunk_doc,
                        chunk_size=chunk_size,
                        chunk_stride=chunk_stride
                    ))
                elif ranking:
                    data_df = self._load_essential_data_df(self.get_corpus_path(
                        corpus,
                        chunk_doc=chunk_doc,
                        chunk_size=chunk_size,
                        chunk_stride=chunk_stride
                    ))
                # Load embeddings index if required
                ann_index_path = None
                if self.index_exists(corpus, chunk_doc, chunk_size, chunk_stride, ranking=ranking) and ranking:
                    ann_index_path = self.get_index_path(corpus, chunk_doc, chunk_size, chunk_stride, ranking=ranking)
                # Load pre-computed embeddings if required
                pre_computed_embeddings_path = None
                if (
                    self.pre_computed_embeddings_exists(
                        corpus, chunk_doc, chunk_size, chunk_stride, ranking=ranking
                    ) and (not ranking or ann_index_path is None)
                ):
                    pre_computed_embeddings_path = self.get_pre_computed_embeddings_path(
                            corpus, chunk_doc, chunk_size, chunk_stride, ranking=ranking
                    )
                # Create
                pt_transformer = BiEncoderPTTransformer(
                    data_df=data_df,
                    ann_index_path=ann_index_path,
                    pre_computed_embeddings_path=pre_computed_embeddings_path,
                    metadata=metadata,
                    **(self.ranking_params if ranking else self.reranking_params)
                )
            elif model == 'xenc':
                # Load data if required
                data_df = None
                if metadata or ranking:
                    data_df = self._load_data_df(self.get_corpus_path(
                        corpus,
                        chunk_doc=chunk_doc,
                        chunk_size=chunk_size,
                        chunk_stride=chunk_stride
                    ))
                # Create PyTerrier transformer instance
                pt_transformer = CrossEncoderPTTransformer(
                    data_df=data_df, metadata=metadata, **(self.ranking_params if ranking else self.reranking_params)
                )
            else:
                raise ValueError(
                    f"Unknown semantic search model: \'{model}\', "
                    f"accepted values are {', '.join(f'{repr(t)}' for t in SemanticSearch)}"
                )

            # Cache model
            self._model_cache[model_id] = pt_transformer

        return self._model_cache[model_id]

    def get_ranking_model(
            self,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None,
            reranking: bool = False,
            cutoff: bool = True,
            metadata: bool = False
    ) -> pt.Transformer:
        if cutoff:
            return self._get_model(
                self.ranking_model,
                corpus,
                chunk_doc,
                chunk_size,
                chunk_stride,
                metadata=metadata or reranking
            ) % (self.ranking_cutoff if not reranking else self.reranking_cutoff)
        else:
            return self._get_model(
                self.ranking_model,
                corpus,
                chunk_doc,
                chunk_size,
                chunk_stride,
                metadata=metadata or reranking
            )

    def get_reranking_model(
            self,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None,
            metadata: bool = False
    ) -> pt.Transformer:
        return self._get_model(
            self.reranking_model,
            corpus,
            chunk_doc,
            chunk_size,
            chunk_stride,
            ranking=False,
            metadata=metadata
        )

    def get_scoring_model(self, corpus: str, txt_col: str = TEXT, ranking: bool = True):
        # Get model id
        model = self.ranking_model if ranking else self.reranking_model
        transformer: str = (self.ranking_params if ranking else self.reranking_params)[TRANSFORMER_PARAM]
        model_id: str = str(np.uint(hash(f'{model}_{transformer}')))
        # Check whether the model is not already in cache
        if model_id not in self._scoring_model_cache:
            # Load index
            pt_transformer = self._get_model(model, corpus, ranking=False).to_semantic_scorer(txt_col=txt_col)
            # Cache model
            self._scoring_model_cache[model_id] = pt_transformer

        return self._scoring_model_cache[model_id]

    def get_index_path(
            self,
            corpus: str,
            chunk_doc: bool,
            chunk_size: Optional[int],
            chunk_stride: Optional[int],
            ranking: bool = True
    ) -> Optional[str]:
        # NOTE: this makes sense only with bi-encoder models
        transformer: str = (self.ranking_params if ranking else self.reranking_params)[TRANSFORMER_PARAM]
        normalised: bool = (self.ranking_params if ranking else self.reranking_params).get(NORM_PARAM, True)
        ann: Optional[ANNSearch] = (self.ranking_params if ranking else self.reranking_params).get(ANN_PARAM)
        indexing_params: Dict = (self.ranking_params if ranking else self.reranking_params).get(
            INDEXING_CONFIG_PARAM, ANN_SEARCH_INDEX_DEFAULT_PARAMETERS[ann]
        )

        return PTDocManager.get_semantic_index_path(
            self.data_dir_path,
            corpus,
            transformer,
            ann,
            indexing_params,
            normalised,
            chunk_doc,
            chunk_size,
            chunk_stride
        )

    def get_pre_computed_embeddings_path(
            self,
            corpus: str,
            chunk_doc: bool,
            chunk_size: Optional[int],
            chunk_stride: Optional[int],
            ranking: bool = True
    ) -> str:
        # NOTE: this makes sense only with bi-encoder models
        transformer: str = (self.ranking_params if ranking else self.reranking_params)[TRANSFORMER_PARAM]
        normalised: bool = (self.ranking_params if ranking else self.reranking_params).get(NORM_PARAM, True)

        return PTDocManager.get_embedding_cache_path(
            self.data_dir_path, corpus, transformer, normalised, chunk_doc, chunk_size, chunk_stride
        )

    def pre_computed_embeddings_exists(self, *args, **kwargs) -> bool:
        return os.path.exists(self.get_pre_computed_embeddings_path(*args, **kwargs))