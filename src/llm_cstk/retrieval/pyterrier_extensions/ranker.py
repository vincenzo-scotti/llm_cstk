import os

from typing import Optional

from .document import PTDocManager
from .utils import *


class _PTRanker:
    def __init__(self, data_dir_path: str, ranking_cutoff: int = 1000, reranking_cutoff: int = 64):
        #
        self.data_dir_path: str = data_dir_path
        self.ranking_model: Optional = None
        self.reranking_model: Optional = None
        #
        self.ranking_cutoff: int = ranking_cutoff
        self.reranking_cutoff: int = reranking_cutoff
        #
        self._model_cache: Dict[str, pt.Transformer] = dict()
        self._scoring_model_cache: Dict[str, pt.Transformer] = dict()

    def get_ranking_model(self, *args, **kwargs) -> pt.Transformer:
        raise NotImplementedError()

    def get_reranking_model(self, *args, **kwargs) -> pt.Transformer:
        raise NotImplementedError()

    def get_scoring_model(self, *args, **kwargs) -> pt.Transformer:
        raise NotImplementedError()

    def get_corpus_dir_path(self, corpus: str):
        return PTDocManager.get_corpus_path(self.data_dir_path, corpus)

    def get_corpus_path(
            self,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None
    ) -> str:
        return PTDocManager.get_corpus_path(
            self.data_dir_path, corpus, chunk_doc=chunk_doc, chunk_size=chunk_size, chunk_stride=chunk_stride
        )

    def get_index_dir_path(
            self,
            corpus: str,
            chunk_doc: bool = False,
            chunk_size: Optional[int] = None,
            chunk_stride: Optional[int] = None
    ) -> str:
        #
        index_dir: str = DATA_FILE_NAME
        if chunk_doc:
            index_dir = f'{index_dir}_{CHUNK_AFFIX}'
        if chunk_size is not None and chunk_stride is not None:
            index_dir = f'{index_dir}_{WIN_SIZE_AFFIX}_{chunk_size}_{STRIDE_SIZE_AFFIX}_{chunk_stride}'
        #
        path: str = os.path.join(self.data_dir_path, corpus, INDEX, index_dir)

        return path

    def get_index_path(self, *args, **kwargs) -> str:
        raise NotImplementedError()

    def index_exists(self, *args, **kwargs) -> bool:
        return os.path.exists(self.get_index_path(*args, **kwargs))