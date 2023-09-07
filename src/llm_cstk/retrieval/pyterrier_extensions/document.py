import os.path

from shutil import rmtree

import pandas as pd
import pyterrier as pt

from .wrapper import BiEncoderPTTransformer
from .utils import *
from .utils import _Singleton


__all__ = ['PTDocManager', 'reset_text_col']


if not pt.started():
    pt.init()


def reset_text_col(df: pd.DataFrame) -> pd.DataFrame:
    tmp_df = df.copy()
    tmp_df[TEXT] = tmp_df[BODY]

    return tmp_df


class PTDocManager(_Singleton):
    def __init__(self, data_dir_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir_path: str = data_dir_path
        if not os.path.exists(self.data_dir_path):
            os.mkdir(self.data_dir_path)

    def register_corpus(
            self,
            corpus: str,
            docs: pd.DataFrame,
            docs_chunked: Optional[pd.DataFrame] = None,
            overwrite: bool = False
    ):
        # Main corpus directory
        corpus_dir_path: str = os.path.join(self.data_dir_path, corpus)
        if os.path.exists(corpus_dir_path):
            if overwrite:
                rmtree(corpus_dir_path)
            else:
                raise ValueError(f"Corpus directory at path `{corpus_dir_path}` already exists")
        else:
            os.mkdir(corpus_dir_path)
        # Corpus data directory
        corpus_data_dir_path = self.get_corpus_data_dir_path(self.data_dir_path, corpus)
        os.mkdir(corpus_data_dir_path)
        # Data file
        docs_file_path = self.get_corpus_path(self.data_dir_path, corpus)
        docs.to_csv(docs_file_path, index=False)
        if docs_chunked is not None:
            docs_file_path = self.get_corpus_path(self.data_dir_path, corpus, chunk_doc=True)
            docs_chunked.to_csv(docs_file_path, index=False)

    def register_large_corpus(self, *args, **kwargs):
        raise NotImplementedError()

    def index_corpus(
            self,
            corpus: str,
            transformers: Optional[List[Dict]] = None,
            overwrite: bool = False,
    ):
        transformers = transformers if transformers is not None else list()
        # Main corpus directory
        corpus_dir_path: str = os.path.join(self.data_dir_path, corpus)
        if not os.path.exists(corpus_dir_path):
            available_corpora = [f'`{c}`' for c in os.listdir(self.data_dir_path) if not c.startswith('.')]
            raise ValueError(
                f"Corpus `{corpus}` is not part of the available corpora, "
                f"available corpora are: {', '.join(available_corpora)}"
            )
        # Corpus data directory
        corpus_data_dir_path = self.get_corpus_data_dir_path(self.data_dir_path, corpus)
        corpus_data_file_names = [f for f in os.listdir(corpus_data_dir_path) if not f.startswith('.')]
        if not os.path.exists(corpus_data_dir_path) or len(corpus_data_file_names) == 0:
            raise ValueError(f"No data available to index for corpus `{corpus}`")
        # Corpus index directory
        corpus_index_dir_path = os.path.join(corpus_dir_path, INDEX)
        if not os.path.exists(corpus_index_dir_path):
            os.mkdir(corpus_index_dir_path)
        for file_name in corpus_data_file_names:
            name, ext = os.path.splitext(file_name)
            index_dir_path = os.path.join(corpus_index_dir_path, name)
            # Check if index already exists
            if os.path.exists(index_dir_path) and overwrite:
                rmtree(index_dir_path)
            if not os.path.exists(index_dir_path):
                os.mkdir(index_dir_path)
            else:
                raise ValueError(f"Index already exists at specified path: `{index_dir_path}`")
            data_df = pd.read_csv(
                os.path.join(corpus_data_dir_path, file_name),
                dtype=DTYPES if CHUNK_AFFIX in file_name else DTYPES  # TODO fix this with proper dtypes (add olddocno)
            )
            # Lexical index
            pt_indexer: pt.Indexer = pt.DFIndexer(index_dir_path)
            pt_indexer.index(data_df[TEXT], data_df.astype(str))
            # Semantic index
            for transformer_configs in transformers:
                ann_index_path = None
                if ANN_PARAM in transformer_configs:
                    ann_index_path = self.get_semantic_index_path(
                        self.data_dir_path,
                        corpus,
                        transformer_configs[TRANSFORMER_PARAM],
                        transformer_configs[ANN_PARAM],
                        transformer_configs.get(
                            INDEXING_CONFIG_PARAM,
                            ANN_SEARCH_INDEX_DEFAULT_PARAMETERS[transformer_configs[ANN_PARAM]]
                        ),
                        transformer_configs.get(NORM_PARAM, True),
                        CHUNK_AFFIX in name
                    )
                pre_computed_embeddings_path = self.get_embedding_cache_path(
                    self.data_dir_path,
                    corpus,
                    transformer_configs[TRANSFORMER_PARAM],
                    transformer_configs.get(NORM_PARAM, True),
                    CHUNK_AFFIX in name
                )
                pt_transformer_bienc = BiEncoderPTTransformer(
                    data_df=data_df,
                    ann_index_path=ann_index_path,
                    pre_computed_embeddings_path=pre_computed_embeddings_path,
                    **transformer_configs
                )
                pt_transformer_bienc.build_ann_index()
                pt_transformer_bienc.save_ann_index()
                pt_transformer_bienc.save_pre_computed_embeddings()

    def index_large_corpus(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_corpus_data_dir_path(cls, data_dir_path: str, corpus: str) -> str:
        return os.path.join(data_dir_path, corpus, DOCS)

    @classmethod
    def get_corpus_path(
            cls,
            data_dir_path: str,
            corpus: str,
            chunk_doc: bool = False
    ) -> str:
        file_name = f'{CHUNKED_DATA_FILE_NAME if chunk_doc else DATA_FILE_NAME }.{RAW_DATA_EXT}'
        path: str = os.path.join(cls.get_corpus_data_dir_path(data_dir_path, corpus), file_name)

        return path

    @classmethod
    def get_index_dir_path(
            cls, data_dir_path: str, corpus: str, chunk_doc: bool = False
    ) -> str:
        index_dir: str = CHUNKED_DATA_FILE_NAME if chunk_doc else DATA_FILE_NAME
        path: str = os.path.join(data_dir_path, corpus, INDEX, index_dir)

        return path

    @classmethod
    def get_lexical_index_path(cls, data_dir_path: str, corpus: str, chunk_doc: bool = False) -> str:
        return os.path.join(
            cls.get_index_dir_path(data_dir_path, corpus, chunk_doc=chunk_doc), TERRIER_INDEX_PROPERTIES_FILE
        )

    @classmethod
    def get_semantic_index_path(
            cls,
            data_dir_path: str,
            corpus: str,
            transformer: str,
            ann: ANNSearch,
            indexing_params: Dict,
            normalised: bool,
            chunk_doc: bool
    ) -> str:
        # NOTE: this makes sense only with bi-encoder models
        ann_search_index_affix: str = f"{ANN_SEARCH_INDEX_AFFIX_MAPPING[ann]}_{str(np.uint(hash(str(indexing_params))))}"
        ann_search_index_ext: str = ANN_SEARCH_INDEX_EXT_MAPPING[ann]
        if normalised:
            file_name = f'{EMBEDDINGS_FILE_NAME}_{ann_search_index_affix}_{transformer}_{NORM}.{ann_search_index_ext}'
        else:
            file_name = f'{EMBEDDINGS_FILE_NAME}_{ann_search_index_affix}_{transformer}.{ann_search_index_ext}'

        return os.path.join(
            cls.get_index_dir_path(data_dir_path, corpus, chunk_doc), file_name
        )

    @classmethod
    def get_embedding_cache_path(
            cls,
            data_dir_path: str,
            corpus: str,
            transformer: str,
            normalised: bool,
            chunk_doc: bool
    ) -> str:
        if normalised:
            file_name = f'{EMBEDDINGS_FILE_NAME}_{CACHE_AFFIX}_{transformer}_{NORM}.{EMBEDDING_CACHE_COMPRESSED_EXT}'
        else:
            file_name = f'{EMBEDDINGS_FILE_NAME}_{CACHE_AFFIX}_{transformer}.{EMBEDDING_CACHE_COMPRESSED_EXT}'

        return os.path.join(
            cls.get_index_dir_path(data_dir_path, corpus, chunk_doc), file_name
        )
