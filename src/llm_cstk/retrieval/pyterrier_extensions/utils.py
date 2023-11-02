import numpy as np
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
import hnswlib

from llm_cstk.utils.common import *
from llm_cstk.utils.common import _Singleton


# Types

Scoring: TypeAlias = Literal['semantic', 'lexical']
DocAggregation: TypeAlias = Literal['max', 'mean']
QueryAggregation: TypeAlias = Literal['max', 'mean']

LexicalSearch: TypeAlias = Literal['tf', 'tf-idf', 'bm25']
SemanticSearch: TypeAlias = Literal['bienc', 'xenc']

ANNSearch: TypeAlias = Literal['annoy', 'faiss', 'hnswlib']
ANNIndex: TypeAlias = Union[hnswlib.Index]
EmbeddingVector: TypeAlias = Union[np.ndarray, torch.tensor]
EmbeddingsCache: TypeAlias = Dict[str, Union[str, EmbeddingVector]]

# Constants

QID: str = 'qid'
OLDQID: str = 'oldqid'
TMPQID: str = 'tmpqid'
QUERY: str = 'query'
OLDQUERY: str = 'oldquery'
TMPQUERY: str = 'tmpquery'
DOCNO: str = 'docno'
DOCID: str = 'docid'
TEXT: str = 'text'
RANK: str = 'rank'
SCORE: str = 'score'

TITLE: str = 'title'
BODY: str = 'body'

METADATA: List[str] = [TEXT, TITLE, BODY]

DTYPES: Dict[str, Type] = {
    DOCNO: str,
    DOCID: int,
    TEXT: str,
    TITLE: str,
    BODY: str,
}
DTYPES_ESSENTIAL: Dict[str, Type] = {
    DOCNO: str,
    DOCID: int
}

SEP: str = '%'
QID_SEP: str = SEP
DOCID_SEP: str = SEP

SNIPPET_SEP: str = '... '

# Directories

RETRIEVER: str = 'retriever'
DOCS: str = 'docs'
INDEX: str = 'index'

# Files

DATA_FILE_NAME: str = 'data'
CHUNKED_DATA_FILE_NAME: str = 'data_chunked'
TERRIER_INDEX_PROPERTIES_FILE: str = 'data.properties'
EMBEDDINGS_FILE_NAME: str = 'embeddings'

CHUNK_AFFIX: str = 'chunked'
WIN_SIZE_AFFIX: str = 'win'
STRIDE_SIZE_AFFIX: str = 'stride'
CACHE_AFFIX: str = 'cache'
HNSWLIB_AFFIX: str = 'hnswlib'

ANN_SEARCH_INDEX_AFFIX_MAPPING: Dict[ANNSearch, str] = {
    'hnswlib': HNSWLIB_AFFIX
}

RAW_DATA_EXT: str = 'csv'
EMBEDDING_CACHE_EXT: str = 'pkl'
EMBEDDING_CACHE_COMPRESSED_EXT: str = 'pbz2'
EMBEDDING_HNSWLIB_INDEX_EXT: str = 'index'

ANN_SEARCH_INDEX_EXT_MAPPING: Dict[ANNSearch, str] = {
    'hnswlib': EMBEDDING_HNSWLIB_INDEX_EXT
}

# Parameters

LEXICAL_SEARCH_MAPPING: Dict[LexicalSearch, str] = {
    'tf': 'Tf', 'tf-idf': 'TF_IDF', 'bm25': 'BM25'
}
SEMANTIC_SEARCH_MAPPING: Dict[SemanticSearch, Type[SentenceTransformer | CrossEncoder]] = {
    'bienc': SentenceTransformer, 'xenc': CrossEncoder
}

EMBEDDINGS_ID: str = 'embeddings'

DOT_PROD_SPACE: str = 'ip'
COS_SIM_SPACE: str = 'cosine'

TRANSFORMER_PARAM: str = 'transformer'
NORM_PARAM: str = 'normalise'
NORM: str = 'normalised'
ANN_PARAM: str = 'ann'
INDEXING_CONFIG_PARAM: str = 'indexing_params'

ANN_SEARCH_INDEX_DEFAULT_PARAMETERS: Dict[ANNSearch, Dict] = {
    'hnswlib': {'ef_construction': 400, 'M': 64}
}
