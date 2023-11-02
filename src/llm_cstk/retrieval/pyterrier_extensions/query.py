import pandas as pd

from .utils import *


__all__ = ['expand_query', 'max_query_passage', 'mean_query_passage', 'min_query_passage']


def _prepare_query(
        oldqid: str,
        query: Union[str, List[str]],
        chunk_size: Optional[int] = None,
        chunk_stride: Optional[int] = None,
        tokeniser: Optional = None
) -> pd.DataFrame:
    # TODO rework with query expansion
    # Prepare query in correct format
    if isinstance(query, str):
        tokenised_query: List[str] = tokeniser(query)
        query = [
            ' '.join(tokenised_query[idx:idx + chunk_size])
            for idx in range(0, len(tokenised_query), chunk_stride)
        ]
    query = pd.DataFrame(
        {TMPQID: [oldqid] * len(query), QID: [f'{oldqid}%p{i + 1}' for i, q in enumerate(query)], QUERY: query}
    )

    return query


def expand_query(chunk_size: Optional[int], chunk_stride: Optional[int], tokeniser: Optional) -> Callable:
    def wrapped_query_expansion(search_results: pd.DataFrame):
        search_results = search_results.rename(columns={QID: TMPQID, QUERY: TMPQUERY})
        queries = pd.concat([
            _prepare_query(oldqid, oldquery, chunk_size=chunk_size, chunk_stride=chunk_stride, tokeniser=tokeniser)
            for _, (oldqid, oldquery) in search_results.groupby(
                TMPQID, as_index=False, sort=False
            ).first()[[TMPQID, TMPQUERY]].iterrows()
        ])
        search_results = search_results.merge(queries, how='outer', on=TMPQID)
        search_results = search_results.drop([TMPQID,  TMPQUERY], axis='columns')

        return search_results

    return wrapped_query_expansion


def reduce_query(query_reduction: Callable) -> Callable:
    def wrapped_query_reduction(search_results: pd.DataFrame):
        if OLDQID in search_results.columns:
            search_results[QID] = search_results[[OLDQID, QID]].apply(
                lambda x: x[QID].replace(x[OLDQID], x[OLDQID].split(QID_SEP, maxsplit=1)[0]), axis=1
            )
            search_results[OLDQID] = search_results[OLDQID].apply(lambda x: x.split(QID_SEP, maxsplit=1)[0])
        else:
            search_results[QID] = search_results[QID].apply(lambda x: x.split(QID_SEP, maxsplit=1)[0])
        search_results = search_results.drop(QUERY, axis='columns')
        search_results = query_reduction(search_results)

        return search_results

    return wrapped_query_reduction


@reduce_query
def max_query_passage(search_results: pd.DataFrame) -> pd.DataFrame:
    search_results = search_results.groupby(DOCNO, as_index=False, sort=False).apply(lambda x: x.loc[x[SCORE].idxmax()])

    return search_results


@reduce_query
def mean_query_passage(search_results: pd.DataFrame) -> pd.DataFrame:
    search_results_groups = search_results.groupby(DOCNO, as_index=False, sort=False)
    search_result_scores = search_results_groups[SCORE].mean()
    search_results = search_results_groups.first().drop(SCORE, axis='columns')
    search_results[SCORE] = search_result_scores[SCORE]

    return search_results


@reduce_query
def min_query_passage(search_results: pd.DataFrame) -> pd.DataFrame:
    search_results = search_results.groupby(DOCNO, as_index=False, sort=False).apply(lambda x: x.loc[x[SCORE].idxmin()])

    return search_results
