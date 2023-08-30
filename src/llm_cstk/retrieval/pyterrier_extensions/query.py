import pandas as pd

from .utils import *


__all__ = ['max_query_passage', 'mean_query_passage', 'min_query_passage']


def reduce_query(query_reduction: Callable) -> Callable:
    def wrapped_query_reduction(search_results: pd.DataFrame):
        search_results[QID] = search_results[QID].apply(lambda x: x[QID].split(QID_SEP, maxsplit=1)[0])
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
    search_results = search_results.merge(search_result_scores, on=QID)

    return search_results


@reduce_query
def min_query_passage(search_results: pd.DataFrame) -> pd.DataFrame:
    search_results = search_results.groupby(DOCNO, as_index=False, sort=False).apply(lambda x: x.loc[x['c'].idxmin()])

    return search_results
