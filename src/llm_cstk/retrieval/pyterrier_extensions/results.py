import pandas as pd

from .utils import *


__all__ = ['sort_results']


def sort_results(search_results: pd.DataFrame):
    search_results = search_results.sort_values(SCORE, ascending=False)
    search_results[RANK] = range(len(search_results))

    return search_results
