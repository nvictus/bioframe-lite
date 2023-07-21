"""
Generic interval join operator.
"""
from __future__ import annotations
from typing import Any, Callable, Tuple, Union, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components


def join_graph_biadj(
    inds1: np.array,
    inds2: np.array,
    n1: int,
    n2: int,
) -> csr_array:
    """
    Return a sparse biadjacency matrix representing the join graph between
    two ordered sets of items.

    Parameters
    ----------
    inds1, inds2 : numpy.ndarray
        Join events: indices of items in the 1st and 2nd set.

    n1, n2 : int
        The total number of records in the 1st and 2nd set.

    Returns
    -------
    scipy.sparse.coo_array (n1, n2)
        A sparse array representing the join graph between the two sets.
    """
    nnz = len(inds1)
    return coo_array(
        (np.ones(nnz, dtype=bool), (inds1, inds2)),
        shape=(n1, n2)
    ).tocsr()


def join_graph_adj(
    inds1: np.array,
    inds2: np.array,
    n1: int,
    n2: int,
) -> csr_array:
    """
    Return a sparse full adjacency matrix representing the join graph between
    two ordered sets of items.

    Parameters
    ----------
    inds1, inds2 : numpy.ndarray
        Join events: indices of items in the 1st and 2nd set.

    n1, n2 : int
        The total number of records in the 1st and 2nd set.

    Returns
    -------
    scipy.sparse.coo_array (n1 + n2, n1 + n2)
        A sparse array representing the join graph between the two sets.
    """
    nnz = len(inds1)
    return coo_array(
        (np.ones(nnz, dtype=bool), (inds1, n1 + inds2)),
        shape=(n1 + n2, n1 + n2)
    ).tocsr()


def _minus(inds1, inds2, n1, n2, how):
    if how in {"left", "outer"} and n1 > 0:
        inds1_unpaired = np.setdiff1d(np.arange(n1), inds1)
    else:
        inds1_unpaired = np.array([], dtype=int)

    if how in {"right", "outer"} and n2 > 0:
        inds2_unpaired = np.setdiff1d(np.arange(n2), inds2)
    else:
        inds2_unpaired = np.array([], dtype=int)

    return inds1_unpaired, inds2_unpaired


class JoinOperator:
    """
    A generic join operator that can be used to expose a variety of join
    operations on genomic interval dataframes with support for:

    * grouping by arbitrary columns
    * inner, left/right/full outer and left/right minus (i.e. diff) join semantics
    * special treatment of self-joins
    * sparse graph representation of join events

    The latter allows for a generic cluster algorithm that can be used to
    cluster intervals satisfying arbitrary join criteria by finding connected
    components in the join graph.
    """
    def join(self, how="inner", suffixes=("", "_")):

        df1, df2 = self._tables()
        i, j, *_ = self._inner()
        oi, oj = _minus(i, j, df1.shape[0], df2.shape[0], how=how)

        result = [
            pd.concat([
                df1.iloc[i].reset_index().rename(columns=lambda x: x + suffixes[0]),
                df2.iloc[j].reset_index().rename(columns=lambda x: x + suffixes[1]),
            ], axis=1)
        ]

        if how in {"left", "outer"}:
            left_minus = (
                df1.iloc[oi]
                .reset_index()
                .rename(columns=lambda x: x + suffixes[0])
            )
            result.append(left_minus)

        if how in {"right", "outer"}:
            right_minus = (
                df2.iloc[oj]
                .reset_index()
                .rename(columns=lambda x: x + suffixes[1])
            )
            result.append(right_minus)

        return pd.concat(
            result,
            axis=0,
            ignore_index=True
        ).convert_dtypes()

    def diff(self, how="left", suffixes=("", "_")):

        df1, df2 = self._tables()
        i, j, *_ = self._inner()
        oi, oj = _minus(i, j, df1.shape[0], df2.shape[0], how=how)

        result = []

        if how in {"left", "outer"}:
            left_minus = (
                df1.iloc[oi]
                .reset_index()
                .rename(columns=lambda x: x + suffixes[0])
            )
            result.append(left_minus)

        if how in {"right", "outer"}:
            right_minus = (
                df2.iloc[oj]
                .reset_index()
                .rename(columns=lambda x: x + suffixes[1])
            )
            result.append(right_minus)

        return pd.concat(
            result,
            axis=0,
            ignore_index=True
        ).convert_dtypes()

    def graph(self, kind="biadj"):
        i, j, *_ = self.inner(**self.kwargs)

        if kind == "biadj":
            return join_graph_biadj(i, j, self.df1.shape[0], self.df2.shape[0])
        elif kind == "adj":
            return join_graph_adj(i, j, self.df1.shape[0], self.df2.shape[0])
        else:
            raise ValueError(f"Unrecognized kind: {kind}")

    def cluster(self):
        mat = self.graph(**self.kwargs)
        return connected_components(mat, directed=False, return_labels=True)


class BinaryJoinOperator(JoinOperator):
    op: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    df1: pd.DataFrame
    df2: pd.DataFrame

    def __init__(self, op, df1, df2, by="chrom", **kwargs):
        self.op = op
        self.df1 = df1
        self.df2 = df2
        self.grouper = by
        self.kwargs = kwargs

    def _tables(self):
        return self.df1, self.df2

    def _inner(self):
        starts1 = self.df1["start"].to_numpy()
        ends1 = self.df1["end"].to_numpy()
        starts2 = self.df2["start"].to_numpy()
        ends2 = self.df2["end"].to_numpy()

        df1_groups = self.df1.groupby(
            self.grouper, observed=True, dropna=False, sort=False
        ).indices
        df2_groups = self.df2.groupby(
            self.grouper, observed=True, dropna=False, sort=False
        ).indices
        group_keys = set.union(set(df1_groups.keys()), set(df2_groups.keys()))

        events1 = []
        events2 = []
        for key in group_keys:
            df1_inds = df1_groups.get(key, np.array([]))
            df2_inds = df2_groups.get(key, np.array([]))

            if len(df1_inds) > 0 and len(df2_inds) > 0:
                ev1, ev2, *_ = self.op(
                    starts1[df1_inds],
                    ends1[df1_inds],
                    starts2[df2_inds],
                    ends2[df2_inds],
                    **self.kwargs
                )
                events1.append(df1_inds[ev1])
                events2.append(df2_inds[ev2])

        events1 = np.concatenate(events1)
        events2 = np.concatenate(events2)

        return events1, events2


class UnaryJoinOperator(JoinOperator):
    op: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]
    df: pd.DataFrame

    def __init__(self, op, df, by="chrom", **kwargs):
        self.op = op
        self.df = df
        self.grouper = by
        self.kwargs = kwargs

    def _tables(self):
        return self.df, self.df

    def _inner(self):
        df = self.df
        starts = df["start"].to_numpy()
        ends = df["end"].to_numpy()

        groups = df.groupby(
            self.grouper, observed=True, dropna=False, sort=False
        ).indices

        events1 = []
        events2 = []
        group_keys = set(groups.keys())
        for key in group_keys:
            inds = groups.get(key, np.array([]))

            if len(inds) > 0:
                ev1, ev2, *_ = self.op(
                    starts[inds],
                    ends[inds],
                    **self.kwargs
                )
                events1.append(inds[ev1])
                events2.append(inds[ev2])

        events1 = np.concatenate(events1)
        events2 = np.concatenate(events2)

        return events1, events2
