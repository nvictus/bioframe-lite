"""
Generic interval join operator.
"""
from typing import Any, Callable, Literal, Union, Optional

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import coo_array, csr_array, connected_components


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
    op: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    op_self: Optional[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]

    def __init__(self, op, op_self=None):
        self.op = op
        self.op_self = op_self

    def inner(self, df1, df2, by=None, **kwargs):
        starts1 = df1["start"].to_numpy()
        ends1 = df1["end"].to_numpy()
        starts2 = df2["start"].to_numpy()
        ends2 = df2["end"].to_numpy()

        keys1 = ["chrom"]
        keys2 = ["chrom"]
        if by is not None:
            if isinstance(by, str):
                by = [by]
            keys1 += by
            keys2 += by
        df1_groups = df1.groupby(keys1, observed=True, dropna=False, sort=False).indices
        df2_groups = df2.groupby(keys2, observed=True, dropna=False, sort=False).indices

        events1 = []
        events2 = []
        group_keys = set.union(set(df1_groups.keys()), set(df2_groups.keys()))
        for key in group_keys:
            df1_inds = df1_groups.get(key, np.array([]))
            df2_inds = df2_groups.get(key, np.array([]))

            if len(df1_inds) > 0 and len(df2_inds) > 0:
                ev1, ev2, *_ = self.op(
                    starts1[df1_inds],
                    ends1[df1_inds],
                    starts2[df2_inds],
                    ends2[df2_inds],
                    **kwargs
                )
                events1.append(df1_inds[ev1])
                events2.append(df2_inds[ev2])

        events1 = np.concatenate(events1)
        events2 = np.concatenate(events2)

        return events1, events2

    def inner_self(self, df, by=None, **kwargs):
        if self.op_self is None:
            raise ValueError("Self-join not supported for this operation.")

        starts = df["start"].to_numpy()
        ends = df["end"].to_numpy()

        keys = ["chrom"]
        if by is not None:
            if isinstance(by, str):
                by = [by]
            keys += by
        groups = df.groupby(keys, observed=True, dropna=False, sort=False).indices

        events1 = []
        events2 = []
        group_keys = set(groups.keys())
        for key in group_keys:
            inds = groups.get(key, np.array([]))

            if len(inds) > 0:
                ev1, ev2, *_ = self.op_self(
                    starts[inds],
                    ends[inds],
                    **kwargs
                )
                events1.append(inds[ev1])
                events2.append(inds[ev2])

        events1 = np.concatenate(events1)
        events2 = np.concatenate(events2)

        return events1, events2

    def _minus(self, inds1, inds2, n1, n2, how="left"):
        if how in {"left", "outer"} and n1 > 0:
            inds1_unpaired = np.setdiff1d(np.arange(n1), inds1)
        else:
            inds1_unpaired = np.array([], dtype=int)

        if how in {"right", "outer"} and n2 > 0:
            inds2_unpaired = np.setdiff1d(np.arange(n2), inds2)
        else:
            inds2_unpaired = np.array([], dtype=int)

        return inds1_unpaired, inds2_unpaired

    def join(self, df1, df2=None, by=None, how="inner", suffixes=("", "_"), **kwargs):

        if df2 is None:
            df2 = df1
            i, j, *_ = self.inner_self(df1, by=by, **kwargs)
        else:
            i, j, *_ = self.inner(df1, df2, by=by, **kwargs)

        oi, oj = self._minus(i, j, df1.shape[0], df2.shape[0], how=how)

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

    def diff(self, df1, df2, by=None, how="left", suffixes=("", "_"), **kwargs):

        if df2 is None:
            i, j = self.inner_self(df1, by=by, **kwargs)
        else:
            i, j = self.inner(df1, df2, by=by, **kwargs)

        oi, oj = self._minus(i, j, df1.shape[0], df2.shape[0], how=how)

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

    def graph(self, df1, df2=None, by=None, kind="biadj", **kwargs):
        if df2 is None:
            df2 = df1
            i, j = self.inner_self(df1, by=by, **kwargs)
        else:
            i, j = self.inner(df1, df2, by=by, **kwargs)

        if kind == "biadj":
            return join_graph_biadj(i, j, df1.shape[0], df2.shape[0])
        elif kind == "adj":
            return join_graph_adj(i, j, df1.shape[0], df2.shape[0])
        else:
            raise ValueError(f"Unrecognized kind: {kind}")

    def cluster(self, df, by=None, **kwargs):
        mat = self.graph(df, by=by, **kwargs)
        return connected_components(mat, directed=False, return_labels=True)
