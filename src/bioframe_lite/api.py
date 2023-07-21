import numpy as np
import pandas as pd

from bioframe_lite._join import UnaryJoinOperator, BinaryJoinOperator
from bioframe_lite import _ops


def overlap(
    df1,
    df2=None,
    by=None,
    how="inner",
    suffixes=("", "_"),
    **kwargs
):
    keys = ["chrom"]
    if by is not None:
        if isinstance(by, str):
            by = [by]
        keys += by
    if df2 is None:
        operator = UnaryJoinOperator(_ops.overlap_self, df1, keys, **kwargs)
        return operator.join(how=how, suffixes=suffixes)
    else:
        operator = BinaryJoinOperator(_ops.overlap, df1, df2, keys, **kwargs)
        return operator.join(how=how, suffixes=suffixes)


def within(
    df1,
    df2=None,
    by=None,
    how="inner",
    suffixes=("", "_"),
    **kwargs
):
    keys = ["chrom"]
    if by is not None:
        if isinstance(by, str):
            by = [by]
        keys += by
    if df2 is None:
        operator = UnaryJoinOperator(_ops.within_self, df1, keys, **kwargs)
        return operator.join(how=how, suffixes=suffixes)
    else:
        operator = BinaryJoinOperator(_ops.within, df1, df2, keys, **kwargs)
        return operator.join(how=how, suffixes=suffixes)


def closest(
    df1,
    df2=None,
    by=None,
    how="left",
    suffixes=("", "_"),
    **kwargs
):
    keys = ["chrom"]
    if by is not None:
        if isinstance(by, str):
            by = [by]
        keys += by
    if df2 is None:
        operator = UnaryJoinOperator(_ops.closest_self, df1, keys, **kwargs)
        return operator.join(how=how, suffixes=suffixes)
    else:
        operator = BinaryJoinOperator(_ops.closest, df1, df2, keys, **kwargs)
        return operator.join(how=how, suffixes=suffixes)


def cluster(df, by=None, within=0, closed=True):
    """
    Cluster intervals by overlap or proximity.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Interval coordinates.

    within : int, optional [default=0]
        Maximum distance between intervals to be considered part of the same
        cluster.

    closed : bool, optional [default=True]
        If True, treat the search distance as closed-ended, allowing overlap
        of bookended intervals.

    Returns
    -------
    cluster_ids : numpy.ndarray
    """
    starts = df["start"].to_numpy()
    ends = df["end"].to_numpy()

    keys = ["chrom"]
    if by is not None:
        if isinstance(by, str):
            by = [by]
        keys += by
    groups = df.groupby(keys, observed=True, dropna=False, sort=False).indices

    indices = []
    labels = []
    n_clusters = 0
    group_keys = groups.keys()
    for key in group_keys:
        inds = groups.get(key, np.array([]))
        if len(inds) > 0:
            ids, n = _ops.cluster(starts[inds], ends[inds] + within, closed)
            indices.append(inds)
            labels.append(n_clusters + ids)
            n_clusters += n
    indices = np.concatenate(indices)
    labels_ = np.full(df.shape[0], -1, dtype=int)
    labels_[indices] = np.concatenate(labels)

    return pd.Series(index=df.index, data=labels_, name="cluster")
