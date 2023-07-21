"""
Core interval join operations on arrays of interval coordinates.
"""
from __future__ import annotations
from typing import Any, Tuple, Union, Optional

import numpy as np

from bioframe_lite._compat.typing import Literal


__all__ = [
    "overlap_self",
    "overlap",
    "within_self",
    "within",
    "cluster",
    "closest_self",
    "closest",
]


def aranges_flat(
    starts: Union[int, np.ndarray],
    stops: Optional[np.ndarray] = None,
    *,
    lengths: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Expand multiple integer ranges and return their concatenation.

    Parameters
    ----------
    starts : int or array (n,)
        Starts for each range
    stops : array (n,)
        Stops for each range
    lengths : array (n,)
        Lengths for each range. Either stops or lengths must be provided.

    Returns
    -------
    array (m,)
        Concatenated ranges.

    Examples
    --------
    >>> starts = np.array([1, 3, 4, 6])
    >>> stops = np.array([1, 5, 7, 6])
    >>> print(aranges_flat(starts, stops))  # [] + [3, 4] + [4, 5, 6] + []
    # [3 4 4 5 6]
    """
    if (stops is None) == (lengths is None):
        raise ValueError("Either stops or lengths must be provided!")

    if lengths is None:
        lengths = stops - starts

    if np.isscalar(starts):
        starts = np.full(len(stops), starts)

    # Repeat each start position `length` times
    runs = np.repeat(starts, lengths)

    # Add a stairstep vector that resets to 0 after each run
    stairs = (
        np.arange(lengths.sum())
        - np.repeat(lengths.cumsum() - lengths, lengths)
    )

    return runs + stairs


def _overlap_right(
    ids1: np.ndarray,
    ids2: np.ndarray,
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    strict: bool = False,
    closed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given two ordered sets of intervals, both sorted by (start, end), find all
    overlaps between intervals in set1 with downstream intervals in set2.

    Parameters
    ----------
    ids1, ids2 : numpy.ndarray
        Interval IDs, sorted.

    starts1, ends1, starts2 : numpy.ndarray
        Interval coordinates, sorted.

    closed : bool, optional [default=False]
        If True, treat intervals as closed on the right as well as the left,
        allowing overlap of bookended intervals.

    Returns
    -------
    events1, events2 : numpy.ndarray
        Pairs of indices of overlapping intervals.

    Notes
    -----
    This function returns all pairs (a, b) where a in set1 overlaps with b
    in set2 and b's start does not lie to the left of a.
    """
    # The range of the insertion positions between an interval's start and
    # end in the other set's `starts` array corresponds to overlaps of that
    # interval with the other set's intervals. This search process will match
    # each query interval with all overlapping intervals in the other set
    # that begin **downstream of the query interval's start position**.
    lo1_in_starts2 = np.searchsorted(starts2, starts1, "right" if strict else "left")
    hi1_in_starts2 = np.searchsorted(starts2, ends1, "right" if closed else "left")

    # The intervals with positive insertion ranges in the other set are the
    # ones that have overlaps with the other set, so we filter for those to
    # find the matches.
    has_overlaps = lo1_in_starts2 < hi1_in_starts2
    ids1_matched = ids1[has_overlaps]
    lo1_in_starts2 = lo1_in_starts2[has_overlaps]
    hi1_in_starts2 = hi1_in_starts2[has_overlaps]

    # Collect the IDs of pairs of overlapping intervals
    # Repeat each ID in set 1 over the ranges matched in set 2
    events1 = np.repeat(ids1_matched, hi1_in_starts2 - lo1_in_starts2)
    # Ranges of IDs in set 2 matched by intervals in set 1
    events2 = ids2[aranges_flat(lo1_in_starts2, hi1_in_starts2)]

    return events1, events2


def overlap_self(
    starts: np.ndarray,
    ends: np.ndarray,
    closed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of pairs of overlapping intervals within a set.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Interval coordinates.

    closed : bool, optional [default=False]
        If True, treat intervals as closed on the right as well as the left,
        allowing overlap of bookended intervals.

    Returns
    -------
    events1, events2 : numpy.ndarray
        Join events, i.e. the indices of all pairs of overlapping intervals.

    Notes
    -----
    Because overlap is commutative, when applied naively from one set onto
    itself, we will always obtain both (a1, a2) and (a2, a1) if the two
    intervals overlap.

    Accordingly, we apply two exclusion criteria for an overlap self-join:

    * We return only unique pairs of overlapping intervals (a, b), where b's
      start does not lie upstream of a's.

    * We exclude all (a, a), i.e. we do not consider an interval a to overlap
      with itself.
    """
    n = len(starts)
    if n == 0:
        return np.array([], dtype=int)

    # Sort the intervals by (start, end)
    ids = np.lexsort([ends, starts])
    starts, ends = starts[ids], ends[ids]

    # Find overlaps between the set and itself in one direction.
    events1, events2 = _overlap_right(
        ids, ids, starts, ends, starts, closed=closed
    )

    # Remove self-matches.
    mask = events1 != events2
    events1 = events1[mask]
    events2 = events2[mask]

    return events1, events2


def overlap(
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    ends2: np.ndarray,
    closed: bool = False,
    sort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of pairs of overlapping intervals between two sets.

    Parameters
    ----------
    starts1, ends1, starts2, ends2 : numpy.ndarray
        Interval coordinates.

    closed : bool, optional [default=False]
        If True, treat intervals as closed on the right as well as the left,
        allowing overlap of bookended intervals.

    Returns
    -------
    events1, events2 : numpy.ndarray
        Join events, i.e. the indices of all pairs of overlapping intervals.
    """
    n1 = len(starts1)
    n2 = len(starts2)
    if n1 == 0 or n2 == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Sort both interval lists and their IDs by (start, end)
    ids1 = np.lexsort([ends1, starts1])
    ids2 = np.lexsort([ends2, starts2])
    starts1, ends1 = starts1[ids1], ends1[ids1]
    starts2, ends2 = starts2[ids2], ends2[ids2]

    # Find overlaps from set1 to intervals downstream in set2.
    ev_right1, ev_right2 = _overlap_right(
        ids1, ids2, starts1, ends1, starts2, closed=closed
    )
    # Find overlaps from set2 to intervals downstream in set1, corresponding
    # to overlaps from set1 to intervals upstream in set2.
    ev_left2, ev_left1 = _overlap_right(
        ids2, ids1, starts2, ends2, starts1, strict=True, closed=closed
    )

    events1 = np.r_[ev_right1, ev_left1]
    events2 = np.r_[ev_right2, ev_left2]

    if sort:
        idx = np.lexsort([events2, events1])
        events1 = events1[idx]
        events2 = events2[idx]

    return events1, events2


def within_self(
    starts: np.ndarray,
    ends: np.ndarray,
    radius: int | tuple[int, int] = 0,
    closed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of pairs of in proximity between two sets.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Interval coordinates.

    radius : int, ndarray, or pair of either [default=0]
        Search radii. If a pair, the first value is the radius to the
        left and the second value is the radius to the right. If a single
        value, the radius is applied symmetrically left and right.

    closed : bool, optional [default=True]
        If True, treat the search radius as closed-ended, allowing
        capture of intervals that abut the search interval.

    Returns
    -------
    events1, events2 : numpy.ndarray
        Join events, i.e. the indices of all pairs of overlapping intervals.
    """
    # Since both query and target intervals extend a search window, we cut it in
    # half to get the correct maximum separation distance. It doesn't matter
    # which way we extend the search window in this case as long as the total
    # amount is half the desired distance, so we just extend the ends.

    if isinstance(radius, tuple):
        left, right = radius
    else:
        left = right = radius

    return overlap_self(
        starts,
        ends + (left + right) / 2,
        closed,
    )


def within(
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    ends2: np.ndarray,
    radius: int | tuple[int, int] = 0,
    closed: bool = True,
    sort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of pairs of in proximity between two sets.

    Parameters
    ----------
    starts1, ends1, starts2, ends2 : numpy.ndarray
        Interval coordinates.

    radius : int, ndarray, or pair of either [default=0]
        Search radii. If a pair, the first value is the radius to the
        left and the second value is the radius to the right. If a single
        value, the radius is applied symmetrically left and right.

    closed : bool, optional [default=True]
        If True, treat the search radius as closed-ended, allowing
        capture of intervals that abut the search interval.

    Returns
    -------
    events1, events2 : numpy.ndarray
        Join events, i.e. the indices of all pairs of overlapping intervals.
    """

    if isinstance(radius, tuple):
        left, right = radius
    else:
        left = right = radius

    return overlap(
        starts1 - left,
        ends1 + right,
        starts2,
        ends2,
        closed,
        sort
    )


def cluster(
    starts: np.ndarray,
    ends: np.ndarray,
    closed: bool = True,
) -> np.ndarray:
    """
    Cluster overlapping or abutting intervals.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Interval coordinates.

    closed : bool, optional [default=True]
        If True, treat ends as closed to cluster bookended intervals.

    Returns
    -------
    cluster_ids : numpy.ndarray
    """
    # Sort intervals by start then end
    order = np.lexsort([ends, starts])
    starts, ends = starts[order], ends[order]
    n = starts.shape[0]

    # Extend ends by running max
    ends = np.maximum.accumulate(ends)

    # Find borders of interval clusters and assign cluster ids.
    # Create a border if the start of an interval exceeds the end of the
    # previous + distance.
    if closed:
        # merge bookended intervals / merge up to distance, inclusive
        is_border = np.r_[True, starts[1:] > ends[:-1], False]
    else:
        # separate bookended intervals / merge up to distance, exclusive
        is_border = np.r_[True, starts[1:] >= ends[:-1], False]

    # Assign cluster IDs to intervals
    cluster_ids = np.full(n, -1)
    border_sums = np.cumsum(is_border)[:-1]
    cluster_ids[order] = border_sums - 1
    n_clusters = border_sums[-1]

    return cluster_ids, n_clusters


def _closest_nooverlap_left(
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    ends2: np.ndarray,
    k: int = 1,
    tie_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n1 = starts1.shape[0]

    if tie_arr is None:
        ids2_endsorted = np.argsort(ends2)
    else:
        ids2_endsorted = np.lexsort([-tie_arr, ends2])

    closest_hi = np.searchsorted(ends2[ids2_endsorted], starts1, "right")
    closest_lo = np.maximum(closest_hi - k, 0)

    ids1 = np.repeat(np.arange(n1), closest_hi - closest_lo)
    ids2 = ids2_endsorted[aranges_flat(closest_lo, closest_hi)]

    return ids1, ids2


def _closest_nooverlap_right(
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    ends2: np.ndarray,
    k: int = 1,
    tie_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n1 = starts1.shape[0]
    n2 = starts2.shape[0]

    if tie_arr is None:
        ids2_startsorted = np.argsort(starts2)
    else:
        ids2_startsorted = np.lexsort([tie_arr, starts2])

    closest_lo = np.searchsorted(starts2[ids2_startsorted], ends1, "left")
    closest_hi = np.minimum(closest_lo + k, n2)

    ids1 = np.repeat(np.arange(n1), closest_hi - closest_lo)
    ids2 = ids2_startsorted[aranges_flat(closest_lo, closest_hi)]

    return ids1, ids2


def _prune_closest(
    ids1: np.ndarray,
    ids2: np.ndarray,
    dists: np.ndarray,
    k: int,
    tie_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort by distance to set 1 intervals and, if present, by the tie-breaking
    # data array.
    if tie_arr is None:
        order = np.lexsort([ids2, dists, ids1])
    else:
        order = np.lexsort([ids2, tie_arr, dists, ids1])
    ids1 = ids1[order]
    ids2 = ids2[order]
    dists = dists[order]

    # For each set 1 interval, select up to k closest neighbours.
    run_borders = np.where(np.r_[True, ids1[:-1] != ids1[1:], True])[0]
    run_starts = run_borders[:-1]
    run_ends = run_borders[1:]
    idx = aranges_flat(run_starts, lengths=np.minimum(k, run_ends - run_starts))
    ids1 = ids1[idx]
    ids2 = ids2[idx]
    dists = dists[idx]

    return ids1, ids2, dists


def closest_self(
    starts: np.ndarray,
    ends: np.ndarray,
    k: int = 1,
    include_overlaps: bool = False,
    direction: Literal["left", "right"] = None,
    tie_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every interval in a set, return the indices of k closest intervals
    in the same set.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Interval coordinates.

    k : int
        The number of neighbors to report.

    direction : numpy.ndarray with dtype bool or None
        Strand vector to define the upstream/downstream orientation of the
        intervals.

    tie_arr : numpy.ndarray or None
        Extra data describing intervals in set 2 to break ties when multiple
        intervals are located at the same distance. Intervals with *lower*
        tie_arr values will be given priority.

    Returns
    -------
    ids1, ids2, dists : numpy.ndarray
        Indices of the closest intervals in for each interval in the set.
        Distances between the intervals.

    """
    # Non-overlapping closest is not commutative. Unlike overlap, we treat
    # pairs (a, b) and (b, a) as distinct events. We can therefore reuse
    # the closest_nonoverlap search functions, but we still need to special
    # case the overlap self-join.
    if direction is None:
        direction = "both"

    if direction not in {"left", "right", "both"}:
        raise ValueError("direction must be one of 'left', 'right', 'both'")

    ids1 = []
    ids2 = []
    dists = []

    if include_overlaps:
        ids1_over, ids2_over = overlap_self(
            starts, ends, closed=False,
        )
        ids1.append(ids1_over)
        ids2.append(ids2_over)
        dists.append(np.zeros_like(ids1_over))

    if direction in {"left", "both"}:
        ids1_left, ids2_left = _closest_nooverlap_left(
            starts, ends, starts, ends, k=k, tie_arr=tie_arr
        )
        ids1.append(ids1_left)
        ids2.append(ids2_left)
        dists.append(starts[ids1_left] - ends[ids2_left])

    if direction in {"right", "both"}:
        ids1_right, ids2_right = _closest_nooverlap_right(
            starts, ends, starts, ends, k=k, tie_arr=tie_arr
        )
        ids1.append(ids1_right)
        ids2.append(ids2_right)
        dists.append(starts[ids2_right] - ends[ids1_right])

    ids1 = np.concatenate(ids1)
    ids2 = np.concatenate(ids2)
    dists = np.concatenate(dists)

    # If searching in both directions, filter the excess nearest neighbors.
    if include_overlaps or direction == "both":
        ids1, ids2, dists = _prune_closest(
            ids1, ids2, dists, k=k, tie_arr=tie_arr
        )

    return ids1, ids2, dists


def closest(
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    ends2: np.ndarray,
    k: int = 1,
    include_overlaps: bool = False,
    direction: Literal["left", "right"] = None,
    tie_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every interval in set 1, return the indices of k closest intervals
    from set 2.

    Parameters
    ----------
    starts1, ends1, starts2, ends2 : numpy.ndarray
        Interval coordinates.

    k : int
        The number of neighbors to report.

    direction : numpy.ndarray with dtype bool or None
        Strand vector to define the upstream/downstream orientation of the
        intervals.

    tie_arr : numpy.ndarray or None
        Extra data describing intervals in set 2 to break ties when multiple
        intervals are located at the same distance. Intervals with *lower*
        tie_arr values will be given priority.

    Returns
    -------
    ids1, ids2, dists : numpy.ndarray
        Indices of the closest intervals in set 2 for each interval in set 1.
        Distances between the intervals.

    """
    if direction is None:
        direction = "both"

    if direction not in {"left", "right", "both"}:
        raise ValueError("direction must be one of 'left', 'right', 'both'")

    ids1 = []
    ids2 = []
    dists = []

    if include_overlaps:
        ids1_over, ids2_over = overlap(
            starts1, ends1, starts2, ends2, closed=False, sort=False
        )
        ids1.append(ids1_over)
        ids2.append(ids2_over)
        dists.append(np.zeros_like(ids1_over))

    if direction in {"left", "both"}:
        ids1_left, ids2_left = _closest_nooverlap_left(
            starts1, ends1, starts2, ends2, k=k, tie_arr=tie_arr
        )
        ids1.append(ids1_left)
        ids2.append(ids2_left)
        dists.append(starts1[ids1_left] - ends2[ids2_left])

    if direction in {"right", "both"}:
        ids1_right, ids2_right = _closest_nooverlap_right(
            starts1, ends1, starts2, ends2, k=k, tie_arr=tie_arr
        )
        ids1.append(ids1_right)
        ids2.append(ids2_right)
        dists.append(starts2[ids2_right] - ends1[ids1_right])

    ids1 = np.concatenate(ids1)
    ids2 = np.concatenate(ids2)
    dists = np.concatenate(dists)

    # If searching in both directions, filter the excess nearest neighbors.
    if include_overlaps or direction == "both":
        ids1, ids2, dists = _prune_closest(
            ids1, ids2, dists, k=k, tie_arr=tie_arr
        )

    return ids1, ids2, dists
