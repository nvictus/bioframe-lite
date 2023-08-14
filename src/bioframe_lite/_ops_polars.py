from __future__ import annotations
from typing import Literal
import polars as pl


def prepare(
    df1: pl.LazyFrame,
    df2: pl.LazyFrame,
    cols1=("start", "end"),
    cols2=("start", "end"),
    by=None,
    how="inner",
    suffix="_",
) -> pl.LazyFrame:
    """
    Prepare two lazy dataframes for overlap join.

    Sort each dataframe by starts and ends, implode them into lists by
    chromosome, then join the coordinate lists on the chrom column.
    """
    df1_sorted = df1.sort(cols1).rename({cols1[0]: "start1", cols1[1]: "end1"})
    df2_sorted = df2.sort(cols2).rename({cols2[0]: "start2", cols2[1]: "end2"})

    if by is None:
        left = df1_sorted.select(pl.all().implode())
        right = df2_sorted.select(pl.all().implode())
        left_right = left.join(right, how="cross", suffix=suffix)
    else:
        left = df1_sorted.groupby(by).all()
        right = df2_sorted.groupby(by).all()
        left_right = left.join(right, how=how, on=by, suffix=suffix)

    return left_right


def int_ranges_flat(starts: pl.Expr, ends: pl.Expr) -> pl.Expr:
    return pl.int_ranges(start=starts, end=ends).explode()


def _overlap_right_expr(
    starts1: pl.Expr,
    ends1: pl.Expr,
    starts2: pl.Expr,
    strict: bool = False,
    closed: bool = False
) -> tuple[pl.Expr, pl.Expr]:
    ids1 = pl.int_ranges(pl.lit(0), starts1.list.lengths()).explode()
    ids2 = pl.int_ranges(pl.lit(0), starts2.list.lengths()).explode()
    starts1 = starts1.explode()
    ends1 = ends1.explode()
    starts2 = starts2.explode()

    lo1_in_starts2 = starts2.search_sorted(
        starts1, side="right" if strict else "left"
    )
    hi1_in_starts2 = starts2.search_sorted(
        ends1, side="right" if closed else "left"
    )
    has_overlap = lo1_in_starts2 < hi1_in_starts2
    ids1_matched = ids1.filter(has_overlap)
    lo1_in_starts2 = lo1_in_starts2.filter(has_overlap)
    hi1_in_starts2 = hi1_in_starts2.filter(has_overlap)

    events1 = (
        ids1_matched
        .repeat_by(hi1_in_starts2 - lo1_in_starts2)
        .explode()
    ).alias("id1")

    events2 = (
        ids2
        .take(int_ranges_flat(lo1_in_starts2, hi1_in_starts2))
        .drop_nulls()
    ).alias("id2")

    return events1, events2


def overlap_self_expr(
    starts1: pl.Expr,
    ends1: pl.Expr,
    closed: bool = False
) -> tuple[pl.Expr, pl.Expr]:
    # Find overlaps between the set and itself in one direction.
    events1, events2 = _overlap_right_expr(starts1, ends1, starts1, closed=closed)

    # Remove self-matches.
    mask = events1 != events2
    events1 = events1.filter(mask)
    events2 = events2.filter(mask)

    return events1, events2


def overlap_expr(
    starts1: pl.Expr,
    ends1: pl.Expr,
    starts2: pl.Expr,
    ends2: pl.Expr,
    closed: bool = False,
) -> tuple[pl.Expr, pl.Expr]:

    ev_right1, ev_right2 = _overlap_right_expr(
        starts1, ends1, starts2, closed=closed
    )
    ev_left2, ev_left1 = _overlap_right_expr(
        starts2, ends2, starts1, strict=True, closed=closed
    )
    events1 = pl.concat([ev_right1, ev_left1])
    events2 = pl.concat([ev_right2, ev_left2])

    return events1, events2


# def within_self(
#     starts,
#     ends,
#     radius: int | tuple[int, int] = 0,
#     closed: bool = True,
# ) -> tuple[pl.Expr, pl.Expr]:
#     if isinstance(radius, tuple):
#         left, right = radius
#     else:
#         left = right = radius

#     return overlap_self_expr(
#         starts,
#         ends + (left + right) / 2,
#         closed,
#     )


# def within(
#     starts1,
#     ends1,
#     starts2,
#     ends2,
#     radius: int | tuple[int, int] = 0,
#     closed: bool = True,
# ) -> tuple[pl.Expr, pl.Expr]:
#     if isinstance(radius, tuple):
#         left, right = radius
#     else:
#         left = right = radius

#     return overlap_expr(
#         starts1 - left,
#         ends1 + right,
#         starts2,
#         ends2,
#         closed,
#     )


def _closest_nooverlap_left(
    starts1: pl.Expr,
    ends2: pl.Expr,
    k: int = 1,
) -> tuple[pl.Expr, pl.Expr]:
    starts1 = starts1.explode()
    ends2 = ends2.explode()
    ids2_endsorted = ends2.arg_sort()
    n1 = starts1.len()

    closest_hi = (
        ends2
        .take(ids2_endsorted)
        .search_sorted(starts1, side="left")
        .cast(pl.Int64)
    )
    closest_lo = pl.max_horizontal(closest_hi - k, pl.lit(0))

    ids1 = pl.int_range(0, n1).repeat_by(closest_hi - closest_lo).explode()
    ids2 = ids2_endsorted.take(int_ranges_flat(closest_lo, closest_hi))

    return ids1, ids2


def _closest_nooverlap_right(
    ends1: pl.Expr,
    starts2: pl.Expr,
    k: int = 1,
) -> tuple[pl.Expr, pl.Expr]:
    ends1 = ends1.explode()
    starts2 = starts2.explode()
    ids2_startsorted = starts2.arg_sort()
    n1 = ends1.len()
    n2 = starts2.len()

    closest_lo = (
        starts2
        .take(ids2_startsorted)
        .search_sorted(ends1, side="left")
        .cast(pl.Int64)
    )
    closest_hi = pl.min_horizontal(closest_lo + k, n2)

    ids1 = pl.int_range(0, n1).repeat_by(closest_hi - closest_lo).explode()
    ids2 = ids2_startsorted.take(int_ranges_flat(closest_lo, closest_hi))

    return ids1, ids2


def _prune_closest(
    ids1: pl.Expr,
    ids2: pl.Expr,
    dists: pl.Expr,
    k: int,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:

    ### FIXME: This doesn't work in a groupby context.
    # Sort by distance to set1 intervals
    order = pl.arg_sort_by([ids1, dists, ids2])
    ids1 = ids1.take(order)
    ids2 = ids2.take(order)
    dists = dists.take(order)

    # For each set1 interval, select up to k closest neighbors.
    run_borders = pl.arg_where(
        pl.concat([pl.lit(True), ids1 == ids1.shift(-1)])
    ).cast(pl.Int64)
    run_starts = run_borders.slice(0, run_borders.len() - 1)
    run_ends = run_borders.shift(-1).slice(0, run_borders.len() - 1)

    idx = pl.int_ranges(
        run_starts,
        pl.min_horizontal(run_ends, run_starts + k)
    ).explode()
    ids1 = ids1.take(idx)
    ids2 = ids2.take(idx)
    dists = dists.take(idx)

    return ids1, ids2, dists


def closest_self(
    starts: pl.Expr,
    ends: pl.Expr,
    k: int = 1,
    include_overlaps: bool = False,
    direction: Literal["left", "right"] = None,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    if direction is None:
        direction = "both"

    if direction not in {"left", "right", "both"}:
        raise ValueError("direction must be one of 'left', 'right', 'both'")

    ids1 = []
    ids2 = []
    dists = []

    if include_overlaps:
        ids1_over, ids2_over = overlap_self_expr(starts, ends, closed=False)
        dists = pl.zeros_like(ids1_over)
        ids1.append(ids1_over)
        ids2.append(ids2_over)
        dists.append(dists)

    if direction in {"left", "both"}:
        ids1_left, ids2_left = _closest_nooverlap_left(starts, ends, k=k)
        ids1.append(ids1_left)
        ids2.append(ids2_left)
        dists.append(starts.take(ids1_left) - ends.take(ids2_left))

    if direction in {"right", "both"}:
        ids1_right, ids2_right = _closest_nooverlap_right(ends, starts, k=k)
        ids1.append(ids1_right)
        ids2.append(ids2_right)
        dists.append(starts.take(ids2_right) - ends.take(ids1_right))

    ids1 = pl.concat(ids1).alias("id1")
    ids2 = pl.concat(ids2).alias("id2")
    dists = pl.concat(dists).alias("dist")

    # If searching in both directions, filter the excess nearest neighbors.
    if include_overlaps or direction == "both":
        ids1, ids2, dists = _prune_closest(ids1, ids2, dists, k=k)

    return ids1, ids2, dists


def closest(
    starts1: pl.Expr,
    ends1: pl.Expr,
    starts2: pl.Expr,
    ends2: pl.Expr,
    k: int = 1,
    include_overlaps: bool = False,
    direction: Literal["left", "right"] = None,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    if direction is None:
        direction = "both"

    if direction not in {"left", "right", "both"}:
        raise ValueError("direction must be one of 'left', 'right', 'both'")

    ids1 = []
    ids2 = []
    dists = []

    if include_overlaps:
        ids1_over, ids2_over = overlap_expr(
            starts1, ends1, starts2, ends2, closed=False, sort=False
        )
        ids1.append(ids1_over)
        ids2.append(ids2_over)
        dists.append(pl.zeros_like(ids1_over))

    if direction in {"left", "both"}:
        ids1_left, ids2_left = _closest_nooverlap_left(starts1, ends2, k=k)
        ids1.append(ids1_left)
        ids2.append(ids2_left)
        dists.append(starts1.take(ids1_left) - ends2.take(ids2_left))

    if direction in {"right", "both"}:
        ids1_right, ids2_right = _closest_nooverlap_right(ends1, starts2, k=k)
        ids1.append(ids1_right)
        ids2.append(ids2_right)
        dists.append(starts2.take(ids2_right) - ends1.take(ids1_right))

    ids1 = pl.concat(ids1).alias("id1")
    ids2 = pl.concat(ids2).alias("id2")
    dists = pl.concat(dists).alias("dist")

    # If searching in both directions, filter the excess nearest neighbors.
    if include_overlaps or direction == "both":
        ids1, ids2, dists = _prune_closest(ids1, ids2, dists, k=k)

    return ids1, ids2, dists
