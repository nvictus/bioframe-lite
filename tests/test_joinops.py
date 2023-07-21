import numpy as np
from numpy.testing import assert_array_equal

from bioframe_lite._ops import (
    aranges_flat,
    overlap_self,
    overlap,
    closest_self,
    closest
)

def test_aranges_flat():
    starts = np.array([1, 3, 4, 6])
    ends = np.array([1, 5, 7, 6])

    assert_array_equal(
        aranges_flat(starts, ends), np.array([3, 4, 4, 5, 6])
    )


def test_overlap_self():
    starts = np.array([1, 3, 4, 6])
    ends = np.array([1, 5, 7, 6])

    ev1, ev2 = overlap_self(starts, ends)
    assert_array_equal(ev1, np.array([1, 2]))
    assert_array_equal(ev2, np.array([2, 3]))


def test_overlap():
    starts1 = np.array([1, 3, 4, 6])
    ends1 =   np.array([2, 5, 7, 6])
    starts2 = np.array([1, 2, 4, 6])
    ends2 =   np.array([2, 5, 7, 6])

    ev1, ev2 = overlap(starts1, ends1, starts2, ends2, sort=True)
    assert_array_equal(ev1, np.array([0, 1, 1, 2, 2, 2, 3]))
    assert_array_equal(ev2, np.array([0, 1, 2, 1, 2, 3, 2]))


def test_closest_nooverlap_self():
    starts = np.array([1, 3, 4, 6])
    ends =   np.array([2, 5, 7, 6])

    ev1, ev2, dists = closest_self(starts, ends, include_overlaps=False, k=1)
    assert_array_equal(ev1, np.array([0, 1, 2, 3]))
    assert_array_equal(ev2, np.array([1, 0, 0, 3]))
    assert_array_equal(dists, np.array([1, 1, 2, 0]))

    ev1, ev2, dists = closest_self(starts, ends, include_overlaps=False, k=1, direction="left")
    assert_array_equal(ev1, np.array([1, 2, 3]))
    assert_array_equal(ev2, np.array([0, 0, 3]))
    assert_array_equal(dists, np.array([1, 2, 0]))

    ev1, ev2, dists = closest_self(starts, ends, include_overlaps=False, k=1, direction="right")
    assert_array_equal(ev1, np.array([0, 1, 3]))
    assert_array_equal(ev2, np.array([1, 3, 3]))
    assert_array_equal(dists, np.array([1, 1, 0]))


def test_closest_nooverlap():
    starts1 = np.array([1, 3, 4, 6])
    ends1 =   np.array([2, 5, 7, 6])
    starts2 = np.array([1, 2, 4, 6])
    ends2 =   np.array([2, 5, 7, 6])

    ev1, ev2, dists = closest(starts1, ends1, starts2, ends2, include_overlaps=False, k=1)
    assert_array_equal(ev1, np.array([0, 1, 2, 3]))
    assert_array_equal(ev2, np.array([1, 0, 0, 3]))
    assert_array_equal(dists, np.array([0, 1, 2, 0]))

    ev1, ev2, dists = closest(starts1, ends1, starts2, ends2, include_overlaps=False, k=1, direction="left")
    assert_array_equal(ev1, np.array([1, 2, 3]))
    assert_array_equal(ev2, np.array([0, 0, 3]))
    assert_array_equal(dists, np.array([1, 2, 0]))

    ev1, ev2, dists = closest(starts1, ends1, starts2, ends2, include_overlaps=False, k=1, direction="right")
    assert_array_equal(ev1, np.array([0, 1, 3]))
    assert_array_equal(ev2, np.array([1, 3, 3]))
    assert_array_equal(dists, np.array([0, 1, 0]))
