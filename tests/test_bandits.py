import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import bandits as B


def test_oracle_is_upper_bound():
    m = B.Market()
    opt = m.oracle()
    assert (opt.max(1) >= opt[:, 2]).all()


def test_linear_ts_beats_ab_test_and_fixed():
    T = 20000
    lin = B.run(B.LinTS, T)["cum_regret"]
    ab = B.run(B.ExploreCommit, T)["cum_regret"]
    fixed = B.run(B.Fixed, T)["cum_regret"]
    assert lin < ab < fixed


def test_regret_is_nonnegative():
    r = B.run(B.Thompson, 5000)
    assert r["cum_regret"] >= 0
