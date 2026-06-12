import pytest

from src.submission_policy import optimal_lambda, submission


def test_no_crowd_passthrough():
    assert submission(0.62) == pytest.approx(0.62)


def test_zero_error_zero_kappa_is_honest():
    assert optimal_lambda(edge=0.15, tau=0.0, kappa=0.0) == pytest.approx(1.0)


def test_estimation_error_shrinks_toward_crowd():
    # noisy model -> submission sits between crowd and model
    f = submission(0.70, crowd=0.50, family="PLAYER_MARKET", position="neutral")
    assert 0.50 < f < 0.70
    # validated family shrinks less than unvalidated one
    f_goal = submission(0.70, crowd=0.50, family="GOAL_MARKET", position="neutral")
    assert f_goal > f


def test_trailing_extremizes_leading_shrinks():
    base = submission(0.70, crowd=0.50, family="GOAL_MARKET", position="neutral")
    trail = submission(0.70, crowd=0.50, family="GOAL_MARKET", position="trailing")
    lead = submission(0.70, crowd=0.50, family="GOAL_MARKET", position="leading")
    assert trail > base > lead


def test_lambda_capped():
    # tiny edge + desperate kappa must not explode past LAMBDA_MAX
    lam = optimal_lambda(edge=0.01, tau=0.05, kappa=0.06, p_hat=0.5)
    assert lam <= 1.5


def test_submission_caps():
    assert submission(0.999, crowd=0.50, family="GOAL_MARKET",
                      position="desperate") <= 0.97
    assert submission(0.001, crowd=0.50, family="GOAL_MARKET",
                      position="desperate") >= 0.03


def test_small_edge_mostly_crowd():
    # 1-point disagreement with a noisy model: submit ~the crowd
    f = submission(0.51, crowd=0.50, family="OFFSIDE_MARKET", position="neutral")
    assert abs(f - 0.50) < 0.005
