import pytest

from src.submission_policy import (opportunity_label, rbp_opportunity,
                                   submission)


def test_no_crowd_passthrough():
    assert submission(0.62) == pytest.approx(0.62)


def test_neutral_submits_our_number_not_crowd():
    """The crowd is NOT truth — neutral submits OUR estimate, no shrink toward crowd."""
    assert submission(0.70, crowd=0.50, position="neutral") == pytest.approx(0.70)
    assert submission(0.30, crowd=0.55, position="neutral") == pytest.approx(0.30)


def test_leading_hugs_crowd_defensively():
    # leading: shrink toward crowd to deny chasers variance
    f = submission(0.70, crowd=0.50, position="leading")
    assert 0.50 < f < 0.70


def test_trailing_extremizes_away_from_crowd():
    f = submission(0.70, crowd=0.50, position="trailing")
    assert f > 0.70                       # push further from crowd for RBP


def test_caps_bind():
    assert submission(0.999, crowd=0.50, position="desperate") <= 0.97
    assert submission(0.001, crowd=0.50, position="desperate") >= 0.03


def test_rbp_opportunity_is_signed_divergence():
    assert rbp_opportunity(0.70, 0.50) == pytest.approx(0.20)
    assert rbp_opportunity(0.30, 0.55) == pytest.approx(-0.25)
    assert rbp_opportunity(0.50, None) is None


def test_opportunity_labels():
    assert "consensus" in opportunity_label(0.51, 0.50)
    assert "ABOVE" in opportunity_label(0.70, 0.50)
    assert "BELOW" in opportunity_label(0.30, 0.55)
    assert "strong" in opportunity_label(0.70, 0.50)
