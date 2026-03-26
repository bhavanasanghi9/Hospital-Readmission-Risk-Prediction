from src.inference import assign_risk_tier


def test_assign_risk_tier_high():
    thresholds = {"high": 0.65, "medium": 0.53}
    assert assign_risk_tier(0.70, thresholds) == "High"


def test_assign_risk_tier_medium():
    thresholds = {"high": 0.65, "medium": 0.53}
    assert assign_risk_tier(0.60, thresholds) == "Medium"


def test_assign_risk_tier_low():
    thresholds = {"high": 0.65, "medium": 0.53}
    assert assign_risk_tier(0.40, thresholds) == "Low"