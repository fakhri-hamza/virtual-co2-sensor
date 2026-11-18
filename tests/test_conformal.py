import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from vsensor.conformal import ConformalRegressor, coverage
from vsensor.data import add_time_features


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 1))
    y = 3 * X[:, 0] + rng.normal(scale=1.0, size=4000)
    cr = ConformalRegressor(LinearRegression(), alpha=0.1).fit(X[:1000], y[:1000], X[1000:2000], y[1000:2000])
    _, lo, hi = cr.predict(X[2000:])
    cov = coverage(y[2000:], lo, hi)
    assert 0.87 <= cov <= 0.93          # ~90% when data are exchangeable


def test_time_features_use_only_the_past():
    df = pd.DataFrame({
        "date": pd.date_range("2015-02-02 08:00", periods=10, freq="min"),
        "Temperature": 20.0, "Humidity": 30.0, "HumidityRatio": 0.004,
        "Light": [0, 0, 500, 500, 500, 0, 0, 0, 0, 0], "CO2": 450, "Occupancy": 0,
    })
    f = add_time_features(df)
    # at row 2 the lights just turned on: only 1 "on" minute counted so far
    assert f.loc[2, "lights_on_30m"] == 1
    assert f.loc[4, "lights_on_30m"] == 3
    # changing a future value must not change past features
    df2 = df.copy(); df2.loc[9, "Light"] = 999
    assert add_time_features(df2).loc[:8, "lights_on_30m"].equals(f.loc[:8, "lights_on_30m"])


def test_mondrian_gives_each_group_its_own_coverage_and_width():
    from vsensor.conformal import MondrianConformalRegressor
    rng = np.random.default_rng(0)
    n = 6000
    X = rng.normal(size=(n, 1))
    group = np.where(rng.random(n) < 0.5, "quiet", "noisy")
    noise = np.where(group == "quiet", 0.2, 3.0)                 # very different error levels
    y = 3 * X[:, 0] + rng.normal(size=n) * noise
    m = MondrianConformalRegressor(LinearRegression(), alpha=0.1).fit(
        X[:2000], y[:2000], X[2000:4000], y[2000:4000], group[2000:4000])
    assert m.q["quiet"] < m.q["noisy"] / 5
    _, lo, hi = m.predict(X[4000:], group[4000:])
    for g in ("quiet", "noisy"):
        sel = group[4000:] == g
        assert 0.86 <= coverage(y[4000:][sel], lo[sel], hi[sel]) <= 0.94   # ~90% inside each group


def test_mass_balance_decays_to_base_and_rises_when_occupied():
    from vsensor.experiments import simulate_mass_balance
    on = np.r_[np.ones(60), np.zeros(600)]
    c = simulate_mass_balance(on, np.zeros(660, bool), gain=10.0, decay=0.02, base=420.0)
    # closed form after 60 occupied minutes: base + gain/decay * (1 - (1 - decay)^60)
    assert abs(c[60] - (420 + 10 / 0.02 * (1 - 0.98 ** 60))) < 1e-6
    assert abs(c[-1] - 420.0) < 5                # ventilation brings it back to outdoor level
