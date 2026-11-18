"""Experiments.

1. Virtual CO2 sensor: predict CO2 (ppm) from cheap sensors + time of day, and attach
   90% conformal prediction intervals. Does the interval really cover 90% on unseen days?
2. Occupancy sensor ablation: which sensor subset is enough to tell whether the room is
   occupied? Do we need the (expensive) CO2 sensor at all?
3. Physics check (v0.2): a 3-parameter CO2 mass-balance model driven by "lights on".
   Can physics do what ML could not?
4. Fault watch (v0.2): group-conditional conformal intervals. A tight interval for empty
   nights should catch a ventilation / sensor fault earlier than one interval for everything.

Usage:
    python -m vsensor.experiments
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy.optimize import minimize

from vsensor import RESULTS_DIR
from vsensor.conformal import ConformalRegressor, MondrianConformalRegressor, coverage
from vsensor.data import CHEAP, TARGET_CO2, TARGET_OCC, condition_group, load

TIME = ["hour_sin", "hour_cos", "weekend"]
TRENDS = ["Temperature_d5", "Humidity_d5", "Light_d5"]
ACCUMULATION = ["lights_on_30m", "lights_on_60m", "lights_on_120m", "lights_on_240m", "Humidity_d30"]
FEATURES_CO2 = CHEAP + TIME + TRENDS + ACCUMULATION
OUTDOOR_PPM = 400


def co2_experiment(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(train) * 0.8)                  # last 20% of training time = calibration set
    fit, cal = train.iloc[:cut], train.iloc[cut:]
    models = {
        "mean (no model)": DummyRegressor(),
        "linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           random_state=0),
    }
    rows, preds = [], {}
    for name, model in models.items():
        cr = ConformalRegressor(model, alpha=0.1).fit(fit[FEATURES_CO2], fit[TARGET_CO2],
                                                      cal[FEATURES_CO2], cal[TARGET_CO2])
        y_hat, lo, hi = cr.predict(test[FEATURES_CO2])
        lo = np.maximum(lo, OUTDOOR_PPM)          # indoor CO2 cannot fall below outdoor air
        y = test[TARGET_CO2]
        occ = test[TARGET_OCC] == 1
        normal = (test["date"] < "2015-02-18").to_numpy()   # before the sensor/ventilation fault
        rows.append({
            "model": name,
            "MAE_ppm": mean_absolute_error(y, y_hat),
            "MAE_ppm_before_fault": mean_absolute_error(y[normal], y_hat[normal]),
            "RMSE_ppm": mean_squared_error(y, y_hat) ** 0.5,
            "R2": r2_score(y, y_hat),
            "interval_halfwidth_ppm": cr.q,
            "coverage_90": coverage(y, lo, hi),
            "coverage_occupied": coverage(y[occ], lo[occ], hi[occ]),
            "coverage_empty": coverage(y[~occ], lo[~occ], hi[~occ]),
        })
        preds[name] = (y_hat, lo, hi)
    best = preds["linear regression"]
    trace = pd.DataFrame({"date": test["date"], "co2": test[TARGET_CO2], "pred": best[0],
                          "lower": best[1], "upper": best[2], "occupied": test[TARGET_OCC]})
    return pd.DataFrame(rows), trace


def per_day(trace: pd.DataFrame) -> pd.DataFrame:
    """Per test day: error, interval coverage, and the empty-room CO2 level.

    The empty-room level (minimum CO2 of the day) should stay near outdoor air (~400–500 ppm).
    If it climbs while nobody is there, the physical sensor or the ventilation is faulty,
    and the virtual sensor's disagreement is the alarm.
    """
    t = trace.assign(day=trace["date"].dt.date,
                     inside=(trace["co2"] >= trace["lower"]) & (trace["co2"] <= trace["upper"]),
                     abs_err=(trace["co2"] - trace["pred"]).abs())
    out = t.groupby("day").agg(
        empty_room_co2_ppm=("co2", "min"),
        occupied_share=("occupied", "mean"),
        MAE_ppm=("abs_err", "mean"),
        coverage_90=("inside", "mean"),
    ).reset_index()
    out["flag"] = np.where(out["coverage_90"] < 0.5, "FAULT?", "ok")
    return out


SUBSETS = {
    "Light only": ["Light"],
    "Temperature + Humidity": ["Temperature", "Humidity", "HumidityRatio"],
    "CO2 only": ["CO2"],
    "All cheap sensors (no CO2)": CHEAP,
    "Cheap sensors + time features": CHEAP + TIME + TRENDS,
    "Everything incl. CO2": CHEAP + ["CO2"] + TIME + TRENDS,
}


def occupancy_experiment(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_tr, y_te = train[TARGET_OCC], test[TARGET_OCC]
    for subset, cols in SUBSETS.items():
        for model_name, model in {
            "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                                                      solver="liblinear")),
            "gradient boosting": HistGradientBoostingClassifier(random_state=0),
        }.items():
            model.fit(train[cols], y_tr)
            p = model.predict(test[cols])
            rows.append({"sensors": subset, "model": model_name,
                         "accuracy": accuracy_score(y_te, p), "f1": f1_score(y_te, p)})
    rows.append({"sensors": "Always 'empty' (no model)", "model": "-",
                 "accuracy": accuracy_score(y_te, np.zeros(len(y_te))), "f1": 0.0})
    return pd.DataFrame(rows)


def simulate_mass_balance(lights_on, new_segment, gain, decay, base):
    """CO2[t+1] = CO2[t] + gain * lights_on[t] - decay * (CO2[t] - base), one step per minute.

    gain: ppm added per minute while the room is (probably) occupied; decay: share of the excess
    over `base` removed by ventilation each minute. Restarts at `base` after a data gap.
    """
    c = np.empty(len(lights_on))
    c[0] = base
    for t in range(1, len(c)):
        prev = base if new_segment[t] else c[t - 1]
        c[t] = prev + gain * lights_on[t - 1] - decay * (prev - base)
    return c


def physics_experiment(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    def inputs(df):
        gaps = df["date"].diff().dt.total_seconds().fillna(np.inf) > 300
        return (df["Light"] > 200).to_numpy(float), gaps.to_numpy()

    on_tr, gap_tr = inputs(train)
    on_te, gap_te = inputs(test)
    y_tr = train[TARGET_CO2].to_numpy()

    def loss(p):
        gain, decay, base = p
        if not 0 < decay < 1:
            return 1e12
        return np.mean(np.abs(simulate_mass_balance(on_tr, gap_tr, gain, decay, base) - y_tr))

    res = minimize(loss, [5.0, 0.01, 450.0], method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-4, "fatol": 0.01})
    gain, decay, base = res.x
    pred = simulate_mass_balance(on_te, gap_te, gain, decay, base)
    normal = (test["date"] < "2015-02-16").to_numpy()   # before the night-time CO2 drift
    y = test[TARGET_CO2].to_numpy()
    return pd.DataFrame([{
        "gain_ppm_per_min": gain, "ventilation_time_constant_min": 1 / decay, "base_ppm": base,
        "MAE_train": res.fun,
        "MAE_test_before_drift": mean_absolute_error(y[normal], pred[normal]),
        "MAE_mean_baseline_before_drift": mean_absolute_error(y[normal], np.full(normal.sum(), y_tr.mean())),
    }])


def fault_watch(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per test night (00:00-06:00, lights off): is the empty-room CO2 inside the 90% interval?

    Compares one global interval (split conformal) with group-conditional intervals
    (Mondrian conformal). Same linear model and calibration data for both.
    A night is flagged when fewer than half of its minutes are inside the interval.
    """
    cut = int(len(train) * 0.8)
    fit, cal = train.iloc[:cut], train.iloc[cut:]

    def linear():
        return make_pipeline(StandardScaler(), LinearRegression())

    plain = ConformalRegressor(linear(), alpha=0.1).fit(
        fit[FEATURES_CO2], fit[TARGET_CO2], cal[FEATURES_CO2], cal[TARGET_CO2])
    mond = MondrianConformalRegressor(linear(), alpha=0.1).fit(
        fit[FEATURES_CO2], fit[TARGET_CO2], cal[FEATURES_CO2], cal[TARGET_CO2], condition_group(cal))
    groups = condition_group(test)
    y = test[TARGET_CO2].to_numpy()
    _, lo_p, hi_p = plain.predict(test[FEATURES_CO2])
    _, lo_m, hi_m = mond.predict(test[FEATURES_CO2], groups)
    night = groups == "night empty"
    t = pd.DataFrame({
        "day": test["date"].dt.date.to_numpy()[night],
        "co2": y[night],
        "in_global": ((y >= np.maximum(lo_p, OUTDOOR_PPM)) & (y <= hi_p))[night],
        "in_conditional": ((y >= np.maximum(lo_m, OUTDOOR_PPM)) & (y <= hi_m))[night],
    })
    out = t.groupby("day").agg(night_co2_median=("co2", "median"),
                               coverage_global=("in_global", "mean"),
                               coverage_conditional=("in_conditional", "mean")).reset_index()
    out["flag_global"] = np.where(out["coverage_global"] < 0.5, "FAULT?", "ok")
    out["flag_conditional"] = np.where(out["coverage_conditional"] < 0.5, "FAULT?", "ok")
    widths = {"global (all minutes)": plain.q, **mond.q}
    return out, widths


def main() -> None:
    train, test = load()
    print(f"train: {len(train):,} minutes ({train['date'].min():%d %b} to {train['date'].max():%d %b}) | "
          f"test: {len(test):,} minutes ({test['date'].min():%d %b} to {test['date'].max():%d %b})")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    co2, trace = co2_experiment(train, test)
    co2.to_csv(RESULTS_DIR / "co2_results.csv", index=False, float_format="%.3f")
    trace.to_csv(RESULTS_DIR / "co2_test_trace.csv", index=False, float_format="%.2f")
    print("\nVirtual CO2 sensor (test days, 90% conformal intervals)")
    print(co2.round(3).to_string(index=False))

    days = per_day(trace)
    days.to_csv(RESULTS_DIR / "co2_per_day.csv", index=False, float_format="%.3f")
    print("\nPer test day (linear model)")
    print(days.round(3).to_string(index=False))

    phys = physics_experiment(train, test)
    phys.to_csv(RESULTS_DIR / "physics_model.csv", index=False, float_format="%.3f")
    print("\nPhysics check: 3-parameter CO2 mass balance driven by 'lights on'")
    print(phys.round(3).to_string(index=False))

    watch, widths = fault_watch(train, test)
    watch.to_csv(RESULTS_DIR / "fault_watch.csv", index=False, float_format="%.3f")
    pd.Series(widths, name="halfwidth_ppm").rename_axis("interval").to_csv(
        RESULTS_DIR / "interval_widths.csv", float_format="%.1f")
    print("\nFault watch: empty-room nights, global vs group-conditional 90% intervals")
    print("half-widths (ppm):", {k: round(v) for k, v in widths.items()})
    print(watch.round(3).to_string(index=False))

    occ = occupancy_experiment(train, test)
    occ.to_csv(RESULTS_DIR / "occupancy_ablation.csv", index=False, float_format="%.4f")
    print("\nOccupancy detection by sensor subset")
    print(occ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
