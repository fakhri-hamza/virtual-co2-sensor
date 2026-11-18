"""Split conformal prediction: turn any regressor into one with calibrated intervals.

Idea (Vovk et al.; Lei et al., 2018):
  1. fit the model on a training part,
  2. on a held-out calibration part, compute absolute errors |y - y_hat|,
  3. the interval for a new point is y_hat ± q, where q is the (1 - alpha) quantile
     of the calibration errors (with a small finite-sample correction).
If calibration and test data are exchangeable, the interval contains the truth at
least (1 - alpha) of the time. With time series that assumption can break, which is
exactly what we measure.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalRegressor:
    model: object
    alpha: float = 0.1
    q: float | None = None

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        self.model.fit(X_fit, y_fit)
        residuals = np.abs(np.asarray(y_cal) - self.model.predict(X_cal))
        n = len(residuals)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict(self, X):
        y_hat = self.model.predict(X)
        return y_hat, y_hat - self.q, y_hat + self.q


@dataclass
class MondrianConformalRegressor:
    """Group-conditional ("Mondrian") split conformal: one interval width per group.

    Plain split conformal gives every point the same width, so it is too wide where the model
    is reliable and too narrow where it is not. Calibrating separately inside each group (here:
    'night empty' / 'lights off' / 'lights on') targets (1 - alpha) coverage *within* each group,
    under the same exchangeability assumption (Vovk et al., 2005).
    """
    model: object
    alpha: float = 0.1
    q: dict | None = None

    def fit(self, X_fit, y_fit, X_cal, y_cal, groups_cal):
        self.model.fit(X_fit, y_fit)
        residuals = np.abs(np.asarray(y_cal) - self.model.predict(X_cal))
        groups_cal = np.asarray(groups_cal)
        self.q = {}
        for g in np.unique(groups_cal):
            r = residuals[groups_cal == g]
            n = len(r)
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q[str(g)] = float(np.quantile(r, level, method="higher"))
        return self

    def predict(self, X, groups):
        y_hat = self.model.predict(X)
        half = np.array([self.q[str(g)] for g in np.asarray(groups)])
        return y_hat, y_hat - half, y_hat + half


def coverage(y, lower, upper) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))
