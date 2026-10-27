# severity_model.py
"""Severity classifier whose decisions are corrected for the dataset's class imbalance."""
import numpy as np


class PriorCorrectedClassifier:
    """Wraps a fitted probabilistic classifier and divides each class probability by
    ``prior ** exponent`` before renormalising.

    About 84% of crashes are "Minor", so the raw argmax almost never flags Serious or
    Fatal. Dividing by the training prior moves the decision point without distorting the
    model's ranking. ``exponent=0`` leaves the wrapped model unchanged and ``exponent=1``
    gives every class an equal prior. The outputs are relative risk scores, not
    calibrated probabilities.
    """

    def __init__(self, estimator, class_priors, exponent):
        self.estimator = estimator
        self.class_priors = np.asarray(class_priors, dtype=float)
        self.exponent = float(exponent)

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X):
        scores = self.estimator.predict_proba(X) / self.class_priors ** self.exponent
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
