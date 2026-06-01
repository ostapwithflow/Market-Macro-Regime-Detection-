import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import logging

logger = logging.getLogger(__name__)


class OODGatekeeper:
    """
    Implements the Out-of-Distribution (OOD) Gatekeeper
    using Minimum Covariance Determinant (MCD) to calculate robust Mahalanobis distance.
    This protects the HMM from "Black Swan" fat tails.

    Production-grade improvements:
    - support_fraction=None lets sklearn pick the optimal value
    - threshold_percentile validated on init
    - NaN-safe transform_predict
    - MCD fit stability guard (min samples & convergence check)
    """

    # Minimum number of observations required to fit MCD reliably
    MIN_SAMPLES_FOR_FIT = 50

    def __init__(self, contamination: float = 0.05, threshold_percentile: float = 99.9):
        # --- Validate inputs ---
        if not (0.0 < contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination}"
            )
        if not (0.0 < threshold_percentile < 100.0):
            raise ValueError(
                f"threshold_percentile must be in (0, 100), got {threshold_percentile}"
            )

        self.contamination = contamination
        self.threshold_percentile = threshold_percentile

        # Fix #1: Let sklearn compute the optimal support_fraction.
        # Previously: support_fraction = 1 - contamination = 0.95
        # sklearn's default (None) uses (n_samples + n_features + 1) / (2 * n_samples)
        # which is statistically optimal for the MCD estimator.
        self.mcd = MinCovDet(support_fraction=None)
        self.threshold_dist: float | None = None
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame):
        """Fits the robust covariance matrix on the data.

        Raises ValueError if data has too few samples or contains NaN.
        """
        # --- Guard: NaN in training data ---
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            logger.warning(
                f"OOD Gatekeeper fit(): dropping {nan_count} NaN values from training data."
            )
            X = X.dropna()

        # --- Guard: minimum sample size ---
        n_samples, n_features = X.shape
        if n_samples < self.MIN_SAMPLES_FOR_FIT:
            raise ValueError(
                f"OOD Gatekeeper requires at least {self.MIN_SAMPLES_FOR_FIT} samples "
                f"to fit MCD, got {n_samples}."
            )

        logger.info(f"Fitting MCD Gatekeeper on data of shape {X.shape}")

        try:
            self.mcd.fit(X)
        except Exception as e:
            logger.error(f"MCD fit failed: {e}")
            raise RuntimeError(
                f"MCD failed to converge. Check data quality. Original error: {e}"
            ) from e

        # Calculate degrees of freedom (number of features)
        df = n_features

        # Mahalanobis distance squared follows a Chi-Square distribution
        self.threshold_dist = chi2.ppf(self.threshold_percentile / 100.0, df)
        self._is_fitted = True
        logger.info(
            f"OOD Threshold (Mahalanobis D^2) set to: {self.threshold_dist:.4f} "
            f"(chi2 df={df}, percentile={self.threshold_percentile})"
        )

    def transform_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculates Mahalanobis distance for each row and flags OOD instances.
        Returns a boolean Series (True = OOD).

        NaN rows are automatically flagged as OOD (safe-side default).
        """
        if not self._is_fitted or self.threshold_dist is None:
            raise ValueError("Gatekeeper must be fitted before prediction.")

        # --- Fix #3: Handle NaN in prediction data ---
        nan_mask = X.isna().any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            logger.warning(
                f"OOD transform_predict: {n_nan} rows contain NaN — flagging as OOD."
            )

        # Work only with clean rows
        X_clean = X.loc[~nan_mask]

        # Initialise result: NaN rows → OOD by default (safe side)
        is_ood = pd.Series(True, index=X.index, name="is_OOD")

        if len(X_clean) > 0:
            # mcd.mahalanobis returns the squared Mahalanobis distance
            d_squared = self.mcd.mahalanobis(X_clean)

            # Flag as OOD if distance exceeds the chi-square threshold
            is_ood.loc[X_clean.index] = d_squared > self.threshold_dist

        logger.info(
            f"Detected {is_ood.sum()} OOD events out of {len(is_ood)} observations "
            f"({n_nan} were NaN-flagged)."
        )
        return is_ood

    def get_robust_covariance(self) -> np.ndarray:
        """Returns the robust covariance matrix estimated by MCD."""
        if not self._is_fitted:
            raise ValueError("Gatekeeper must be fitted first.")
        return self.mcd.covariance_

    def get_robust_mean(self) -> np.ndarray:
        """Returns the robust location (mean) estimated by MCD."""
        if not self._is_fitted:
            raise ValueError("Gatekeeper must be fitted first.")
        return self.mcd.location_
