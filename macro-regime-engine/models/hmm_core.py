import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from hmmlearn.hmm import GaussianHMM
import logging

logger = logging.getLogger(__name__)

class RegimeEngineHMM:
    """
    Core Hidden Markov Model for Macro Regime Classification.
    Implements 7 Random Restarts to avoid local minima, and aligns states logically.
    """
    def __init__(self, n_components: int = 3, n_restarts: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.best_model = None
        self.best_score = -np.inf
        
    def _score_state(self, mean_vector: np.ndarray) -> float:
        """
        Scores a hidden state based on its feature means.
        Features are [L_t, S_t]. Higher Liquidity and Steeper Spread = CALM.
        We simply sum the normalized means to create a heuristic score.
        """
        return np.sum(mean_vector)

    def fit(self, X: pd.DataFrame, ood_mask: pd.Series):
        """
        Fits the HMM on clean data, completely ignoring OOD events.
        """
        # Filter out OOD points for training to protect covariance matrix
        X_train = X[~ood_mask].values
        
        logger.info(f"Starting {self.n_restarts} random restarts for HMM fitting...")
        
        for i in range(self.n_restarts):
            # Seed varies per restart
            model = GaussianHMM(
                n_components=self.n_components, 
                covariance_type="full", 
                n_iter=200, 
                random_state=self.random_state + i,
                tol=1e-4
            )
            
            try:
                model.fit(X_train)
                score = model.score(X_train)
                logger.info(f"Restart {i+1}/{self.n_restarts} Log-Likelihood: {score:.2f}")
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
            except Exception as e:
                logger.warning(f"Restart {i+1} failed to converge: {e}")
                
        if self.best_model is None:
            raise RuntimeError("All HMM restarts failed to converge.")
            
        logger.info(f"Best HMM Log-Likelihood: {self.best_score:.2f}")
        self._align_states()
        
    def _align_states(self):
        """
        Aligns the arbitrary hidden states to semantic meanings:
        0: CALM, 1: TRANSITION, 2: STRESS
        Using the sum of feature means as the scoring heuristic.
        """
        means = self.best_model.means_
        scores = [self._score_state(m) for m in means]
        
        # sorted_indices[0] -> min score -> STRESS
        # sorted_indices[2] -> max score -> CALM
        sorted_indices = np.argsort(scores)
        mapping = {
            sorted_indices[2]: 0, # CALM
            sorted_indices[1]: 1, # TRANSITION
            sorted_indices[0]: 2  # STRESS
        }
        
        # Reorder model parameters in-place
        new_means = np.zeros_like(self.best_model.means_)
        new_covars = np.zeros_like(self.best_model.covars_)
        new_transmat = np.zeros_like(self.best_model.transmat_)
        new_startprob = np.zeros_like(self.best_model.startprob_)
        
        for old_idx, new_idx in mapping.items():
            new_means[new_idx] = self.best_model.means_[old_idx]
            new_covars[new_idx] = self.best_model.covars_[old_idx]
            new_startprob[new_idx] = self.best_model.startprob_[old_idx]
            
            for old_idx2, new_idx2 in mapping.items():
                new_transmat[new_idx, new_idx2] = self.best_model.transmat_[old_idx, old_idx2]
                
        self.best_model.means_ = new_means
        self.best_model.covars_ = new_covars
        self.best_model.transmat_ = new_transmat
        self.best_model.startprob_ = new_startprob
        logger.info("States semantically aligned: 0=CALM, 1=TRANSITION, 2=STRESS")
        
    def calculate_cluster_ood(self, X: pd.DataFrame, threshold_percentile: float = 99.9) -> pd.Series:
        """
        Rule 10: Calculates minimum Mahalanobis distance to HMM cluster centroids.
        Flags as OOD if the distance exceeds the chi-square threshold.
        """
        from scipy.stats import chi2
        means = self.best_model.means_
        covars = self.best_model.covars_
        n_features = X.shape[1]
        
        # Chi-Square threshold for Out-of-Distribution
        threshold = chi2.ppf(threshold_percentile / 100.0, df=n_features)
        
        # Precompute inverse covariances once
        inv_covars = [np.linalg.inv(covars[k]) for k in range(self.n_components)]
        
        X_arr = X.values # shape: (N, n_features)
        min_d_sq = np.full(X_arr.shape[0], np.inf)
        
        for k in range(self.n_components):
            diff = X_arr - means[k] # shape: (N, n_features)
            # Efficiently compute Mahalanobis distance for all rows:
            # d_sq = sum((diff @ inv_cov) * diff) along columns
            d_sq = np.sum((diff @ inv_covars[k]) * diff, axis=1) # shape: (N,)
            min_d_sq = np.minimum(min_d_sq, d_sq)
            
        is_cluster_ood = min_d_sq > threshold
        return pd.Series(is_cluster_ood, index=X.index, name="is_cluster_OOD")

    def predict_regimes(self, X: pd.DataFrame, ood_mask: pd.Series) -> pd.DataFrame:
        """
        Predicts regimes for all points. If a point is OOD (from MCD gatekeeper or HMM clusters),
        it is automatically assigned to STRESS (2) and its probability of STRESS is clamped to 1.0.
        """
        states = self.best_model.predict(X.values)
        probas = self.best_model.predict_proba(X.values)
        
        # Calculate cluster-based OOD mask
        cluster_ood = self.calculate_cluster_ood(X)
        
        # Combine MCD gatekeeper and cluster-based OOD masks
        final_ood_mask = ood_mask | cluster_ood
        
        df_res = pd.DataFrame({
            'Regime': states,
            'Prob_CALM': probas[:, 0],
            'Prob_TRANSITION': probas[:, 1],
            'Prob_STRESS': probas[:, 2],
            'is_OOD': final_ood_mask
        }, index=X.index)
        
        # Override OOD points with STRESS logic
        df_res.loc[final_ood_mask, 'Regime'] = 2
        df_res.loc[final_ood_mask, 'Prob_CALM'] = 0.0
        df_res.loc[final_ood_mask, 'Prob_TRANSITION'] = 0.0
        df_res.loc[final_ood_mask, 'Prob_STRESS'] = 1.0
        
        return df_res
