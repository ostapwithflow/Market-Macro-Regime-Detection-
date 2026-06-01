import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_cohens_d(mu1: float, mu2: float, var1: float, var2: float) -> float:
    """
    Calculates Cohen's d to measure effect size between two state distributions.
    """
    pooled_std = np.sqrt((var1 + var2) / 2.0)
    if pooled_std == 0:
        return 0.0
    return abs(mu1 - mu2) / pooled_std

def check_feature_separability(hmm_model, feature_names: list, threshold: float = 0.5) -> dict:
    """
    Validates if features genuinely separate the found regimes.
    Checks separability between CALM (0) vs STRESS (2).
    """
    means = hmm_model.means_
    covars = hmm_model.covars_
    
    separability_report = {}
    
    # 0 = CALM, 2 = STRESS
    for i, feature in enumerate(feature_names):
        mu_calm = means[0][i]
        mu_stress = means[2][i]
        
        # Diagonal elements of covariance matrix are the variances
        var_calm = covars[0][i][i]
        var_stress = covars[2][i][i]
        
        d = calculate_cohens_d(mu_calm, mu_stress, var_calm, var_stress)
        
        separability_report[feature] = {
            'Cohen_d': d,
            'is_separable': d >= threshold
        }
        
        if d < threshold:
            logger.warning(f"Feature '{feature}' fails separability check! Cohen's d = {d:.2f} < {threshold}")
        else:
            logger.info(f"Feature '{feature}' separates well. Cohen's d = {d:.2f}")
            
    return separability_report
