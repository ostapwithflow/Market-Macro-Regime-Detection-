import numpy as np
import logging

logger = logging.getLogger(__name__)

def check_centroid_collapse(hmm_model, distance_threshold: float = 0.1) -> bool:
    """
    Checks if any two states have collapsed into the same centroid (meaning).
    Returns True if a collapse is detected.
    """
    means = hmm_model.means_
    n_states = means.shape[0]
    
    for i in range(n_states):
        for j in range(i + 1, n_states):
            dist = np.linalg.norm(means[i] - means[j])
            if dist < distance_threshold:
                logger.error(f"Centroid Collapse Detected! State {i} and State {j} distance = {dist:.4f}")
                return True
                
    logger.info("No centroid collapse detected. States are distinct.")
    return False
