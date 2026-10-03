import numpy as np

def wmse(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """Calculate Weighted Mean Squared Error.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        weights: Weight array for each element.
        
    Returns:
        Weighted mean squared error between x1 and x2.
    """
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    normalized_weights = weights_arr / np.sum(weights_arr)
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))
