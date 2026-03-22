import math
import numpy as np

def _bayer_matrix(n: int) -> np.ndarray:
    """
    Create an n x n Bayer threshold matrix normalized to [0,1).
    n must be power of two (2,4,8,16,...). n=8 is a good default.
    """
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two")
    M = np.array([[0, 2],
                  [3, 1]], dtype=np.float32)
    size = 2
    while size < n:
        M = np.block([
            [4*M + 0, 4*M + 2],
            [4*M + 3, 4*M + 1],
        ])
        size *= 2
    # Normalize to [0,1)
    return (M + 0.5) / (n * n)

def bayer_dither_pattern(height: int, width: int, n: int) -> np.ndarray:
    """
    Create a Bayer dithering pattern of shape (height, width) using an n x n Bayer matrix.
    n must be power of two (2,4,8,16,...). n=8 is a good default.
    The pattern values are in [0,1).
    """
    bayer_matrix = _bayer_matrix(n)
    pattern = np.tile(bayer_matrix, (math.ceil(height / n), math.ceil(width / n)))
    return pattern[:height, :width]