import math
from numba import cuda

@cuda.jit
def calc_features_gpu(close, high, low, window, z_out, atr_out):
    """Compute Z-Score and ATR on the GPU in a single pass."""
    idx = cuda.grid(1)
    if idx >= window and idx < close.shape[0]:
        # Z-Score Logic
        s = 0.0
        for i in range(idx - window, idx): s += close[i]
        mean = s / window
        
        sq_diff = 0.0
        for i in range(idx - window, idx): sq_diff += (close[i] - mean)**2
        std = math.sqrt(sq_diff / window)
        
        if std > 0: z_out[idx] = (close[idx] - mean) / std
        
        # ATR Logic
        tr_sum = 0.0
        for i in range(idx - window, idx):
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            tr_sum += tr
        atr_out[idx] = tr_sum / window