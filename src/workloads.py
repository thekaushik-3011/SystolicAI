"""
Workload generation and matrix skewing.
"""
import numpy as np

def generate_matrices(M, K, N, seed=None):
    """
    Generate synthetic random matrices for testing.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randint(1, 10, size=(M, K)).astype(float)
    B = np.random.randint(1, 10, size=(K, N)).astype(float)
    return A, B

def skew_matrices(A, B):
    """
    Skew the input matrices so they flow correctly into the systolic array.
    
    A flows left-to-right. Element A[r, k] arrives at PE(r, 0) at cycle r + k.
    B flows top-to-bottom. Element B[k, c] arrives at PE(0, c) at cycle c + k.
    
    Returns:
        skewed_A: numpy array of shape (total_cycles, M)
        skewed_B: numpy array of shape (total_cycles, N)
    """
    M, K = A.shape
    K_b, N = B.shape
    if K != K_b:
        raise ValueError(f"Inner dimensions must match: {K} != {K_b}")
        
    total_cycles = M + N + K - 2
    
    # +1 because cycles are 0-indexed and max cycle is `total_cycles`
    num_cycles = total_cycles + 1
    
    skewed_A = np.zeros((num_cycles, M))
    skewed_B = np.zeros((num_cycles, N))
    
    for r in range(M):
        for k in range(K):
            cycle = r + k
            skewed_A[cycle, r] = A[r, k]
            
    for c in range(N):
        for k in range(K):
            cycle = c + k
            skewed_B[cycle, c] = B[k, c]
            
    return skewed_A, skewed_B

def generate_conv2d(batch, in_channels, height, width, out_channels, kernel_size, seed=None):
    """
    Generate synthetic random data for a Conv2D layer.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random image: (batch, in_channels, height, width)
    image = np.random.randint(1, 10, size=(batch, in_channels, height, width)).astype(float)
    
    # Random filters: (out_channels, in_channels, kernel_size, kernel_size)
    filters = np.random.randint(1, 10, size=(out_channels, in_channels, kernel_size, kernel_size)).astype(float)
    
    return image, filters

def reference_conv2d(image, filters):
    """
    Direct NumPy reference implementation of 2D Convolution.
    """
    batch, in_channels, height, width = image.shape
    out_channels, _, kernel_size, _ = filters.shape
    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1
    
    output = np.zeros((batch, out_channels, out_h, out_w))
    for b in range(batch):
        for oc in range(out_channels):
            for y in range(out_h):
                for x in range(out_w):
                    patch = image[b, :, y:y+kernel_size, x:x+kernel_size]
                    filt = filters[oc, :, :, :]
                    output[b, oc, y, x] = np.sum(patch * filt)
    return output

def im2col(image, filters):
    """
    Lower a Convolutional layer to Matrix Multiplication (A x B).
    
    Args:
        image: (batch, in_channels, height, width)
        filters: (out_channels, in_channels, kernel_size, kernel_size)
        
    Returns:
        A: The image lowered to a matrix of shape (batch * out_h * out_w, in_channels * kernel_size * kernel_size)
        B: The filters lowered to a matrix of shape (in_channels * kernel_size * kernel_size, out_channels)
        out_shape: Tuple of the expected output shape (batch, out_channels, out_h, out_w)
    """
    batch, in_channels, height, width = image.shape
    out_channels, _, kernel_size, _ = filters.shape
    
    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1
    
    # Prepare A (Activations/Image)
    # Number of columns is the size of one flattened patch
    patch_size = in_channels * kernel_size * kernel_size
    A = np.zeros((batch * out_h * out_w, patch_size))
    
    row_idx = 0
    for b in range(batch):
        for y in range(out_h):
            for x in range(out_w):
                # Extract patch
                patch = image[b, :, y:y+kernel_size, x:x+kernel_size]
                A[row_idx, :] = patch.flatten()
                row_idx += 1
                
    # Prepare B (Weights/Filters)
    # Flatten each filter into a column
    B = np.zeros((patch_size, out_channels))
    for c in range(out_channels):
        B[:, c] = filters[c, :, :, :].flatten()
        
    out_shape = (batch, out_channels, out_h, out_w)
    
    return A, B, out_shape

def col2im(C, out_shape):
    """
    Reconstruct the 4D output tensor from the 2D GEMM output matrix.
    
    Args:
        C: The output matrix of shape (batch * out_h * out_w, out_channels)
        out_shape: Tuple of (batch, out_channels, out_h, out_w)
        
    Returns:
        output: Reshaped and transposed 4D tensor of shape out_shape
    """
    batch, out_channels, out_h, out_w = out_shape
    # C is ordered by: batch, then y (out_h), then x (out_w) as rows, and out_channels as columns.
    # Therefore, reshape to (batch, out_h, out_w, out_channels)
    output = C.reshape(batch, out_h, out_w, out_channels)
    # Transpose to (batch, out_channels, out_h, out_w)
    output = np.transpose(output, (0, 3, 1, 2))
    return output
