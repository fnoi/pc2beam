import numpy as np
from sklearn.neighbors import KDTree

def calculate_s1(points: np.ndarray, normals: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Calculate local orientation supernormal feature s1 for each point.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Normal vectors of shape (N, 3)
        k: Number of nearest neighbors for local analysis
        
    Returns:
        s1_features: Dictionary containing:
            - s1: Point-wise s1 values of shape (N,)
            - sigma1: First singular values of shape (N,)
            - sigma2: Second singular values of shape (N,)
            - sigma3: Third singular values of shape (N,)
    """
    if normals is None:
        raise ValueError("Normals are required to calculate s1 feature")
        
    # Build KD-tree for nearest neighbor search
    tree = KDTree(points)
    
    # Initialize output arrays
    N = len(points)
    s1 = np.zeros(N)
    sigma1 = np.zeros(N)
    sigma2 = np.zeros(N)
    sigma3 = np.zeros(N)
    
    # For each point, analyze its local neighborhood
    for i in range(N):
        # Find k nearest neighbors
        _, indices = tree.query(points[i:i+1], k=k)
        local_normals = normals[indices[0]]
        
        # Ensure normals are unit vectors
        local_normals = local_normals / np.linalg.norm(local_normals, axis=1, keepdims=True)
        
        # Flip normals for consistency
        local_normals = consistency_flip(local_normals)
        
        # Calculate SVD of local normals
        U, S, V = np.linalg.svd(local_normals, full_matrices=True)
        
        # Store results
        s1[i] = V[-1, 2]  # Z component of the least significant direction
        sigma1[i] = S[0]
        sigma2[i] = S[1]
        sigma3[i] = S[2]
    
    return {
        "s1": s1,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3
    }


def consistency_flip(normals):
    """Ensure consistent orientation of normal vectors."""
    # if input is empty or only one, return as is
    if normals.size == 0 or normals.shape[0] == 1:
        return normals
    
    # mean resulting vector
    vector_mean = np.mean(normals, axis=0)
    # normalize vector_mean
    vector_mean = vector_mean / np.linalg.norm(vector_mean)

    # calculate dot product between vector_mean and global Z axis
    dot_product = np.dot(vector_mean, np.array([0, 0, 1]))

    # flip if dot product is negative
    if dot_product < 0:
        normals = -normals

    return normals