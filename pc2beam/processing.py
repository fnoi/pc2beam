import numpy as np
from sklearn.neighbors import KDTree

def calculate_s1(
    points: np.ndarray, 
    normals: np.ndarray, 
    radius: float = 0.1,
    k: int = 30, 
    use_radius: bool = True
) -> np.ndarray:
    """
    Calculate local orientation supernormal feature s1 for each point.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Normal vectors of shape (N, 3)
        radius: Radius for spherical neighborhood search
        k: Number of nearest neighbors if not using radius-based search
        use_radius: Whether to use radius-based search (True) or k-nearest neighbors (False)
        
    Returns:
        s1_features: Dictionary containing:
            - s1: Point-wise s1 values of shape (N, 3)
            - sigma1: First singular values of shape (N,)
            - sigma2: Second singular values of shape (N,)
            - sigma3: Third singular values of shape (N,)
    """
    if normals is None:
        raise ValueError("Normals are required to calculate s1 feature")
        
    # Build KD-tree for neighborhood search
    tree = KDTree(points)
    
    # Initialize output arrays
    N = len(points)
    s1 = np.zeros((N, 3))
    sigma1 = np.zeros(N)
    sigma2 = np.zeros(N)
    sigma3 = np.zeros(N)
    
    # For each point, analyze its local neighborhood
    for i in range(N):
        # Get local neighborhood
        if use_radius:
            # Radius-based search (spherical neighborhood)
            indices = tree.query_radius(points[i:i+1], radius)[0]
            # Need at least 4 points to perform SVD reliably
            if len(indices) < 4:
                # Fall back to KNN if not enough points in radius
                _, indices = tree.query(points[i:i+1], k=min(k, N))
                indices = indices[0]
        else:
            # K-nearest neighbors search
            _, indices = tree.query(points[i:i+1], k=min(k, N))
            indices = indices[0]
            
        # Get local normals
        local_normals = normals[indices]
        
        # Ensure normals are unit vectors
        local_normals = local_normals / np.linalg.norm(local_normals, axis=1, keepdims=True)
        
        # Flip normals for consistency
        local_normals = consistency_flip(local_normals)
        
        # Calculate SVD of local normals
        try:
            U, S, V = np.linalg.svd(local_normals, full_matrices=True)
            
            # Store results
            s1[i, :] = V[-1, :] # last column of V is the s1 vector
            sigma1[i] = S[0]
            sigma2[i] = S[1]
            sigma3[i] = S[2]
        except np.linalg.LinAlgError:
            # If SVD fails, set default values
            s1[i, :] = [0, 0, 1]  # Default to vertical
            sigma1[i] = 1.0
            sigma2[i] = 0.0
            sigma3[i] = 0.0
    
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