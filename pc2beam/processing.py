import numpy as np

def calculate_s1(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Calculate local orientation supernormal feature s1.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Normal vectors of shape (N, 3)
        
    Returns:
        s1 feature values of shape (N,)
    """
    if normals is None:
        raise ValueError("Normals are required to calculate s1 feature")
    
    
    
    # Simple implementation - can be enhanced with more sophisticated algorithms
    # For now, we'll use the dot product of each normal with the global Z axis
    z_axis = np.array([0, 0, 1])
    s1 = np.abs(np.dot(normals, z_axis))
    
    return s1


