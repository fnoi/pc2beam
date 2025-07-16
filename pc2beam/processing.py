import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
from .data import S2Features

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

def calculate_s2(
    points: np.ndarray, 
    instances: np.ndarray,
    distance_threshold: float = 0.01, 
    ransac_n: int = 3, 
    num_iterations: int = 1000
) -> dict:
    """
    Calculate segment orientation feature s2 for each cluster.
    
    Args:
        points: Point coordinates of shape (N, 3)
        instances: Instance labels of shape (N,)
        distance_threshold: Maximum distance a point can be from the plane model
        ransac_n: Number of points to randomly sample for each RANSAC iteration
        num_iterations: Number of RANSAC iterations
        
    Returns:
        s2_features: Dictionary containing instance-level features:
            - Dictionary keys are instance IDs
            - Each instance has 's2' direction vector and 'line_point' position
    """
    if instances is None:
        raise ValueError("Instances are required to calculate s2 feature")
    
    # Process each instance separately
    unique_instances = np.unique(instances)
    
    # Dictionary to store results
    instance_features = {}
    
    for instance_id in unique_instances:
        # Get points for this instance
        instance_mask = instances == instance_id
        instance_points = points[instance_mask]
        
        # Create open3d point cloud for this instance
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.float64(instance_points))

        # Run RANSAC to fit plane model
        plane_P1, inliers_P1 = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

        # remove inliers from pcd
        pcd = pcd.select_by_index(inliers_P1, invert=True)

        # find a suitable P2
        while True:
            plane_P2, inliers_P2 = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
            angle_P1_P2 = np.rad2deg(np.arccos(np.dot(plane_P1[:3], plane_P2[:3])))
            print(f'angle_P1_P2: {angle_P1_P2}')
            if 45 < angle_P1_P2 < 135:
                break
            else:
                print('not suitable P2, trying again')
                pcd = pcd.select_by_index(inliers_P2, invert=True)
                if len(pcd.points) < 0.1 * len(instance_points):
                    print('not enough points left, breaking')
                    break

        n_P1, d_P1 = np.array(plane_P1[:3], dtype=np.float64), plane_P1[3]
        n_P2, d_P2 = np.array(plane_P2[:3], dtype=np.float64), plane_P2[3]

        # calculate s2
        s2 = np.cross(n_P1, n_P2)
        s2 = s2 / np.linalg.norm(s2)  # Normalize
        
        line_point = np.cross((n_P1 * d_P2 - n_P2 * d_P1), s2) / np.linalg.norm(s2) ** 2
        
        # Store features for this instance
        instance_features[instance_id] = {
            "s2": s2,
            "line_point": line_point
        }

    # Return instance features dictionary
    return instance_features


def calculate_s2_object(
    points: np.ndarray, 
    instances: np.ndarray,
    distance_threshold: float = 0.01, 
    ransac_n: int = 3, 
    num_iterations: int = 1000
) -> S2Features:
    """
    Calculate segment orientation feature s2 for each cluster.
    
    Args:
        points: Point coordinates of shape (N, 3)
        instances: Instance labels of shape (N,)
        distance_threshold: Maximum distance a point can be from the plane model
        ransac_n: Number of points to randomly sample for each RANSAC iteration
        num_iterations: Number of RANSAC iterations
        
    Returns:
        s2_features: S2Features object containing instance-level features
    """
    # Get dictionary result from original function
    instance_features_dict = calculate_s2(
        points, instances, distance_threshold, ransac_n, num_iterations
    )
    
    # Convert to S2Features object
    return S2Features.from_dict(instance_features_dict)

def project_to_line(
    points: np.ndarray,
    instances: np.ndarray,
    s2_vectors: np.ndarray,
    line_points: np.ndarray,
    min_points_per_instance: int = 2
) -> dict:
    """
    Project the points of each instance to the line defined by s2 and line_point.
    
    Args:
        points: Point coordinates of shape (N, 3)
        instances: Instance labels of shape (N,)
        s2_vectors: Beam direction vectors of shape (N, 3)
        line_points: Points on the beam line of shape (N, 3)
        min_points_per_instance: Minimum number of points required per instance
        
    Returns:
        Dictionary containing:
            - projected_points: Points projected onto beam lines of shape (N, 3)
            - distances: Distances from original points to beam lines of shape (N,)
            - instance_info: Length of each beam instance
    """
    if instances is None:
        raise ValueError("Instances are required for beam projection")
    
    if s2_vectors is None or line_points is None:
        raise ValueError("S2 features (s2 vectors and line points) are required for beam projection")
    
    # Initialize output arrays
    N = len(points)
    projected_points = np.zeros((N, 3))
    distances = np.zeros(N)
    
    # Dictionary to store instance-specific information
    instance_info = {}
    
    # Process each instance separately
    unique_instances = np.unique(instances)
    
    for instance_id in unique_instances:
        # Get points for this instance
        instance_mask = instances == instance_id
        instance_points = points[instance_mask]
        
        # Skip instances with too few points
        if len(instance_points) < min_points_per_instance:
            continue
        
        # Get s2 vector and line point for this instance
        # Assuming all points in the instance have the same s2 vector and line point
        s2_vec = s2_vectors[instance_mask][0]  # Direction vector (normalized)
        line_pt = line_points[instance_mask][0]  # Point on the line
        
        # Project each point in the instance to the line
        for i, idx in enumerate(np.where(instance_mask)[0]):
            # Vector from line point to the current point
            v = points[idx] - line_pt
            
            # Projection of v onto s2 (dot product)
            proj_dist = np.dot(v, s2_vec)
            
            # Calculate projected point
            projected_points[idx] = line_pt + proj_dist * s2_vec
            
            # Calculate distance from original point to the line
            dist_vec = points[idx] - projected_points[idx]
            distances[idx] = np.linalg.norm(dist_vec)
        
        # Calculate instance length by finding the extent of projected points
        instance_projected = projected_points[instance_mask]
        
        # Project all points to the line
        v = instance_points - line_pt
        proj_dists = np.dot(v, s2_vec)
        
        # Get min and max distances along the beam direction
        min_dist = np.min(proj_dists)
        max_dist = np.max(proj_dists)
        
        # Store the length and endpoints
        instance_length = max_dist - min_dist
        start_point = line_pt + min_dist * s2_vec
        end_point = line_pt + max_dist * s2_vec
        
        instance_info[instance_id] = {
            "length": instance_length,
            "start_point": start_point,
            "end_point": end_point
        }
    
    return {
        "projected_points": projected_points,
        "distances": distances,
        "instance_info": instance_info
    }

def project_to_centerline(
    points: np.ndarray,
    instances: np.ndarray,
    instance_features: dict
) -> dict:
    """
    Project the points of each instance to the line defined by s2 and line_point,
    and extract centerline endpoints.
    
    Args:
        points: Point coordinates of shape (N, 3)
        instances: Instance labels of shape (N,)
        instance_features: Dictionary with s2 and line_point for each instance
        
    Returns:
        Dictionary containing:
            - distances: Distances from original points to centerlines of shape (N,)
            - centerlines: Dictionary with information about each centerline
    """
    if instances is None:
        raise ValueError("Instances are required for centerline projection")
    
    if instance_features is None:
        raise ValueError("Instance features are required for centerline projection")
    
    # Initialize output arrays
    N = len(points)
    distances = np.zeros(N)
    
    # Dictionary to store centerline information
    centerlines = {}
    
    # Process each instance separately
    unique_instances = np.unique(instances)
    
    for instance_id in unique_instances:
        # Skip if instance features are not available
        if instance_id not in instance_features:
            continue
            
        # Get points for this instance
        instance_mask = instances == instance_id
        instance_points = points[instance_mask]
        
        # Skip instances with too few points (need at least 2 to define a line)
        if len(instance_points) < 2:
            continue
        
        # Get s2 vector and line point for this instance
        s2_vec = instance_features[instance_id]["s2"]  # Direction vector (normalized)
        line_pt = instance_features[instance_id]["line_point"]  # Point on the line
        
        # Project all points to the line (for distance calculation)
        for i, idx in enumerate(np.where(instance_mask)[0]):
            # Vector from line point to the current point
            v = points[idx] - line_pt
            
            # Projection of v onto s2 (dot product)
            proj_dist = np.dot(v, s2_vec)
            
            # Calculate projected point for distance calculation
            projected_point = line_pt + proj_dist * s2_vec
            
            # Calculate distance from original point to the line
            dist_vec = points[idx] - projected_point
            distances[idx] = np.linalg.norm(dist_vec)
        
        # Calculate centerline by finding the extent of projected points
        # Project all points to the line at once
        v = instance_points - line_pt
        proj_dists = np.dot(v, s2_vec)
        
        # Get min and max distances along the centerline direction
        min_dist = np.min(proj_dists)
        max_dist = np.max(proj_dists)
        
        # Store the centerline information
        centerline_length = max_dist - min_dist
        start_point = line_pt + min_dist * s2_vec
        end_point = line_pt + max_dist * s2_vec
        
        centerlines[instance_id] = {
            "length": centerline_length,
            "start_point": start_point,
            "end_point": end_point,
            "direction": s2_vec
        }
    
    return {
        "distances": distances,
        "centerlines": centerlines
    }