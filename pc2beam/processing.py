import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
from tqdm import tqdm


def normal_and_point_to_plane(normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    normal_arr = np.asarray(normal, dtype=np.float64)
    point_arr = np.asarray(point, dtype=np.float64)
    if normal_arr.shape != (3,) or point_arr.shape != (3,):
        raise ValueError("normal and point must both have shape (3,)")
    normal_norm = np.linalg.norm(normal_arr)
    if normal_norm < 1e-12:
        raise ValueError("normal has near-zero norm")
    n = normal_arr / normal_norm
    d = -float(np.dot(n, point_arr))
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def intersecting_line(plane1: np.ndarray, plane2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.asarray(plane1, dtype=np.float64)
    p2 = np.asarray(plane2, dtype=np.float64)
    if p1.shape != (4,) or p2.shape != (4,):
        raise ValueError("planes must have shape (4,)")
    normal1, d1 = p1[:3], float(p1[3])
    normal2, d2 = p2[:3], float(p2[3])
    n1_norm = np.linalg.norm(normal1)
    n2_norm = np.linalg.norm(normal2)
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        raise ValueError("plane normal has near-zero norm")
    normal1 = normal1 / n1_norm
    normal2 = normal2 / n2_norm
    direction = np.cross(normal1, normal2)
    dir_norm_sq = float(np.dot(direction, direction))
    if dir_norm_sq < 1e-12:
        raise ValueError("planes are parallel or coincident")
    origin = np.cross((normal1 * d2 - normal2 * d1), direction) / dir_norm_sq
    return direction / np.linalg.norm(direction), origin


def points_to_actual_plane(
    points: np.ndarray,
    normal: np.ndarray,
    point_on_plane: np.ndarray,
) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float64)
    normal_arr = np.asarray(normal, dtype=np.float64)
    point_arr = np.asarray(point_on_plane, dtype=np.float64)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    n_norm = np.linalg.norm(normal_arr)
    if n_norm < 1e-12:
        raise ValueError("normal has near-zero norm")
    n = normal_arr / n_norm
    return points_arr - np.dot(points_arr - point_arr, n)[:, np.newaxis] * n


def simplified_transform_lines(source_angle: tuple, target_angle: tuple) -> np.ndarray:
    src_left, src_common, src_right = map(lambda x: np.asarray(x, dtype=np.float64), source_angle)
    tgt_left, tgt_common, tgt_right = map(lambda x: np.asarray(x, dtype=np.float64), target_angle)

    src_x = src_left - src_common
    src_y = src_right - src_common
    src_z = np.cross(src_x, src_y)
    tgt_x = tgt_left - tgt_common
    tgt_y = tgt_right - tgt_common
    tgt_z = np.cross(tgt_x, tgt_y)

    if (
        np.linalg.norm(src_x) < 1e-12
        or np.linalg.norm(src_y) < 1e-12
        or np.linalg.norm(src_z) < 1e-12
        or np.linalg.norm(tgt_x) < 1e-12
        or np.linalg.norm(tgt_y) < 1e-12
        or np.linalg.norm(tgt_z) < 1e-12
    ):
        raise ValueError("degenerate source/target angle basis")

    src_x = src_x / np.linalg.norm(src_x)
    src_y = src_y / np.linalg.norm(src_y)
    src_z = src_z / np.linalg.norm(src_z)
    tgt_x = tgt_x / np.linalg.norm(tgt_x)
    tgt_y = tgt_y / np.linalg.norm(tgt_y)
    tgt_z = tgt_z / np.linalg.norm(tgt_z)

    rot_src = np.column_stack((src_x, src_y, src_z))
    rot_tgt = np.column_stack((tgt_x, tgt_y, tgt_z))
    rotation = np.dot(rot_tgt, rot_src.T)
    translation = tgt_common - np.dot(rotation, src_common)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

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
        "s1_sigma1": sigma1,
        "s1_sigma2": sigma2,
        "s1_sigma3": sigma3
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
    num_iterations: int = 1000,
    min_points_per_instance: int = 20,
    min_remaining_points: int = 10,
    min_plane_inliers: int = 10,
    angle_min_deg: float = 30.0,
    angle_max_deg: float = 150.0,
    enable_fallback: bool = True,
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
    
    if min_points_per_instance < ransac_n:
        min_points_per_instance = ransac_n

    # Process each instance separately
    unique_instances = np.unique(instances)
    
    # Dictionary to store results
    instance_features = {}
    
    for instance_idx, instance_id in enumerate(unique_instances, start=1):
        # Get points for this instance
        instance_mask = instances == instance_id
        instance_points = points[instance_mask]
        point_count = int(len(instance_points))

        if point_count < min_points_per_instance:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "insufficient_points",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": 0,
                "plane2_inliers": 0,
                "plane_angle_deg": None,
                "method": "none",
                "message": (
                    f"instance has {point_count} points; requires at least "
                    f"{min_points_per_instance}"
                ),
            }
            continue

        # Create open3d point cloud for this instance
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.float64(instance_points))

        # Run RANSAC to fit first plane model
        try:
            plane_P1, inliers_P1 = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )
        except RuntimeError:
            inliers_P1 = []
            plane_P1 = None

        if plane_P1 is None or len(inliers_P1) < min_plane_inliers:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "plane1_failed",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": int(len(inliers_P1)),
                "plane2_inliers": 0,
                "plane_angle_deg": None,
                "method": "none",
                "message": "failed to estimate first plane robustly",
            }
            continue

        n_P1 = np.array(plane_P1[:3], dtype=np.float64)
        norm_P1 = np.linalg.norm(n_P1)
        if norm_P1 < 1e-12:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "plane1_degenerate",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": int(len(inliers_P1)),
                "plane2_inliers": 0,
                "plane_angle_deg": None,
                "method": "none",
                "message": "first plane normal is degenerate",
            }
            continue
        n_P1 = n_P1 / norm_P1
        d_P1 = float(plane_P1[3])

        # remove inliers from pcd by inverse selection
        pcd_remaining = pcd.select_by_index(inliers_P1, invert=True)

        # find a suitable second plane P2
        plane_P2 = None
        inliers_P2 = []
        angle_P1_P2 = None
        while len(pcd_remaining.points) >= max(min_remaining_points, ransac_n):
            try:
                plane_P2_candidate, inliers_P2_candidate = pcd_remaining.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations,
                )
            except RuntimeError:
                break

            if len(inliers_P2_candidate) < min_plane_inliers:
                break

            n_P2_candidate = np.array(plane_P2_candidate[:3], dtype=np.float64)
            norm_P2 = np.linalg.norm(n_P2_candidate)
            if norm_P2 < 1e-12:
                pcd_remaining = pcd_remaining.select_by_index(
                    inliers_P2_candidate, invert=True
                )
                continue
            n_P2_candidate = n_P2_candidate / norm_P2

            dot = float(np.clip(np.dot(n_P1, n_P2_candidate), -1.0, 1.0))
            angle = float(np.rad2deg(np.arccos(dot)))
            print(
                f"(instance {instance_idx}/{len(unique_instances)}) "
                f"angle_P1_P2: {angle:.2f}"
            )
            if angle_min_deg <= angle <= angle_max_deg:
                plane_P2 = plane_P2_candidate
                inliers_P2 = inliers_P2_candidate
                angle_P1_P2 = angle
                break

            pcd_remaining = pcd_remaining.select_by_index(inliers_P2_candidate, invert=True)

        used_fallback = False
        if plane_P2 is None and enable_fallback:
            centered = instance_points - instance_points.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            fallback_direction = vh[0]
            fallback_norm = np.linalg.norm(fallback_direction)
            if fallback_norm > 1e-12:
                fallback_direction = fallback_direction / fallback_norm
                fallback_point = instance_points.mean(axis=0)
                instance_features[instance_id] = {
                    "s2_vector": fallback_direction.astype(np.float64),
                    "s2_point": fallback_point.astype(np.float64),
                    "status": "fallback_pca",
                    "confidence": 0.3,
                    "point_count": point_count,
                    "plane1_inliers": int(len(inliers_P1)),
                    "plane2_inliers": 0,
                    "plane_angle_deg": None,
                    "method": "pca",
                    "message": "second plane estimation failed, used PCA fallback",
                }
                used_fallback = True
        if used_fallback:
            continue

        if plane_P2 is None:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "plane2_failed",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": int(len(inliers_P1)),
                "plane2_inliers": 0,
                "plane_angle_deg": None,
                "method": "none",
                "message": "failed to find a valid second plane",
            }
            continue

        n_P2 = np.array(plane_P2[:3], dtype=np.float64)
        norm_P2 = np.linalg.norm(n_P2)
        if norm_P2 < 1e-12:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "plane2_degenerate",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": int(len(inliers_P1)),
                "plane2_inliers": int(len(inliers_P2)),
                "plane_angle_deg": angle_P1_P2,
                "method": "none",
                "message": "second plane normal is degenerate",
            }
            continue
        n_P2 = n_P2 / norm_P2
        d_P2 = float(plane_P2[3])

        # calculate s2
        s2 = np.cross(n_P1, n_P2)
        norm_s2 = np.linalg.norm(s2)
        if norm_s2 < 1e-12:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "cross_product_degenerate",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": int(len(inliers_P1)),
                "plane2_inliers": int(len(inliers_P2)),
                "plane_angle_deg": angle_P1_P2,
                "method": "none",
                "message": "plane normals are nearly parallel",
            }
            continue
        s2 = s2 / norm_s2  # Normalize

        line_point = np.cross((n_P1 * d_P2 - n_P2 * d_P1), s2)
        denom = np.linalg.norm(s2) ** 2
        if denom < 1e-12:
            instance_features[instance_id] = {
                "s2_vector": None,
                "s2_point": None,
                "status": "line_point_failed",
                "confidence": 0.0,
                "point_count": point_count,
                "plane1_inliers": int(len(inliers_P1)),
                "plane2_inliers": int(len(inliers_P2)),
                "plane_angle_deg": angle_P1_P2,
                "method": "none",
                "message": "failed to compute robust line anchor point",
            }
            continue
        line_point = line_point / denom

        inlier_ratio = min(
            1.0,
            (len(inliers_P1) + len(inliers_P2)) / max(float(point_count), 1.0),
        )
        if angle_P1_P2 is None:
            angle_score = 0.0
        else:
            angle_score = 1.0 - min(abs(angle_P1_P2 - 90.0) / 90.0, 1.0)
        confidence = float(np.clip(0.7 * inlier_ratio + 0.3 * angle_score, 0.0, 1.0))
        
        # Store features for this instance
        instance_features[instance_id] = {
            "s2_vector": s2.astype(np.float64),
            "s2_point": np.asarray(line_point, dtype=np.float64),
            "status": "ok",
            "confidence": confidence,
            "point_count": point_count,
            "plane1_inliers": int(len(inliers_P1)),
            "plane2_inliers": int(len(inliers_P2)),
            "plane_angle_deg": float(angle_P1_P2) if angle_P1_P2 is not None else None,
            "method": "plane_intersection",
            "message": None,
        }

    # Return instance features dictionary
    return instance_features


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

def project_points_to_line(points, s2_vector, s2_point):
    """projects points to a line and returns endpoints of line segment containing all projected points"""
    points_arr = np.asarray(points, dtype=np.float64)
    s2_vec_arr = np.asarray(s2_vector, dtype=np.float64)
    s2_point_arr = np.asarray(s2_point, dtype=np.float64)

    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points_arr.shape[0] < 2:
        raise ValueError("at least two points are required to define a segment")
    if not np.isfinite(points_arr).all():
        raise ValueError("points contain non-finite values")
    if s2_vec_arr.shape != (3,) or not np.isfinite(s2_vec_arr).all():
        raise ValueError("s2_vector must be finite shape (3,)")
    if s2_point_arr.shape != (3,) or not np.isfinite(s2_point_arr).all():
        raise ValueError("s2_point must be finite shape (3,)")

    norm = np.linalg.norm(s2_vec_arr)
    if norm < 1e-12:
        raise ValueError("s2_vector norm is too small")

    direction = s2_vec_arr / norm
    t = np.dot(points_arr - s2_point_arr, direction)
    projections = s2_point_arr + t[:, np.newaxis] * direction
    start_pt = projections[np.argmin(t)]
    end_pt = projections[np.argmax(t)]
    if not np.isfinite(start_pt).all() or not np.isfinite(end_pt).all():
        raise ValueError("projected segment contains non-finite endpoints")
    return start_pt, end_pt


def extract_segment_endpoints_safe(
    points: np.ndarray,
    s2_vector: np.ndarray,
    s2_point: np.ndarray,
    min_segment_length: float = 1e-6,
) -> dict:
    """
    Project points to line and return structured success/failure result.
    """
    try:
        start_pt, end_pt = project_points_to_line(points, s2_vector, s2_point)
    except ValueError as exc:
        return {
            "ok": False,
            "reason": "projection_failed",
            "message": str(exc),
            "start_point": None,
            "end_point": None,
            "length": None,
        }

    seg_len = float(np.linalg.norm(end_pt - start_pt))
    if not np.isfinite(seg_len):
        return {
            "ok": False,
            "reason": "non_finite_length",
            "message": "segment length is non-finite",
            "start_point": None,
            "end_point": None,
            "length": None,
        }
    if seg_len < float(min_segment_length):
        return {
            "ok": False,
            "reason": "segment_too_short",
            "message": f"segment length {seg_len:.6g} below minimum {min_segment_length:.6g}",
            "start_point": None,
            "end_point": None,
            "length": seg_len,
        }

    return {
        "ok": True,
        "reason": "ok",
        "message": None,
        "start_point": np.asarray(start_pt, dtype=np.float64),
        "end_point": np.asarray(end_pt, dtype=np.float64),
        "length": seg_len,
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


def orientation_estimation_s2_legacy(
    points: np.ndarray,
    normals: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_plane_inliers: int = 10,
    angle_min_deg: float = 45.0,
    angle_max_deg: float = 135.0,
) -> dict:
    """Legacy-like two-plane estimation used for projection alignment."""
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or nrm.shape != pts.shape:
        raise ValueError("points and normals must both have shape (N, 3)")
    if len(pts) < max(2 * ransac_n, 20):
        return {"ok": False, "message": "insufficient_points"}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    try:
        plane0, inliers0 = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
    except RuntimeError:
        return {"ok": False, "message": "plane1_failed"}
    if len(inliers0) < min_plane_inliers:
        return {"ok": False, "message": "plane1_insufficient_inliers"}

    remaining = pcd.select_by_index(inliers0, invert=True)
    plane1 = None
    inliers1 = []
    angle_deg = None
    while len(remaining.points) >= max(min_plane_inliers, ransac_n):
        try:
            candidate, candidate_inliers = remaining.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )
        except RuntimeError:
            break
        if len(candidate_inliers) < min_plane_inliers:
            break
        n0 = np.asarray(plane0[:3], dtype=np.float64)
        n1 = np.asarray(candidate[:3], dtype=np.float64)
        if np.linalg.norm(n0) < 1e-12 or np.linalg.norm(n1) < 1e-12:
            remaining = remaining.select_by_index(candidate_inliers, invert=True)
            continue
        n0 = n0 / np.linalg.norm(n0)
        n1 = n1 / np.linalg.norm(n1)
        angle_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))))
        if angle_min_deg <= (angle_deg % 180.0) <= angle_max_deg:
            plane1 = candidate
            candidate_coords = np.asarray(remaining.points)[candidate_inliers]
            inliers1 = np.where(np.isin(pts, candidate_coords).all(axis=1))[0].tolist()
            break
        remaining = remaining.select_by_index(candidate_inliers, invert=True)

    if plane1 is None:
        return {"ok": False, "message": "plane2_failed"}

    n0 = np.asarray(plane0[:3], dtype=np.float64)
    n1 = np.asarray(plane1[:3], dtype=np.float64)
    direction = np.cross(n0, n1)
    if np.linalg.norm(direction) < 1e-12:
        return {"ok": False, "message": "cross_product_degenerate"}
    point_on_line = np.cross((n0 * plane1[3] - n1 * plane0[3]), direction) / (np.linalg.norm(direction) ** 2)
    return {
        "ok": True,
        "planes": (np.asarray(plane0, dtype=np.float64), np.asarray(plane1, dtype=np.float64)),
        "direction": direction / np.linalg.norm(direction),
        "origin": np.asarray(point_on_line, dtype=np.float64),
        "inliers_0": inliers0,
        "inliers_1": inliers1,
        "angle_deg": angle_deg,
    }


def project_points_by_plane_alignment(
    points: np.ndarray,
    normals: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_plane_inliers: int = 10,
) -> dict:
    """
    Legacy-style point projection/alignment by two fitted planes.
    """
    points_arr = np.asarray(points, dtype=np.float64)
    normals_arr = np.asarray(normals, dtype=np.float64)
    estimate = orientation_estimation_s2_legacy(
        points_arr,
        normals_arr,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        min_plane_inliers=min_plane_inliers,
    )
    if not estimate.get("ok", False):
        return {"ok": False, "status": estimate.get("message", "failed")}

    direction = estimate["direction"]
    origin = estimate["origin"]
    planes = estimate["planes"]
    projected_line_points, _ = project_points_to_line(points_arr, direction, origin)
    ref_t = (-1e5 - origin[0]) / direction[0] if abs(direction[0]) > 1e-12 else 0.0
    ref_pt = origin + ref_t * direction
    line_dists = np.linalg.norm(projected_line_points - ref_pt, axis=1)
    l_ind = int(np.argmin(line_dists))
    r_ind = int(np.argmax(line_dists))
    left_3d = projected_line_points[l_ind]
    right_3d = projected_line_points[r_ind]
    vector_3d = right_3d - left_3d
    v_norm = np.linalg.norm(vector_3d)
    if v_norm < 1e-12:
        return {"ok": False, "status": "degenerate_segment"}
    vector_3d = vector_3d / v_norm
    center_3d = (left_3d + right_3d) / 2.0

    proj_plane = normal_and_point_to_plane(vector_3d, left_3d)
    proj_dir_0, _ = intersecting_line(proj_plane, planes[0])
    proj_dir_1, _ = intersecting_line(proj_plane, planes[1])
    proj_points_plane = points_to_actual_plane(points_arr, vector_3d, left_3d)

    target_left = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    target_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target_right = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    source_left = left_3d + np.asarray(planes[0][:3], dtype=np.float64)
    source_center = left_3d
    source_right = source_center + vector_3d
    source_angle = (source_left, source_center, source_right)
    target_angle = (target_left, target_center, target_right)
    transform = simplified_transform_lines(source_angle, target_angle)

    points_hom = np.hstack((points_arr, np.ones((points_arr.shape[0], 1), dtype=np.float64)))
    normals_hom = np.hstack((normals_arr, np.zeros((normals_arr.shape[0], 1), dtype=np.float64)))
    points_target = (transform @ points_hom.T).T[:, :3]
    normals_target = (transform @ normals_hom.T).T[:, :3]
    normals_2d = normals_target[:, :2]
    n2d_norm = np.linalg.norm(normals_2d, axis=1, keepdims=True)
    n2d_norm[n2d_norm < 1e-12] = 1.0

    return {
        "ok": True,
        "status": "ok",
        "planes": planes,
        "inliers_0": estimate["inliers_0"],
        "inliers_1": estimate["inliers_1"],
        "plane_angle_deg": estimate.get("angle_deg"),
        "line_direction": direction,
        "line_origin": origin,
        "left_3d": left_3d,
        "right_3d": right_3d,
        "center_3d": center_3d,
        "vector_3d": vector_3d,
        "source_angle": source_angle,
        "target_angle": target_angle,
        "transform": transform,
        "points_plane_projected": proj_points_plane,
        "points_2d": points_target[:, :2],
        "normals_2d": normals_2d / n2d_norm,
        "proj_dir_0": proj_dir_0,
        "proj_dir_1": proj_dir_1,
    }