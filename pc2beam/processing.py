from __future__ import annotations

from typing import Any, Dict, Optional

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

def _build_threshold_schedule(
    distance_threshold: float,
    distance_threshold_schedule: Optional[list[float]],
) -> list[float]:
    if not distance_threshold_schedule:
        return [float(distance_threshold)]
    cleaned: list[float] = []
    for value in distance_threshold_schedule:
        try:
            thr = float(value)
        except (TypeError, ValueError):
            continue
        if thr > 0.0:
            cleaned.append(thr)
    if not cleaned:
        return [float(distance_threshold)]
    return sorted(set(cleaned))


def _nearest_plane_line_residuals(
    points: np.ndarray,
    plane1: np.ndarray,
    plane2: np.ndarray,
) -> np.ndarray:
    p_arr = np.asarray(points, dtype=np.float64)
    p1 = np.asarray(plane1, dtype=np.float64)
    p2 = np.asarray(plane2, dtype=np.float64)
    n1 = p1[:3]
    n2 = p2[:3]
    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        raise ValueError('plane normal has near-zero norm')
    d1 = np.abs(np.dot(p_arr, n1) + float(p1[3])) / n1_norm
    d2 = np.abs(np.dot(p_arr, n2) + float(p2[3])) / n2_norm
    return np.minimum(d1, d2)


def _compute_candidate_quality(
    points: np.ndarray,
    plane1: np.ndarray,
    plane2: np.ndarray,
    inliers_p1: int,
    inliers_p2: int,
    point_count: int,
    threshold: float,
    angle_deg: Optional[float],
    quality: Dict[str, float],
) -> Dict[str, float]:
    residuals = _nearest_plane_line_residuals(points, plane1, plane2)
    residual_med = float(np.median(residuals))
    residual_p90 = float(np.quantile(residuals, 0.90))
    residual_tube = float(quality['residual_tube_factor']) * float(threshold)
    support_ratio = float(np.mean(residuals <= residual_tube))
    denom = max(inliers_p1, inliers_p2, 1)
    plane_balance = float(min(inliers_p1, inliers_p2) / denom)
    inlier_ratio = min(1.0, (inliers_p1 + inliers_p2) / max(float(point_count), 1.0))
    if angle_deg is None:
        angle_score = 0.0
    else:
        angle_score = 1.0 - min(abs(float(angle_deg) - 90.0) / 90.0, 1.0)

    norm_med = min(residual_med / max(float(threshold), 1e-6), 1.0)
    score = (
        float(quality['weight_residual']) * (1.0 - norm_med)
        + float(quality['weight_support']) * support_ratio
        + float(quality['weight_balance']) * plane_balance
        + float(quality['weight_angle']) * angle_score
    )
    confidence = float(np.clip(0.6 * inlier_ratio + 0.4 * score, 0.0, 1.0))
    return {
        'projection_residual_median': residual_med,
        'projection_residual_p90': residual_p90,
        'projection_support_ratio': support_ratio,
        'plane_support_balance': plane_balance,
        'suitability_score': float(np.clip(score, 0.0, 1.0)),
        'confidence': confidence,
    }


def calculate_s2(
    points: np.ndarray,
    instances: np.ndarray,
    distance_threshold: float = 0.01,
    distance_threshold_schedule: Optional[list[float]] = None,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_points_per_instance: int = 20,
    min_remaining_points: int = 10,
    min_plane_inliers: int = 10,
    angle_min_deg: float = 30.0,
    angle_max_deg: float = 150.0,
    enable_fallback: bool = True,
    quality: Optional[Dict[str, float]] = None,
) -> dict:
    """Calculate segment orientation feature s2 for each cluster."""
    if instances is None:
        raise ValueError('Instances are required to calculate s2 feature')

    if min_points_per_instance < ransac_n:
        min_points_per_instance = ransac_n

    quality_cfg = {
        'support_ratio_min': 0.35,
        'p90_max_factor': 2.5,
        'residual_tube_factor': 1.5,
        'weight_residual': 0.45,
        'weight_support': 0.35,
        'weight_balance': 0.10,
        'weight_angle': 0.10,
    }
    if quality:
        for key in quality_cfg:
            if key in quality and quality[key] is not None:
                quality_cfg[key] = float(quality[key])

    threshold_schedule = _build_threshold_schedule(
        distance_threshold=distance_threshold,
        distance_threshold_schedule=distance_threshold_schedule,
    )

    unique_instances = np.unique(instances)
    instance_features = {}

    for instance_idx, instance_id in enumerate(unique_instances, start=1):
        instance_mask = instances == instance_id
        instance_points = points[instance_mask]
        point_count = int(len(instance_points))

        if point_count < min_points_per_instance:
            instance_features[instance_id] = {
                's2_vector': None,
                's2_point': None,
                'status': 'insufficient_points',
                'confidence': 0.0,
                'point_count': point_count,
                'plane1_inliers': 0,
                'plane2_inliers': 0,
                'plane_angle_deg': None,
                'method': 'none',
                'message': (
                    f'instance has {point_count} points; requires at least '
                    f'{min_points_per_instance}'
                ),
                'selected_distance_threshold': None,
                'threshold_attempts': 0,
                'projection_residual_median': None,
                'projection_residual_p90': None,
                'projection_support_ratio': None,
                'plane_support_balance': None,
                'suitability_score': None,
            }
            continue

        best_result = None
        best_diag = None
        best_threshold = None
        attempts = 0
        fail_status = 'plane1_failed'
        fail_message = 'failed to estimate first plane robustly'
        fail_plane1_inliers = 0
        fail_plane2_inliers = 0
        fail_angle = None

        for active_threshold in threshold_schedule:
            attempts += 1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.float64(instance_points))

            try:
                plane_p1, inliers_p1 = pcd.segment_plane(
                    distance_threshold=active_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations,
                )
            except RuntimeError:
                inliers_p1 = []
                plane_p1 = None

            fail_plane1_inliers = int(len(inliers_p1))
            fail_plane2_inliers = 0
            fail_angle = None
            if plane_p1 is None or len(inliers_p1) < min_plane_inliers:
                fail_status = 'plane1_failed'
                fail_message = 'failed to estimate first plane robustly'
                continue

            n_p1 = np.array(plane_p1[:3], dtype=np.float64)
            norm_p1 = np.linalg.norm(n_p1)
            if norm_p1 < 1e-12:
                fail_status = 'plane1_degenerate'
                fail_message = 'first plane normal is degenerate'
                continue
            n_p1 = n_p1 / norm_p1
            d_p1 = float(plane_p1[3])

            pcd_remaining = pcd.select_by_index(inliers_p1, invert=True)

            plane_p2 = None
            inliers_p2 = []
            angle_p1_p2 = None
            while len(pcd_remaining.points) >= max(min_remaining_points, ransac_n):
                try:
                    plane_p2_candidate, inliers_p2_candidate = pcd_remaining.segment_plane(
                        distance_threshold=active_threshold,
                        ransac_n=ransac_n,
                        num_iterations=num_iterations,
                    )
                except RuntimeError:
                    break

                if len(inliers_p2_candidate) < min_plane_inliers:
                    break

                n_p2_candidate = np.array(plane_p2_candidate[:3], dtype=np.float64)
                norm_p2 = np.linalg.norm(n_p2_candidate)
                if norm_p2 < 1e-12:
                    pcd_remaining = pcd_remaining.select_by_index(
                        inliers_p2_candidate, invert=True
                    )
                    continue
                n_p2_candidate = n_p2_candidate / norm_p2

                dot = float(np.clip(np.dot(n_p1, n_p2_candidate), -1.0, 1.0))
                angle = float(np.rad2deg(np.arccos(dot)))
                print(
                    f'(instance {instance_idx}/{len(unique_instances)}) '
                    f'threshold={active_threshold:.4f} angle_P1_P2: {angle:.2f}'
                )
                if angle_min_deg <= angle <= angle_max_deg:
                    plane_p2 = plane_p2_candidate
                    inliers_p2 = inliers_p2_candidate
                    angle_p1_p2 = angle
                    break

                pcd_remaining = pcd_remaining.select_by_index(inliers_p2_candidate, invert=True)

            fail_plane2_inliers = int(len(inliers_p2))
            fail_angle = angle_p1_p2
            if plane_p2 is None:
                fail_status = 'plane2_failed'
                fail_message = 'failed to find a valid second plane'
                continue

            n_p2 = np.array(plane_p2[:3], dtype=np.float64)
            norm_p2 = np.linalg.norm(n_p2)
            if norm_p2 < 1e-12:
                fail_status = 'plane2_degenerate'
                fail_message = 'second plane normal is degenerate'
                continue
            n_p2 = n_p2 / norm_p2
            d_p2 = float(plane_p2[3])

            s2 = np.cross(n_p1, n_p2)
            norm_s2 = np.linalg.norm(s2)
            if norm_s2 < 1e-12:
                fail_status = 'cross_product_degenerate'
                fail_message = 'plane normals are nearly parallel'
                continue
            s2 = s2 / norm_s2

            line_point = np.cross((n_p1 * d_p2 - n_p2 * d_p1), s2)
            denom = np.linalg.norm(s2) ** 2
            if denom < 1e-12:
                fail_status = 'line_point_failed'
                fail_message = 'failed to compute robust line anchor point'
                continue
            line_point = line_point / denom

            try:
                diag = _compute_candidate_quality(
                    points=instance_points,
                    plane1=np.asarray(plane_p1, dtype=np.float64),
                    plane2=np.asarray(plane_p2, dtype=np.float64),
                    inliers_p1=int(len(inliers_p1)),
                    inliers_p2=int(len(inliers_p2)),
                    point_count=point_count,
                    threshold=float(active_threshold),
                    angle_deg=angle_p1_p2,
                    quality=quality_cfg,
                )
            except ValueError:
                fail_status = 'quality_failed'
                fail_message = 'quality scoring failed for candidate planes'
                continue

            p90_limit = float(quality_cfg['p90_max_factor']) * float(active_threshold)
            hard_pass = (
                diag['projection_support_ratio'] >= float(quality_cfg['support_ratio_min'])
                and diag['projection_residual_p90'] <= p90_limit
            )
            if not hard_pass:
                fail_status = 'quality_gate_failed'
                fail_message = 'candidate failed projection quality gates'
                continue

            best_result = {
                's2_vector': s2.astype(np.float64),
                's2_point': np.asarray(line_point, dtype=np.float64),
                'status': 'ok',
                'point_count': point_count,
                'plane1_inliers': int(len(inliers_p1)),
                'plane2_inliers': int(len(inliers_p2)),
                'plane_angle_deg': float(angle_p1_p2) if angle_p1_p2 is not None else None,
                'method': 'plane_intersection',
                'message': None,
            }
            best_diag = diag
            best_threshold = float(active_threshold)
            break

        used_fallback = False
        if best_result is None and enable_fallback:
            centered = instance_points - instance_points.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            fallback_direction = vh[0]
            fallback_norm = np.linalg.norm(fallback_direction)
            if fallback_norm > 1e-12:
                fallback_direction = fallback_direction / fallback_norm
                fallback_point = instance_points.mean(axis=0)
                instance_features[instance_id] = {
                    's2_vector': fallback_direction.astype(np.float64),
                    's2_point': fallback_point.astype(np.float64),
                    'status': 'fallback_pca',
                    'confidence': 0.3,
                    'point_count': point_count,
                    'plane1_inliers': fail_plane1_inliers,
                    'plane2_inliers': 0,
                    'plane_angle_deg': fail_angle,
                    'method': 'pca',
                    'message': 'second plane estimation failed, used PCA fallback',
                    'selected_distance_threshold': None,
                    'threshold_attempts': attempts,
                    'projection_residual_median': None,
                    'projection_residual_p90': None,
                    'projection_support_ratio': None,
                    'plane_support_balance': None,
                    'suitability_score': None,
                }
                used_fallback = True
        if used_fallback:
            continue

        if best_result is None:
            instance_features[instance_id] = {
                's2_vector': None,
                's2_point': None,
                'status': fail_status,
                'confidence': 0.0,
                'point_count': point_count,
                'plane1_inliers': fail_plane1_inliers,
                'plane2_inliers': fail_plane2_inliers,
                'plane_angle_deg': fail_angle,
                'method': 'none',
                'message': fail_message,
                'selected_distance_threshold': None,
                'threshold_attempts': attempts,
                'projection_residual_median': None,
                'projection_residual_p90': None,
                'projection_support_ratio': None,
                'plane_support_balance': None,
                'suitability_score': None,
            }
            continue

        instance_features[instance_id] = {
            **best_result,
            'confidence': float(best_diag['confidence']) if best_diag else 0.0,
            'selected_distance_threshold': best_threshold,
            'threshold_attempts': attempts,
            'projection_residual_median': (
                float(best_diag['projection_residual_median']) if best_diag else None
            ),
            'projection_residual_p90': (
                float(best_diag['projection_residual_p90']) if best_diag else None
            ),
            'projection_support_ratio': (
                float(best_diag['projection_support_ratio']) if best_diag else None
            ),
            'plane_support_balance': (
                float(best_diag['plane_support_balance']) if best_diag else None
            ),
            'suitability_score': (
                float(best_diag['suitability_score']) if best_diag else None
            ),
        }

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


def project_instance_to_section_2d(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_plane_inliers: int = 10,
    ransac_max_points: Optional[int] = None,
) -> dict:
    """
    Project one beam instance into a canonical 2D section plane (legacy pipeline).

    **Output contract** (stored under ``PointCloud.features['legacy_projection'][id]``):

    **On success** (``ok`` is True):

    - ``status``: ``"ok"``
    - ``points_2d``: (N, 2), ``normals_2d``: (N, 2)
    - ``planes``: pair of Open3D-style plane coefficients (4,)
    - ``vector_3d``, ``left_3d``, ``right_3d``, ``center_3d``: beam-axis frame in 3D
    - ``line_direction``, ``line_origin``: axis used for extent
    - ``transform``: 4x4 rigid transform used to align the section
    - ``proj_dir_0``, ``proj_dir_1``, ``points_plane_projected``: intersection geometry
    - ``inliers_0``, ``inliers_1``: plane RANSAC inlier indices (empty if subsampling was used)
    - ``plane_angle_deg``: angle between dominant planes when available

    **On failure** (``ok`` is False):

    - ``status``: short machine-readable reason (e.g. ``"plane1_failed"``, ``"transform_failed"``).
    """
    points_arr = np.asarray(points, dtype=np.float64)
    normals_arr = np.asarray(normals, dtype=np.float64)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        return {"ok": False, "status": "bad_points_shape"}
    if normals_arr.shape != points_arr.shape:
        return {"ok": False, "status": "normals_shape_mismatch"}
    if len(points_arr) < 2:
        return {"ok": False, "status": "insufficient_points"}
    if not np.isfinite(points_arr).all() or not np.isfinite(normals_arr).all():
        return {"ok": False, "status": "non_finite_input"}

    pts_est = points_arr
    nrm_est = normals_arr
    inliers_stripped = False
    if ransac_max_points is not None:
        cap = int(ransac_max_points)
        if cap > 0 and len(points_arr) > cap:
            rng = np.random.default_rng(0)
            sub_idx = rng.choice(len(points_arr), size=cap, replace=False)
            pts_est = points_arr[sub_idx]
            nrm_est = normals_arr[sub_idx]
            inliers_stripped = True

    estimate = orientation_estimation_s2_legacy(
        pts_est,
        nrm_est,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        min_plane_inliers=min_plane_inliers,
    )
    if not estimate.get("ok", False):
        return {"ok": False, "status": estimate.get("message", "failed")}

    direction = np.asarray(estimate["direction"], dtype=np.float64)
    dn = float(np.linalg.norm(direction))
    if dn < 1e-12:
        return {"ok": False, "status": "degenerate_direction"}
    direction = direction / dn
    origin = np.asarray(estimate["origin"], dtype=np.float64)

    planes = estimate["planes"]
    t = np.dot(points_arr - origin, direction)
    p_on_line = origin + t[:, np.newaxis] * direction
    l_ind = int(np.argmin(t))
    r_ind = int(np.argmax(t))
    left_3d = p_on_line[l_ind]
    right_3d = p_on_line[r_ind]
    span = right_3d - left_3d
    v_norm = float(np.linalg.norm(span))
    if v_norm < 1e-12:
        vector_3d = direction
    else:
        vector_3d = span / v_norm
        if float(np.dot(vector_3d, direction)) < 0.0:
            vector_3d = -vector_3d
    center_3d = (left_3d + right_3d) / 2.0

    try:
        proj_plane = normal_and_point_to_plane(vector_3d, left_3d)
        proj_dir_0, _ = intersecting_line(proj_plane, planes[0])
        proj_dir_1, _ = intersecting_line(proj_plane, planes[1])
        proj_points_plane = points_to_actual_plane(points_arr, vector_3d, left_3d)
    except ValueError as exc:
        return {"ok": False, "status": f"geometry_failed:{exc}"}

    target_left = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    target_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target_right = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    source_left = left_3d + np.asarray(planes[0][:3], dtype=np.float64)
    source_center = left_3d
    source_right = source_center + vector_3d
    source_angle = (source_left, source_center, source_right)
    target_angle = (target_left, target_center, target_right)
    try:
        transform = simplified_transform_lines(source_angle, target_angle)
    except ValueError:
        return {"ok": False, "status": "transform_failed"}

    points_hom = np.hstack((points_arr, np.ones((points_arr.shape[0], 1), dtype=np.float64)))
    normals_hom = np.hstack((normals_arr, np.zeros((normals_arr.shape[0], 1), dtype=np.float64)))
    points_target = (transform @ points_hom.T).T[:, :3]
    normals_target = (transform @ normals_hom.T).T[:, :3]
    normals_2d = normals_target[:, :2]
    n2d_norm = np.linalg.norm(normals_2d, axis=1, keepdims=True)
    n2d_norm[n2d_norm < 1e-12] = 1.0

    in0 = [] if inliers_stripped else estimate["inliers_0"]
    in1 = [] if inliers_stripped else estimate["inliers_1"]

    return {
        "ok": True,
        "status": "ok",
        "planes": planes,
        "inliers_0": in0,
        "inliers_1": in1,
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

    Delegates to :func:`project_instance_to_section_2d` (backward-compatible defaults).
    """
    return project_instance_to_section_2d(
        points,
        normals,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        min_plane_inliers=min_plane_inliers,
        ransac_max_points=None,
    )