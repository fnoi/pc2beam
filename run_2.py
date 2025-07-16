import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pc2beam.data import PointCloud
data_path = project_root / "data" / "test_points.txt"
pc = PointCloud.from_txt(data_path)
print(f"loaded point cloud with {pc.size} points")
print(f"has normals: {pc.has_normals}")
print(f"has instance labels: {pc.has_instances}")

# fig = pc.visualize(
#     color_by="instance",
#     show_normals=True,
#     normal_length=0.1,
#     title="point cloud - colored by instance labels",
#     max_points=2000
# )
# fig.show()

pc.calculate_s1()
# fig = pc.visualize_with_supernormals(
#     color_by="instance",
#     normal_length=0.1,
#     title="point cloud - colored by instance labels",
#     max_points=2000
# )
# fig.show()

pc.calculate_s2()

pc.project_to_beam()

# Create and display XY scatter plot of projected points
print("\nCreating XY scatter plot of projected points...")
fig_xy = pc.visualize_projected_points_xy(
    title="Projected Points - XY View",
    color_by="instance",
    point_size=4,
    max_points=2000
)
fig_xy.show()

# Also create 3D beam projection visualization for comparison
print("\nCreating 3D beam projection visualization...")
fig_3d = pc.visualize_beam_projection(
    title="Beam Projection - 3D View",
    color_by="instance",
    max_points=2000
)
fig_3d.show()

print("\nVisualizations completed!")
