from shapely.geometry import Polygon
from shapely.ops import split
from shapely import unary_union
import numpy as np
from ipydex import IPS, activate_ips_on_exception


activate_ips_on_exception()

def subdivide_polygon0(polygon, N):
    # Calculate the total area of the polygon
    total_area = polygon.area

    # Calculate target area for each segment
    target_area = total_area / N

    # Initialize variables for storing segments
    segments = []
    remaining_polygon = polygon

    # Loop until we have N segments
    while len(segments) < N and remaining_polygon.area > 0:
        # Start with a small slice of the remaining polygon
        slice_area = 0
        slice_polygon = None

        # Create slices until we reach or exceed target_area
        for angle in np.linspace(0, 2 * np.pi, 100):
            point_on_edge = remaining_polygon.exterior.interpolate(angle)
            new_segment = Polygon([remaining_polygon.centroid, point_on_edge])

            if slice_polygon is None:
                slice_polygon = new_segment
            else:
                slice_polygon = slice_polygon.union(new_segment)

            slice_area = slice_polygon.area

            if slice_area >= target_area:
                break

        # Store the valid segment and update remaining polygon
        segments.append(slice_polygon)
        remaining_polygon = remaining_polygon.difference(slice_polygon)

    return segments



def subdivide_polygon(polygon, N):
    # Calculate total area and target area for each segment
    total_area = polygon.area
    target_area = total_area / N

    # Initialize variables for storing segments
    segments = []
    remaining_polygon = polygon

    # Loop until we have N segments or no area left
    while len(segments) < N and remaining_polygon.area > 0:
        # Start with an empty segment
        slice_polygon = None

        # Create slices until we reach or exceed target_area
        for i in range(len(remaining_polygon.exterior.coords) - 1):
            p1 = remaining_polygon.exterior.coords[i]
            p2 = remaining_polygon.exterior.coords[i + 1]

            # Create a triangle between centroid and edge points
            triangle = Polygon([remaining_polygon.centroid, p1, p2])

            if slice_polygon is None:
                slice_polygon = triangle
            else:
                slice_polygon = unary_union([slice_polygon, triangle])

            # Check if we have reached or exceeded target_area
            if slice_polygon.area >= target_area:
                break

        # Store the valid segment and update remaining polygon
        segments.append(slice_polygon)
        remaining_polygon = remaining_polygon.difference(slice_polygon)

    return segments




# Example usage:
input_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])  # A square polygon
N = 10
segments = subdivide_polygon(input_polygon, N)

for i, seg in enumerate(segments):
    print(f"Segment {i + 1}: Area = {seg.area}, Length = {seg.length}")



import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# Define additional input polygons
polygons = [
    Polygon([(0, 0), (6, 0), (6, 3), (0, 3)]),  # Rectangle
    Polygon([(0, 0), (4, 2), (2, 4), (0, 2)]),  # Quadrilateral
    Polygon([(0, 0), (1, 2), (3, 1), (2, -1)]) , # Irregular shape
    Polygon([(0, 0), (5, 5), (10, 0), (5, -5)]), # Rhombus
    Polygon([(1, 1), (3, 1), (4, 3), (2, 4), (0, 3)]) # Star-like shape
]

# Number of segments for subdivision
N = 10

# Create a function to visualize the polygons and their segments
def visualize_polygons(original_polygons):

    for polygon in original_polygons:
        fig, ax = plt.subplots(figsize=(12, 8))
        segments = subdivide_polygon(polygon, N)

        # Plot original polygon
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2, label='Original Polygon')

        # Plot segments
        for seg in segments:
            x_seg, y_seg = seg.exterior.xy
            ax.fill(x_seg, y_seg, alpha=0.5)

    ax.set_title('Subdivided Polygons into Equal Area Segments')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_aspect('equal')
    plt.legend()
    plt.grid()
    plt.show()

# Call the visualization function with the defined polygons
visualize_polygons(polygons)
