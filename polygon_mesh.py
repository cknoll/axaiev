import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate
# import random
from matplotlib.tri import Triangulation

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()

np.random.seed = 1629

# Function to generate a random polygon
def generate_random_polygon(num_points=5):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = np.random.rand(num_points) * 10  # Random radius
    points = [(radius[i] * np.cos(angle[i]), radius[i] * np.sin(angle[i])) for i in range(num_points)]
    return Polygon(points)

# Function to create a regular grid of points within a polygon
def create_grid_within_polygon(polygon, spacing=1):
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)
    grid_points = [(x, y) for x in x_coords for y in y_coords]
    return [point for point in grid_points if polygon.contains(Point(point))]


def triangle_in_polygon(triangle, polygon) -> bool:
    return polygon.contains(triangle.centroid)


def get_polygon_contour_points(polygon, n_points=10):
    # Get exterior points of the polygon
    exterior_coords = np.array(polygon.exterior.coords)

    # Calculate the total length of the polygon's perimeter
    perimeter_length = polygon.length

    # Calculate distances between points
    distances = np.linspace(0, perimeter_length, n_points)

    # Interpolating points along the contour
    contour_points = []

    for distance in distances:
        point = polygon.exterior.interpolate(distance)
        contour_points.append((point.x, point.y))

    return contour_points

# Generate 10 random polygons
polygons = [generate_random_polygon(np.random.randint(5, 10)) for _ in range(10)]

# Create a figure for each polygon and its triangulation
for i, polygon in enumerate(polygons):
    # Create a grid of points inside the polygon
    grid_points = create_grid_within_polygon(polygon)

    contour_points = get_polygon_contour_points(polygon)

    grid_points.extend(contour_points)


    # Perform Delaunay triangulation on the grid points
    if len(grid_points) > 2:
        triangulation = Triangulation(*zip(*grid_points))

        # Create a new figure
        plt.figure(figsize=(6, 6))

        # Plot the original polygon
        x_poly, y_poly = polygon.exterior.xy
        plt.fill(x_poly, y_poly, alpha=0.5, fc='lightblue', ec='black', label='Polygon')

        # Plot the triangles and color them based on their position relative to the polygon
        for triangle in triangulation.triangles:
            pts = triangulation.x[triangle], triangulation.y[triangle]
            triangle_shape = Polygon(zip(*pts))
            if triangle_in_polygon(triangle_shape, polygon):
                plt.fill(*pts, alpha=0.3, fc='orange', ec='black')  # Inside triangles in orange
            else:
                # plt.fill(*pts, alpha=0.3, fc='red', ec='black')  # Outside triangles in red
                pass

        # Set plot limits and title
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)
        plt.title(f'Triangular Mesh for Polygon {i + 1}')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.legend()

# Show all figures
plt.show()
