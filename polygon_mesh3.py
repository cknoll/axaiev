import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate
import random

# Function to generate a random polygon
def generate_random_polygon(num_points=5):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = np.random.rand(num_points) * 10  # Random radius
    points = [(radius[i] * np.cos(angle[i]), radius[i] * np.sin(angle[i])) for i in range(num_points)]
    return Polygon(points)

# Generate 10 random polygons
polygons = [generate_random_polygon(random.randint(3, 8)) for _ in range(10)]

# Create a figure for each polygon and its triangulation
for i, polygon in enumerate(polygons):
    # Perform Delaunay triangulation
    triangles = triangulate(polygon)

    # Create a new figure
    plt.figure(figsize=(6, 6))

    # Plot the original polygon
    x, y = polygon.exterior.xy
    plt.fill(x, y, alpha=0.5, fc='lightblue', ec='tab:blue', lw=3, label='Polygon')

    # Plot the triangles
    for triangle in triangles:
        x_tri, y_tri = triangle.exterior.xy
        plt.fill(x_tri, y_tri, alpha=0.3, fc='orange', ec='black')

    # Set plot limits and title
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.title(f'Triangular Mesh for Polygon {i + 1}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.legend()

# Show all figures
plt.show()
