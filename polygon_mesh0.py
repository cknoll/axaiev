import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
import random

def generate_random_polygon(num_vertices):
    """Generate a random polygon with a specified number of vertices."""
    angle = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    radius = np.random.rand(num_vertices) * 10  # Random radius for vertices
    points = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radius, angle)]
    return Polygon(points)

def create_mesh(polygon, num_points):
    """Create a mesh of random points within the given polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    while len(points) < num_points:
        # Generate random point
        p = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(p):
            points.append(p)

    return points

def visualize_polygons_and_mesh(polygons, meshes):
    """Visualize the polygons and their respective meshes."""

    for polygon, mesh in zip(polygons, meshes):
        plt.figure(figsize=(12, 8))
        x_poly, y_poly = polygon.exterior.xy
        plt.fill(x_poly, y_poly, alpha=0.5)

        x_mesh = [p.x for p in mesh]
        y_mesh = [p.y for p in mesh]
        plt.scatter(x_mesh, y_mesh, color='red', s=10)

        plt.title('Random Polygons and Mesh Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
    plt.show()

# Parameters
num_polygons = 10
num_vertices_per_polygon = 5
num_points_per_mesh = 50

# Generate random polygons and their meshes
polygons = [generate_random_polygon(num_vertices_per_polygon) for _ in range(num_polygons)]
meshes = [create_mesh(polygon, num_points_per_mesh) for polygon in polygons]

# Visualize the result
visualize_polygons_and_mesh(polygons, meshes)
