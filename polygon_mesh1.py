import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.geometry import MultiPolygon
from shapely.ops import triangulate
import random

# Function to generate random polygons
def generate_random_polygon(num_vertices):
    angles = np.sort(np.random.rand(num_vertices) * 2 * np.pi)
    radius = np.random.rand(num_vertices) + 1  # Random radius between 1 and 2
    points = [(radius[i] * np.cos(angles[i]), radius[i] * np.sin(angles[i])) for i in range(num_vertices)]
    return Polygon(points)

# Function to create a mesh grid within a polygon
def create_mesh(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.linspace(minx, maxx, int(np.sqrt(num_points)))
    y_coords = np.linspace(miny, maxy, int(np.sqrt(num_points)))

    mesh_points = []
    for x in x_coords:
        for y in y_coords:
            point = (x, y)
            if polygon.contains(Point(point)):
                mesh_points.append(point)

    return mesh_points

# Generate 10 random polygons
polygons = [generate_random_polygon(random.randint(3, 10)) for _ in range(10)]

# Create a plot

for polygon in polygons:
    plt.figure(figsize=(12, 12))
    # Create mesh for each polygon
    mesh_points = create_mesh(polygon, 100)

    # Plot the polygon
    x, y = polygon.exterior.xy
    plt.fill(x, y, alpha=0.5, fc='lightblue', ec='black')

    # Plot the mesh points
    mesh_x, mesh_y = zip(*mesh_points)
    plt.scatter(mesh_x, mesh_y, s=10, color='red')

    # Draw lines of the mesh (optional)
    for i in range(len(mesh_x)):
        plt.plot([mesh_x[i], mesh_x[i]], [min(y), max(y)], color='gray', alpha=0.5)
        plt.plot([min(x), max(x)], [mesh_y[i], mesh_y[i]], color='gray', alpha=0.5)

    plt.title('Random Polygons with Mesh Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.grid()
plt.show()
