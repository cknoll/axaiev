import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay

# Function to create random polygons
def create_random_polygon(num_points=5):
    points = np.random.rand(num_points, 2) * 10  # Random points in a 10x10 area
    return Polygon(points)

# Function to triangulate the polygon and plot the mesh
def plot_triangulated_polygon(polygon):
    # Create a mesh of points within the polygon
    minx, miny, maxx, maxy = polygon.bounds
    x_points = np.linspace(minx, maxx, int(np.sqrt(100)))  # N=100 points
    y_points = np.linspace(miny, maxy, int(np.sqrt(100)))
    grid_points = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1, 2)

    # Filter points that are inside the polygon
    inside_points = [point for point in grid_points if polygon.contains(Point(point))]

    # Perform Delaunay triangulation on the inside points
    if len(inside_points) >= 3:  # Need at least 3 points for triangulation
        tri = Delaunay(inside_points)

        # Plotting the triangulated mesh
        plt.figure()
        plt.triplot(np.array(inside_points)[:,0], np.array(inside_points)[:,1], tri.simplices)
        plt.plot(np.array(inside_points)[:,0], np.array(inside_points)[:,1], 'o')
        plt.fill(*polygon.exterior.xy, alpha=0.5, fc='lightgray', ec='black')
        plt.title('Triangular Mesh for Polygon')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.xlim(minx-1, maxx+1)
        plt.ylim(miny-1, maxy+1)
        plt.gca().set_aspect('equal')
        plt.show()

# Main execution block
for _ in range(10):
    random_polygon = create_random_polygon(num_points=np.random.randint(3, 8))  # Random number of vertices between 3 and 7
    plot_triangulated_polygon(random_polygon)
