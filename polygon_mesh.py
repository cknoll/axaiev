from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import triangulate
# import random
from matplotlib.tri import Triangulation

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()

np.random.seed(1642)


global_attribute_store = defaultdict(dict)


# based on https://github.com/shapely/shapely/issues/1698#issuecomment-1371382145
class MeshPolygon(Polygon):

    __slots__ = Polygon.__slots__

    def __new__(cls, *args, **kwargs) -> "MeshPolygon":
        x = super().__new__(cls, *args, **kwargs)
        x.__class__ = cls
        return x

    def edges(self):
        edge_list = global_attribute_store[id(self)].get("edge_list", None)
        if  edge_list is not None:
            return edge_list
        b = self.boundary.coords
        global_attribute_store[id(self)]["edge_list"] = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
        return global_attribute_store[id(self)]["edge_list"]

    def corners(self):

        corner_list = global_attribute_store[id(self)].get("corner_list", None)
        if corner_list is not None:
            return corner_list

        corners = [Point(c) for c in self.boundary.coords]
        global_attribute_store[id(self)]["corner_list"] = corners
        return global_attribute_store[id(self)]["corner_list"]


# Function to generate a random polygon
def generate_random_polygon(num_points=5):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = np.random.rand(num_points) * 10  # Random radius
    points = [(radius[i] * np.cos(angle[i]), radius[i] * np.sin(angle[i])) for i in range(num_points)]
    return MeshPolygon(points)

# Function to create a regular grid of points within a polygon
def create_grid_within_polygon(polygon, spacing=1):
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)
    grid_points = [(x, y) for x in x_coords for y in y_coords]
    return [point for point in grid_points if polygon.contains(Point(point))]


def triangle_in_polygon(triangle: MeshPolygon, polygon: MeshPolygon) -> tuple[bool, list]:
    # if not polygon.contains(triangle.centroid):
    #     return False, None

    for edge in polygon.edges():
        intersection = triangle.intersection(edge)
        if intersection.is_empty or (intersection in triangle.corners()):
            continue
        if isinstance(intersection, LineString):
            if intersection.length < 1e-8:
                continue
            plot_line_string(intersection, "ro-")
        elif isinstance(intersection, Point):
            plot_line_string(intersection, "go")
        elif isinstance(intersection, MultiLineString):
            for ls in intersection.geoms:
                plot_line_string(ls, "mx")

        else:
            msg = "unexpected intersection type"
            raise TypeError(msg)


    # for corner in polygon.exterior.coords:
    #     if triangle.contains(Point(corner)):
    #         return False
    return True

def plot_line_string(ls: LineString, *args, **kwargs):
    xx, yy = np.array(ls.coords).T
    plt.plot(xx, yy, *args, **kwargs)




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

N = 1
# Generate 10 random polygons
polygons = [generate_random_polygon(np.random.randint(5, 10)) for _ in range(1)]

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
        plt.fill(x_poly, y_poly, alpha=0.5, fc='lightblue', ec='tab:blue', lw=3, label='Polygon')

        # Plot the triangles and color them based on their position relative to the polygon
        for triangle in triangulation.triangles:

            pts = triangulation.x[triangle], triangulation.y[triangle]
            triangle_shape = MeshPolygon(zip(*pts))
            e = triangle_shape.edges()
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
