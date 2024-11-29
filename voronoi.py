import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from shapely.ops import voronoi_diagram

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()


def generate_voronoi_mesh(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(point):
            points.append(point)

    multi_point = MultiPoint(points)
    voronoi = voronoi_diagram(multi_point, envelope=polygon)

    clipped_cells = [cell.intersection(polygon) for cell in voronoi.geoms]
    return clipped_cells

def plot_polygons(polygons: list[Polygon], **kwargs):
    for pg in polygons:
        if isinstance(pg, MultiPolygon):
            plot_polygons(pg.geoms)
        plot_polygon(pg, **kwargs)

def plot_polygon(polygon: Polygon, **kwargs):

    kwargs_used = dict(fc='lightblue', ec='tab:blue', lw=3, label='Polygon')
    kwargs_used.update(kwargs)
    x_poly, y_poly = polygon.exterior.xy
    plt.fill(x_poly, y_poly, alpha=0.5, **kwargs_used)

# Example usage
# Create an arbitrary polygon (non-convex in this case)
exterior = [(0, 0), (4, 0), (4, 4), (2, 3), (0, 4)]
hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
polygon = Polygon(exterior, [hole])

# Generate Voronoi mesh
N = 100  # Number of inner points
voronoi_cells = generate_voronoi_mesh(polygon, N)

IPS()

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
plot_polygon(polygon, ec='red', fc='none', lw=2)
plot_polygons(voronoi_cells)
# GeoSeries(polygon).plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)
ax.set_title(f'Voronoi-like Mesh with {N} Inner Points')
plt.axis('equal')
plt.show()
