import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, LineString
from shapely.ops import polygonize

from scipy.spatial import Voronoi

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()


def create_mesh(polygon: Polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.linspace(minx, maxx, int(np.sqrt(num_points)) + 1)
    y_coords = np.linspace(miny, maxy, int(np.sqrt(num_points)) + 1)

    mesh_points = []
    # note: by `[1:-1]` we reduce the chance of possible boundary points
    for x in x_coords[1:-1]:
        for y in y_coords[1:-1]:
            point = (x, y)
            if polygon.contains(Point(point)):
                mesh_points.append(point)

    return np.array(mesh_points)


def plot_polygons(polygons: list[Polygon], **kwargs):
    for pg in polygons:
        if isinstance(pg, MultiPolygon):
            plot_polygons(pg.geoms)
        plot_polygon_like_obj(pg, **kwargs)

def plot_polygon_like_obj(obj: Polygon, **kwargs):

    kwargs_used = dict(fc='lightblue', ec='tab:blue', lw=3, label='Polygon')
    kwargs_used.update(kwargs)
    if isinstance(obj, Polygon):
        x_poly, y_poly = obj.exterior.xy
    elif isinstance(obj, LineString):
        x_poly, y_poly = obj.xy
    else:
        raise TypeError

    plt.fill(x_poly, y_poly, alpha=0.5, **kwargs_used)

# Example usage
# Create an arbitrary polygon (non-convex in this case)
exterior = [(0, 0), (4, 0), (4, 4), (2, 3), (0, 4)]
hole = [] # [(1, 1), (2, 1), (2, 2), (1, 2)]
main_pg = Polygon(exterior, [hole])

bound = main_pg.buffer(2).envelope.boundary #Create a large rectangle surrounding it


boundary_points = [bound.interpolate(distance=d, normalized=True) for d in np.linspace(0, 1, 100)[:-1]]
boundary_coords = np.array([[p.x, p.y] for p in boundary_points])

inner_points = create_mesh(main_pg, 100)

# all_coords = inner_points

# Create an array of all points on the boundary and inside the polygon
all_coords = np.concatenate((boundary_coords, inner_points))

vor = Voronoi(points=all_coords)
lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

polys = list(polygonize(lines))

pg: Polygon
inner_polys = []
for pg in polys:
    intersection_pg: Polygon = pg.intersection(main_pg)
    if intersection_pg.is_empty:
        continue
    inner_polys.append(intersection_pg)

IPS()

# plot_polygons(lines, ec="tab:green")
plot_polygons(inner_polys, ec="tab:green")
# IPS()

plot_polygon_like_obj(main_pg)
# plot_polygon_like_obj(bound)

plt.scatter(*inner_points.T, color='magenta', s=10, alpha=0.5)
plt.scatter(*boundary_coords.T, color='tab:orange', s=10, alpha=0.5)

plt.show()
exit()
# Generate Voronoi mesh
N = 100  # Number of inner points
voronoi_cells = generate_voronoi_mesh(main_pg, N)

IPS()

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
plot_polygon_like_obj(main_pg, ec='red', fc='none', lw=2)
plot_polygons(voronoi_cells)
# GeoSeries(polygon).plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)
ax.set_title(f'Voronoi-like Mesh with {N} Inner Points')
plt.axis('equal')
plt.show()
