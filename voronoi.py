import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, LineString
from shapely.ops import polygonize

from scipy.spatial import Voronoi

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()


# Function to generate a random polygon
def generate_random_polygon(num_points=5):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = np.random.rand(num_points) * 10  # Random radius
    points = [(radius[i] * np.cos(angle[i]), radius[i] * np.sin(angle[i])) for i in range(num_points)]
    return Polygon(points)


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


class VoronoiMesher:
    def __init__(self, main_pg: Polygon):
        self.main_pg = main_pg
        self.inner_polys = None
        self.boundary_coords = None
        self.inner_points = None
        self.vor = None
        self.voronoi_polys = None
        self._ip_xx = None
        self._ip_yy = None
        self.xmin, self.ymin, self.xmax, self.ymax = self.main_pg.bounds

    def create_voronoi_mesh_for_polygon(self, num_points=100) -> list[Polygon]:

        self.inner_points = self.create_mesh_for_inner_points(num_points)
        # diff_x = self.xmax - self.xmin
        # diff_y = self.ymax - self.ymin

        diff_x = np.diff(self._ip_xx[:2])[0]
        diff_y = np.diff(self._ip_yy[:2])[0]

        dist = np.max((diff_x, diff_y))*2  # double the inner mesh-parameter to "ensure" the cell reaches outward
        boundary_pg = self.main_pg.buffer(dist).envelope.boundary #Create a large rectangle surrounding it

        boundary_points = [boundary_pg.interpolate(distance=d, normalized=True) for d in np.linspace(0, 1, 100)[:-1]]
        self.boundary_coords = np.array([[p.x, p.y] for p in boundary_points])

        # Create an array of all points on the boundary and inside the polygon
        self.all_coords = np.concatenate((self.boundary_coords, self.inner_points))

        self.vor = Voronoi(points=self.all_coords)
        lines = [LineString(self.vor.vertices[line]) for line in self.vor.ridge_vertices if -1 not in line]

        self.voronoi_polys = list(polygonize(lines))

        pg: Polygon
        self.inner_polys: list[Polygon] = []
        for pg in self.voronoi_polys:
            intersection_pg: Polygon = pg.intersection(self.main_pg)
            if intersection_pg.is_empty:
                continue
            self.inner_polys.append(intersection_pg)

        return self.inner_polys

    def create_mesh_for_inner_points(self, num_points):
        minx, miny, maxx, maxy = self.main_pg.bounds

        # note: by `[1:-1]` we reduce the chance of possible boundary points
        self._ip_xx = np.linspace(minx, maxx, int(np.sqrt(num_points)) + 1)[1:-1]
        self._ip_yy = np.linspace(miny, maxy, int(np.sqrt(num_points)) + 1)[1:-1]

        mesh_points = []
        for x in self._ip_xx:
            for y in self._ip_yy:
                point = (x, y)
                if self.main_pg.contains(Point(point)):
                    mesh_points.append(point)

        return np.array(mesh_points)


if __name__ == "__main__":


    np.random.seed(1642)
    N = 5
    exterior = [(0, 0), (4, 0), (4, 4), (2, 3), (0, 4)]
    polygons = [Polygon(exterior, )] + [generate_random_polygon(np.random.randint(5, 10)) for _ in range(N)]

    for main_pg in polygons:

        vm = VoronoiMesher(main_pg)
        inner_polys = vm.create_voronoi_mesh_for_polygon()
        plt.figure()

        # plot_polygons(lines, ec="tab:green")
        plot_polygons(inner_polys, ec="tab:green")
        # IPS()

        plot_polygon_like_obj(main_pg)
        # plot_polygon_like_obj(bound)

        plt.scatter(*vm.inner_points.T, color='magenta', s=10, alpha=0.5)
        plt.scatter(*vm.boundary_coords.T, color='tab:orange', s=10, alpha=0.5)

    plt.show()
