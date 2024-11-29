from collections import defaultdict
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


def plot_polygons(ax, polygons: list[Polygon], **kwargs):
    for pg in polygons:
        if isinstance(pg, MultiPolygon):
            plot_polygons(ax, pg.geoms)
        plot_polygon_like_obj(ax, pg, **kwargs)


def plot_polygon_like_obj(ax, obj: Polygon, **kwargs):

    if ax is None:
        ax = plt.gca()

    kwargs_used = dict(fc='lightblue', ec='tab:blue', lw=3, label='Polygon', alpha=0.5)
    kwargs_used.update(kwargs)
    if isinstance(obj, Polygon):
        x_poly, y_poly = obj.exterior.xy
    elif isinstance(obj, LineString):
        x_poly, y_poly = obj.xy
    else:
        raise TypeError

    ax.fill(x_poly, y_poly, **kwargs_used)


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
        self.areas = None

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

        self.areas = []
        pg: Polygon
        self.inner_polys: list[Polygon] = []
        for pg in self.voronoi_polys:
            intersection_pg: Polygon = pg.intersection(self.main_pg)
            if intersection_pg.is_empty:
                continue
            self.inner_polys.append(intersection_pg)
            self.areas.append(intersection_pg.area)

        self.areas = np.array(self.areas)

        # ensure that the mesh cells cover main_pg with suitable precision
        assert abs(self.main_pg.area - np.sum(self.areas)) / self.main_pg.area < 1e-5
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


class SegmentCreator(VoronoiMesher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertex_y_map = defaultdict(list)
        self.vertex_y_map_items = None

    def fill_vertex_map(self):

        pg: Polygon
        for pg in self.inner_polys:
            for y in pg.boundary.coords.xy[1]:
                self.vertex_y_map[y].append(pg)

        self.vertex_y_map_items = list(self.vertex_y_map.items())

        # sort by keys
        self.vertex_y_map_items.sort()
        self.vertex_y_map.clear()
        self.vertex_y_map.update(self.vertex_y_map_items)

    def do_segmentation(self, n_segments: int):
        """
        Decompose self.main_pg into n segments (consisting of suitable mesh cells)
        """
        self.fill_vertex_map()

        # min_y_vrtx_idx = np.argmin(self.vor.vertices[:, 1])
        max_y =  np.max(self.vor.vertices[:, 1])

        start_pg = self._select_min_x_pg(self.vertex_y_map_items[-1][1])

        plot_polygon_like_obj(None, start_pg, fc="tab:green", alpha=0.9)

        # select the cell with miny, minx coordinates
        # IPS()

    def _select_min_x_pg(self, pg_list: list[Polygon]) -> Polygon:
        xmin = float("inf")
        pg_res: Polygon = None


        for pg in pg_list:
            x_pg_min = np.min(pg.boundary.coords.xy[0])
            if x_pg_min < xmin:
                xmin = x_pg_min
                pg_res = pg

        return pg_res







if __name__ == "__main__":


    np.random.seed(1642)
    N = 5
    exterior = [(0, 0), (4, 0), (4, 4), (2, 3), (0, 4)]
    polygons = [Polygon(exterior, )] + [generate_random_polygon(np.random.randint(5, 10)) for _ in range(N)]

    # for development select the most difficult of them:
    for main_pg in polygons:

        plt.figure()
        ax1 = plt.subplot(111)

        sc = SegmentCreator(main_pg)
        inner_polys = sc.create_voronoi_mesh_for_polygon(num_points=500)
        sc.do_segmentation(n_segments=10)

        plot_polygons(ax1, inner_polys, ec="tab:blue", alpha=0.3)

        plot_polygon_like_obj(ax1, main_pg, alpha=0.5)

        ax1.scatter(*sc.inner_points.T, color='magenta', s=10, alpha=0.5)
        ax1.scatter(*sc.boundary_coords.T, color='tab:orange', s=10, alpha=0.5)
        ax1.axis("equal")

        if 0:
            ax2 = plt.subplot(122)
            ax2.hist(sc.areas)

    plt.show()
