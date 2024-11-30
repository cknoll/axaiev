import os
import copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, LineString
from shapely.ops import polygonize, unary_union
from scipy.spatial import Voronoi, cKDTree

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()

# #############################################################################
# shapely Polygon monkey patch section


# #############################################################################

class ShapelyPolygonMonkeyPatcher:
    """
    # Rationale: Polygon-is not recommended to subclass, see [1].
    # -> We add methods via monkey-patching
    """

    data_store = defaultdict(dict)

    @classmethod
    def doit(cls):
        Polygon.edges = cls.edges
        Polygon.corners = cls.corners

    def edges(self: Polygon, mode=None):
        ds = ShapelyPolygonMonkeyPatcher.data_store
        key = ("edge_list", mode)
        edge_list = ds[self].get(key, None)
        if  edge_list is not None:
            return edge_list
        b = self.boundary.coords
        if mode == "tuples":
            edges = [tuple(b[k:k+2]) for k in range(len(b) - 1)]
        elif mode == "sorted_tuples":
            se_tup = ShapelyPolygonMonkeyPatcher.sorted_edge_tuple
            edges = [se_tup(b[k:k+2]) for k in range(len(b) - 1)]
        else:
            edges = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
        ds[self][key] = edges
        return ds[self][key]

    def corners(self: Polygon, as_np: bool = True):
        ds = ShapelyPolygonMonkeyPatcher.data_store
        key = ("corner_list", as_np)
        corner_list = ds[self].get(key, None)
        if corner_list is not None:
            return corner_list
        if as_np:
            corners = np.array(self.boundary.coords)
        else:
            # list of points
            corners = [Point(c) for c in self.boundary.coords]
        ds[self][key] = corners

        return ds[self][key]

    @staticmethod
    def sorted_edge_tuple(line_points):
        """
        map both (p1, p2) and (p2, p1) to (p1, p2)
        """
        p1, p2 = line_points

        if p1[0] < p2[0]:
            return (p1, p2)
        elif p2[0] < p1[0]:
            return (p2, p1)
        else:
            assert p2[0] == p1[0]
            if p1[1] < p2[1]:
                return (p1, p2)
            else:
                # p2[1] < p1[1] and p2[1] == p1[1]
                return (p2, p1)

ShapelyPolygonMonkeyPatcher.doit()




# #############################################################################

# Function to generate a random polygon
def generate_random_polygon(num_points=5):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = np.random.rand(num_points) * 10  # Random radius
    points = [(radius[i] * np.cos(angle[i]), radius[i] * np.sin(angle[i])) for i in range(num_points)]
    return Polygon(points)


def plot_polygons(ax, polygons: list[Polygon], **kwargs):
    for pg in polygons:
        if isinstance(pg, MultiPolygon):
            plot_polygons(ax, pg.geoms, **kwargs)
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
    elif isinstance(obj, MultiPolygon):
        return plot_polygons(ax, obj.geoms, **kwargs)
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
        self.lines = None
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
        self.lines = [LineString(self.vor.vertices[line]) for line in self.vor.ridge_vertices if -1 not in line]

        self.voronoi_polys = list(polygonize(self.lines))

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
        self.avg_cell_area = np.average(self.areas)

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


class DebugProtocolMixin:
    def debug(self, *args):
        if not hasattr(self, "debug_protocol"):
            self.debug_protocol = []

        # store a snapshot of relevant data structures
        new_args = [copy.copy(arg) for arg in args]
        self.debug_protocol.append(new_args)

    def visualize_debug_protocol(self, base_name="poly"):
        from tqdm import tqdm
        def plot_background():
            plt.cla()
            plot_polygon_like_obj(None, self.main_pg, alpha=0.1)
            plot_polygons(None, self.inner_polys, fc=None, ec="black", lw=0.5, alpha=0.3)

        plot_background()
        n_corners = len(self.main_pg.exterior.coords)
        i = 0
        plt.title(f"N = {n_corners}, #Segments = {self.n_segments} ({i:04d})")

        fpath = os.path.join("img", base_name + "_{:04d}.png")
        plt.savefig(fpath.format(0))

        sec_cntr = 1
        for i, (msg, obj) in tqdm(list(enumerate(self.debug_protocol, start=1))):

            if msg == "start":
                plot_background()
                plot_polygon_like_obj(None, obj, fc="tab:green", ec=None, alpha=1)

                for idx in range(sec_cntr - 1):
                    segment_pg = self.segments[idx]
                    plot_polygon_like_obj(None, segment_pg, fc="tab:blue", ec="tab:orange", lw=0.5, alpha=1)

            elif msg == "test":
                plot_polygon_like_obj(None, obj, fc="tab:orange", ec=None, alpha=1)
            elif msg == "pick":
                plot_polygon_like_obj(None, obj, fc="tab:green", ec=None, alpha=1)
            elif msg == "candidates":
                plot_polygons(None, obj, fc="tab:purple", ec=None, alpha=1)
            elif msg == "new segment":
                plot_polygon_like_obj(None, obj, fc="tab:blue", ec="tab:orange", lw=0.5, alpha=1)
                sec_cntr += 1

            plt.title(f"N = {n_corners}, Segment {sec_cntr}/{self.n_segments} ({i:04d})")
            plt.savefig(fpath.format(i))


class SegmentCreator(VoronoiMesher, DebugProtocolMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertex_y_map = defaultdict(list)
        self.vertex_y_map_items = None
        self.edge_pg_map = defaultdict(list)
        self.pg_neighbor_map = {}

        self.n_segments: int = None
        self.main_pg_area = self.main_pg.area
        self.target_segment_area: float = None

        self.segments = []
        self.current_partial_segment_list: list = None
        self.partial_segment_pg = None
        self.current_rest_pg = None
        self.candidates_pgs: list = None
        self.already_picked_pgs: dict[Polygon: bool] = {}


    def _fill_neighbor_map(self):

        # self.inner_polys = self.inner_polys[:2]

        all_corners = []
        for pg in self.inner_polys:
            all_corners.extend(pg.corners(as_np=True))

        for pg in self.inner_polys:
            for edge in pg.edges(mode="sorted_tuples"):
                self.edge_pg_map[edge].append(pg)

        for pg in self.inner_polys:
            potential_neighbors = []
            for edge in pg.edges(mode="sorted_tuples"):
                # all polygons which share an edge
                potential_neighbors.extend(self.edge_pg_map[edge])
            potential_neighbors = set(potential_neighbors)
            potential_neighbors.remove(pg)
            self.pg_neighbor_map[pg] = potential_neighbors

    def _fill_vertex_map(self):

        pg: Polygon
        for pg in self.inner_polys:
            for y in pg.boundary.coords.xy[1]:
                self.vertex_y_map[y].append(pg)

        self.vertex_y_map_items = list(self.vertex_y_map.items())

        # sort by keys
        self.vertex_y_map_items.sort()
        self.vertex_y_map.clear()
        self.vertex_y_map.update(self.vertex_y_map_items)

    def _update_vertex_map_items(self):
        """
        Iterate over nested lists and then pop those elements which have become invalid.
        Sorting is not changed
        """
        pop_idcs = []
        for idx, (y, pg_list) in enumerate(self.vertex_y_map_items):
            pop_pg_idcs = []
            for pg_idx, pg in enumerate(pg_list):
                if self.already_picked_pgs.get(pg):
                    pop_pg_idcs.append(pg_idx)

            # pop those polygons which have been picked
            for pg_idx in pop_pg_idcs[::-1]:
                pg_list.pop(pg_idx)

            # register empty lists for popping
            if not pg_list:
                pop_idcs.append(idx)

        # pop those items where the lists have become empty
        for idx in pop_idcs[::-1]:
            self.vertex_y_map_items.pop(idx)


    def do_segmentation(self, n_segments: int):
        """
        Decompose self.main_pg into n segments (consisting of suitable mesh cells)
        """
        self.n_segments = n_segments
        self.target_segment_area = self.main_pg_area / n_segments
        self._fill_vertex_map()
        self._fill_neighbor_map()

        while len(self.segments) < self.n_segments:
            start_pg = self._select_min_x_pg(self.vertex_y_map_items[-1][1])
            self.current_rest_pg = self.main_pg
            self.construct_segment(start_pg)
            self._update_vertex_map_items()

        self.visualize_debug_protocol()

        # plot_polygon_like_obj(None, start_pg, fc="tab:green", ec="red", alpha=0.9)

    def construct_segment(self, start_pg):

        self.already_picked_pgs[start_pg] = True

        self.current_partial_segment_list = [start_pg]
        self.candidates_pgs = []
        self._add_neighbors_to_candidates(start_pg)

        self.debug("start", start_pg)
        self.debug("candidates", self.candidates_pgs)
        while True:
            break_flag = self._optimization_loop_body()
            if break_flag:
                break
        self.debug("new segment", self.partial_segment_pg)
        self.segments.append(self.partial_segment_pg)

    def _optimization_loop_body(self) -> bool:
        """
        One turn of the optimization loop
        """

        self.partial_segment_pg = unary_union(self.current_partial_segment_list)

        # estimate area after adding the next cell
        est_relative_segment_area = (self.partial_segment_pg.area + self.avg_cell_area)/self.target_segment_area

        if est_relative_segment_area > 1:
            # this segment would be too big with additional cell ()
            return True

        if not self.candidates_pgs:
            # no more candidates to choose from
            return True

        best_cost = float("inf")
        best_idx = None
        for idx, candidate_pg in enumerate(self.candidates_pgs):
            candidate_partial_segment = unary_union(self.current_partial_segment_list + [candidate_pg])
            candidate_cost = self.cost_func(candidate_partial_segment)

            self.debug("test", candidate_pg)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_idx = idx

        picked_candidate_pg = self.candidates_pgs.pop(best_idx)
        self.already_picked_pgs[picked_candidate_pg] = True
        self.current_partial_segment_list.append(picked_candidate_pg)
        self.current_rest_pg = self.current_rest_pg.difference(picked_candidate_pg)
        self.debug("pick", picked_candidate_pg)

        self._add_neighbors_to_candidates(picked_candidate_pg)

        self.debug("candidates", self.candidates_pgs)

        # do not break the loop
        return False

    def _add_neighbors_to_candidates(self, picked_pg):
        for pg in self.pg_neighbor_map[picked_pg]:
            if self.already_picked_pgs.get(pg) or pg in self.candidates_pgs:
                continue
            self.candidates_pgs.append(pg)

    def cost_func(self, partial_segment: Polygon):

        rest_pg: Polygon = self.current_rest_pg.difference(partial_segment)

        cost = rest_pg.length/rest_pg.area + partial_segment.length/partial_segment.area
        return cost

    def _select_min_x_pg(self, pg_list: list[Polygon]) -> Polygon:
        """
        From a sequence of polygons (which might be those with max y  vertices) select
        the one with smallest x value.
        """
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
    for main_pg in polygons[2:3]:

        plt.figure()
        ax1 = plt.subplot(111)

        sc = SegmentCreator(main_pg)
        inner_polys = sc.create_voronoi_mesh_for_polygon(num_points=50)
        sc.do_segmentation(n_segments=10)

        plot_polygons(ax1, sc.inner_polys, ec="tab:blue", alpha=0.3)

        plot_polygon_like_obj(ax1, main_pg, alpha=0.5)

        ax1.scatter(*sc.inner_points.T, color='magenta', s=10, alpha=0.5)
        ax1.scatter(*sc.boundary_coords.T, color='tab:orange', s=10, alpha=0.5)
        ax1.axis("equal")

        if 0:
            ax2 = plt.subplot(122)
            ax2.hist(sc.areas)

    plt.show()
