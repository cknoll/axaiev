import os
import copy
import itertools
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
        for c in [Polygon, MultiPolygon]:
            c.edges = cls.edges
            c.corners = cls.corners
            c.label = property(cls.get_label, cls.set_label)

    def edges(self: Polygon, mode=None):
        ds = ShapelyPolygonMonkeyPatcher.data_store
        key = ("edge_list", mode)
        edge_list = ds[self].get(key, None)
        if edge_list is not None:
            return edge_list
        b = self.boundary.coords
        if mode == "tuples":
            edges = [tuple(b[k : k + 2]) for k in range(len(b) - 1)]
        elif mode == "sorted_tuples":
            se_tup = ShapelyPolygonMonkeyPatcher.sorted_edge_tuple
            edges = [se_tup(b[k : k + 2]) for k in range(len(b) - 1)]
        else:
            edges = [LineString(b[k : k + 2]) for k in range(len(b) - 1)]
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

    def get_label(self):
        ds = ShapelyPolygonMonkeyPatcher.data_store
        key = "label"
        return ds[self].get(key)

    def set_label(self, arg: str):
        ds = ShapelyPolygonMonkeyPatcher.data_store
        key = "label"
        ds[self][key] = arg


ShapelyPolygonMonkeyPatcher.doit()


# this builds upon ShapelyPolygonMonkeyPatcher
def get_poly_labels(pg_list: list[Polygon]):
    return [pg.label for pg in pg_list]


# #############################################################################


# Function to generate a random polygon
def generate_polygon(num_points=5, random=True):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    if random:
        radius = np.random.rand(num_points) * 10  # Random radius
        points = [(radius[i] * np.cos(angle[i]), radius[i] * np.sin(angle[i])) for i in range(num_points)]
    else:
        # just approximate a circle
        radius = 5
        points = [(radius * np.cos(angle[i]), radius * np.sin(angle[i])) for i in range(num_points)]
    return Polygon(points)


def plot_polygons(ax, polygons: list[Polygon], **kwargs):
    for pg in polygons:
        if isinstance(pg, MultiPolygon):
            plot_polygons(ax, pg.geoms, **kwargs)
        plot_polygon_like_obj(ax, pg, **kwargs)


def plot_polygon_like_obj(ax, obj: Polygon, **kwargs):

    if ax is None:
        ax = plt.gca()

    pg_label = kwargs.pop("pg_label", None)

    kwargs_used = dict(fc="lightblue", ec="tab:blue", lw=3, label="Polygon", alpha=0.5)
    kwargs_used.update(kwargs)
    if isinstance(obj, Polygon):
        x_poly, y_poly = obj.exterior.xy
    elif isinstance(obj, LineString):
        x_poly, y_poly = obj.xy
    elif isinstance(obj, MultiPolygon):
        kwargs["pg_label"] = pg_label
        return plot_polygons(ax, obj.geoms, **kwargs)
    else:
        raise TypeError

    ax.fill(x_poly, y_poly, **kwargs_used)
    if pg_label:
        assert isinstance(pg_label, str)
        centroid_xy = np.array(obj.centroid.coords).squeeze()
        plt.text(*centroid_xy, pg_label)


class VoronoiMesher:
    _id_counter = itertools.count(0)

    def __init__(self, main_pg: Polygon, hex=True, key=None):
        self.id = next(self._id_counter)

        if key is None:
            key = str(self.id)
        self.key = key

        self.main_pg = main_pg
        self.hex = hex
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

        dist = (
            np.max((diff_x, diff_y)) * 2
        )  # double the inner mesh-parameter to "ensure" the cell reaches outward
        boundary_pg = self.main_pg.buffer(dist).envelope.boundary  # Create a large rectangle surrounding it

        boundary_points = [
            boundary_pg.interpolate(distance=d, normalized=True) for d in np.linspace(0, 1, 100)[:-1]
        ]
        self.boundary_coords = np.array([[p.x, p.y] for p in boundary_points])

        # Create an array of all points on the boundary and inside the polygon
        self.all_coords = np.concatenate((self.boundary_coords, self.inner_points))

        self.vor = Voronoi(points=self.all_coords)
        self.lines = [
            LineString(self.vor.vertices[line]) for line in self.vor.ridge_vertices if -1 not in line
        ]

        self.voronoi_polys = list(polygonize(self.lines))

        self.areas = []
        pg: Polygon
        self.inner_polys: list[Polygon] = []
        for pg in self.voronoi_polys:
            intersection_pg: Polygon = pg.intersection(self.main_pg)
            if intersection_pg.is_empty:
                continue
            self.inner_polys.append(intersection_pg)
            intersection_pg.label = f"c{len(self.inner_polys)}"
            self.areas.append(intersection_pg.area)

        self.areas = np.array(self.areas)
        self.avg_cell_area = np.average(self.areas)

        # ensure that the mesh cells cover main_pg with suitable precision
        assert abs(self.main_pg.area - np.sum(self.areas)) / self.main_pg.area < 1e-5
        return self.inner_polys

    def create_mesh_for_inner_points(self, num_points):
        minx, miny, maxx, maxy = self.main_pg.bounds

        # note: by `[1:-1]` we reduce the chance of possible boundary points
        nx = int(np.sqrt(num_points)) - 2
        ny = int(np.sqrt(num_points)) + 2
        self._ip_xx = np.linspace(minx, maxx, nx + 1)[1:-1]
        self._ip_yy = np.linspace(miny, maxy, ny + 1)[1:-1]

        square_grid = list(itertools.product(self._ip_xx, self._ip_yy))
        if self.hex:
            # for hexagons we add intermediate points
            dx = np.diff(self._ip_xx[:2])[0]
            dy = np.diff(self._ip_yy[:2])[0]

            xx2 = np.linspace(min(self._ip_xx) - dx / 2, max(self._ip_xx) + dx / 2, nx)
            yy2 = np.linspace(min(self._ip_yy) - dy / 2, max(self._ip_yy) + dy / 2, ny)

            square_grid2 = list(itertools.product(xx2, yy2))

            # for testing:
            if 0:
                plt.scatter(*np.array(list(square_grid)).T)
                plt.scatter(*np.array(list(square_grid2)).T)
                plt.show()
                exit()
            grid = square_grid + square_grid2
        else:
            grid = square_grid

        mesh_points = []
        for point in grid:
            if self.main_pg.contains(Point(point)):
                mesh_points.append(point)

        return np.array(mesh_points)

    def plot_background(self):
        plt.cla()
        plot_polygon_like_obj(None, self.main_pg, alpha=0.1)
        plot_polygons(None, self.inner_polys, fc=None, ec="black", lw=0.5, alpha=0.3)


class DebugProtocolMixin:
    def debug(self, *args):
        if not hasattr(self, "debug_protocol"):
            self.debug_protocol = []

        # store a snapshot of relevant data structures
        new_args = []
        # manually perform medium deep copy
        for arg in args:
            if isinstance(arg, (list, tuple)):
                cls = type(arg)
                res = [copy.copy(elt) for elt in arg]
                new_args.append(cls(res))
            else:
                new_args.append(copy.copy(arg))

        self.debug_protocol.append(new_args)

    def visualize_debug_protocol(self, base_name="poly"):
        from tqdm import tqdm

        self.plot_background()
        n_corners = len(self.main_pg.exterior.coords)
        i = 0
        plt.title(f"N = {n_corners}, #Segments = {self.n_segments} ({i:04d})")

        dirpath = f"img_{self.key}"
        os.makedirs(dirpath, exist_ok=True)

        fpath = os.path.join(dirpath, base_name + "_{:04d}.png")
        plt.savefig(fpath.format(0))

        sec_cntr = 1
        for i, (msg, obj) in tqdm(list(enumerate(self.debug_protocol, start=1))):
            if msg == "new segment":
                sec_cntr += 1

            sec_cntr_diff = 0
            if msg == "start":
                self.plot_background()
                start_pg, current_segments = obj
                plot_polygon_like_obj(None, start_pg, fc="tab:green", ec=None, alpha=1)

                self.plot_labeled_segments(segments=current_segments)

            elif msg == "test":
                plot_polygon_like_obj(None, obj, fc="tab:orange", ec=None, alpha=1)
            elif msg == "pick":
                plot_polygon_like_obj(None, obj, fc="tab:green", ec=None, alpha=1)
            elif msg == "candidates":
                plot_polygons(None, obj, fc="tab:purple", ec=None, alpha=1)
            elif msg == "merge segments":
                plot_polygons(None, obj, fc="tab:cyan", ec=None, alpha=1)
            elif msg == "new merged segment":
                # like new segment but without increasing the counter
                new_segment, current_segments = obj
                self.plot_background()
                self.plot_labeled_segments(segments=current_segments)
            elif msg == "new segment":
                new_segment, current_segments = obj
                self.plot_labeled_segments(segments=current_segments)
                sec_cntr_diff = 1  # increment title not yet

            plt.title(f"N = {n_corners}, Segment {sec_cntr - sec_cntr_diff}/{self.n_segments} ({i:04d})")
            plt.savefig(fpath.format(i))

        # store copies of the last image to hold that frame in the video for 1s
        for j in range(25):
            plt.savefig(fpath.format(i + j + 1))

    def plot_labeled_segments(self, seq: list = None, segments=None):
        """
        This method allows to draw the actual segments or earlier versions.
        """
        if segments is None:
            segments = self.segments
        if seq is None:
            seq = range(len(segments))
        for idx in seq:
            segment_pg = segments[idx]
            plot_polygon_like_obj(
                None, segment_pg, fc="tab:blue", ec="tab:orange", lw=0.5, alpha=1, pg_label=str(idx + 1)
            )


class SegmentCreator(VoronoiMesher, DebugProtocolMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertex_y_map = defaultdict(list)
        self.vertex_y_map_items = None
        self.edge_pg_map = defaultdict(list)
        # map cell to list of neighbors
        self.pg_neighbor_map: dict[Polygon, Polygon] = {}

        # map cell to its assigned segment
        self.cell_segment_map: dict[Polygon, Polygon] = {}
        # map segment to its cells
        self.segment_cell_list_map: dict[Polygon, list] = {}

        self.n_segments: int = None
        self.main_pg_area = self.main_pg.area
        self.target_segment_area: float = None

        self.segments = []
        self.current_partial_segment_list: list = None
        self.partial_segment_pg = None
        self.current_rest_pg = None
        self.candidates_pgs: list = None
        self.already_picked_pgs: dict[Polygon:bool] = {}

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

        while len(self.segments) < self.n_segments - 1:
            start_pg = self._select_min_x_pg(self.vertex_y_map_items[-1][1])
            self.current_rest_pg = self.main_pg
            self.construct_segment(start_pg)
            self._update_vertex_map_items()

        self.construct_last_segment()
        self.visualize_debug_protocol()

    def construct_segment(self, start_pg):

        self.already_picked_pgs[start_pg] = True

        self.current_partial_segment_list = [start_pg]
        self.candidates_pgs = []
        self._add_neighbors_to_candidates(start_pg)

        self.debug("start", (start_pg, self.segments))
        self.debug("candidates", self.candidates_pgs)
        while True:
            break_flag = self._optimization_loop_body()
            if break_flag:
                break
        self.finalize_segment_construction()

    def finalize_segment_construction(self):
        """
        Maybe merge the segment or parts of it with others, then maybe create a new segment
        from the remainder.
        """
        merge_result_list, remainder_parts = self.merge_seg_with_neighbor_if_necessary()
        if len(merge_result_list) > 0:
            assert isinstance(merge_result_list, list)
            for merge_results in merge_result_list:
                too_small_segment, smallest_neighbor, merged_segment = merge_results
                self.debug("merge segments", (too_small_segment, smallest_neighbor))
                self.debug("new merged segment", (merged_segment, self.segments))

            if remainder_parts:
                if len(remainder_parts) >= 1:
                    part_pg_list, cell_list_list = zip(*remainder_parts)
                    self.partial_segment_pg = MultiPolygon(part_pg_list)
                    self.current_partial_segment_list = []
                    for cell_list in cell_list_list:
                        self.current_partial_segment_list.extend(cell_list)
                else:
                    self.partial_segment_pg, self.current_partial_segment_list = remainder_parts

                self._create_new_segment()

        else:
            # there was no merging
            self._create_new_segment()

    def _create_new_segment(self):
        """
        Collect the processed information and form new segment
        """
        self.segments.append(self.partial_segment_pg)
        self.partial_segment_pg.label = f"S{len(self.segments)}"
        self.debug("new segment", (self.partial_segment_pg, self.segments))
        # store the individual cells
        for pg in self.current_partial_segment_list:
            self.cell_segment_map[pg] = self.partial_segment_pg
        self.segment_cell_list_map[self.partial_segment_pg] = self.current_partial_segment_list

    def merge_seg_with_neighbor_if_necessary(self) -> tuple[list, Polygon | MultiPolygon]:
        """
        The recently created segment might be very small or in case of a MultiPolygon contain a small
        sub-segment. Then, this (sub-segment) should be merged with a neighbor.
        """
        processed_segment = self.partial_segment_pg

        if isinstance(processed_segment, MultiPolygon):
            part_seg_list = list(processed_segment.geoms)
            cell_list_list = []
            total_length = 0
            for part_seg in part_seg_list:
                cell_list = self._filter_cell_list(part_seg, self.current_partial_segment_list)
                cell_list_list.append(cell_list)
                total_length += len(cell_list)
            # all cells should occur exactly once
            assert total_length == len(self.current_partial_segment_list)
        else:
            part_seg_list = [processed_segment]
            cell_list_list = [self.current_partial_segment_list]

        # contains information about the merged parts
        res_list = []
        # contains information about the non-merged parts
        remainder_parts = []
        for part_seg_pg, cell_list in zip(part_seg_list, cell_list_list):
            res = self._merge_part_seg_with_neighbor_if_necessary(part_seg_pg, cell_list)
            if res is not None:
                res_list.append(res)
            else:
                remainder_parts.append((part_seg_pg, cell_list))

        return res_list, remainder_parts

    def _filter_cell_list(self, reference_pg, candidate_cells):
        res_cell_list = []
        for pg in candidate_cells:
            if reference_pg.intersects(pg):
                res_cell_list.append(pg)

        return res_cell_list

    def _merge_part_seg_with_neighbor_if_necessary(self, partial_seg_pg: Polygon, cell_list: list[Polygon]):
        assert isinstance(partial_seg_pg, Polygon)

        rel_area = partial_seg_pg.area / self.target_segment_area
        if rel_area > 1.0 / 3:
            return
        # find neighbor-segments
        neighbor_cells = []
        for pg in cell_list:
            neighbor_cells.extend(self.pg_neighbor_map[pg])

        # find smallest neighbor_segment
        min_area = float("inf")
        smallest_neighbor = None
        neighbor_segments = []
        for nb_pg in set(neighbor_cells):
            seg = self.cell_segment_map.get(nb_pg)
            if seg is not None:
                neighbor_segments.append(seg)
                if seg.area < min_area:
                    min_area = seg.area
                    smallest_neighbor = seg

        if smallest_neighbor is None:
            return

        # now merge the two segments:
        idx = self.segments.index(smallest_neighbor)
        merged_segment = unary_union([partial_seg_pg, smallest_neighbor])

        # update cell <-> segment assignments:
        self.segment_cell_list_map[merged_segment] = []
        for pg in self.segment_cell_list_map.pop(smallest_neighbor) + cell_list:
            self.cell_segment_map[pg] = merged_segment
            self.segment_cell_list_map[merged_segment].append(pg)

        # overwrite old segment
        self.segments[idx] = merged_segment
        merged_segment.label = smallest_neighbor.label

        smallest_neighbor.label = f"old_S{idx+1}"

        # return involved segments for debug_protocol
        return (partial_seg_pg, smallest_neighbor, merged_segment)

    def construct_last_segment(self):
        self.current_partial_segment_list = []
        for pg in self.inner_polys:
            if self.already_picked_pgs.get(pg):
                continue
            self.current_partial_segment_list.append(pg)
            self.debug("pick", pg)

        self.partial_segment_pg = unary_union(self.current_partial_segment_list)
        self.partial_segment_pg.label = "S_final"

        self.finalize_segment_construction()

    def _optimization_loop_body(self) -> bool:
        """
        One turn of the optimization loop
        """

        self.partial_segment_pg = unary_union(self.current_partial_segment_list)

        # estimate area after adding the next cell
        est_relative_segment_area = (
            self.partial_segment_pg.area + self.avg_cell_area
        ) / self.target_segment_area

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

        cost = rest_pg.length / rest_pg.area + partial_segment.length / partial_segment.area
        if isinstance(rest_pg, MultiPolygon):
            # penalty for dividing the remaining polygon
            cost *= 10
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


def render_video(dirpath=None, key=None):
    # cmd = f"ffmpeg -y -f image2 -framerate 25 -i {dirpath}/poly_%04d.png -vcodec libx264 -crf 22 video_{key}.mp4"
    cmd = "ffmpeg -framerate 25 -pattern_type glob -i 'img*/poly_*.png' -c:v libx264 -pix_fmt yuv420p all.mp4"
    os.system(cmd)


if __name__ == "__main__":

    np.random.seed(1215)
    N = 15
    exterior = [(0, 0), (4, 0), (4, 4), (2, 3), (0, 4)]
    circle_pg = generate_polygon(num_points=20, random=False)
    # Polygon(exterior, )
    polygons = [
        circle_pg,
    ] + [generate_polygon(np.random.randint(5, 10)) for _ in range(N)]

    # for development select the most difficult of them:
    for main_pg in polygons:

        plt.figure()
        ax1 = plt.subplot(111)

        sc = SegmentCreator(main_pg)
        num_points = np.random.randint(40, 150)
        inner_polys = sc.create_voronoi_mesh_for_polygon(num_points=num_points)
        sc.do_segmentation(n_segments=int(num_points) / 8)

        # useful for debugging the grid:
        if 0:
            plot_polygons(ax1, sc.inner_polys, ec="tab:blue", alpha=0.3)
            plot_polygon_like_obj(ax1, main_pg, alpha=0.5)

            ax1.scatter(*sc.inner_points.T, color="magenta", s=10, alpha=0.5)
            # ax1.scatter(*sc.boundary_coords.T, color='tab:orange', s=10, alpha=0.5)
            ax1.axis("equal")

            plt.show()
    render_video()
