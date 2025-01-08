import os
import glob
from itertools import cycle

import numpy as np
import cv2
from shapely import Polygon, affinity
import matplotlib.pyplot as plt

from ipydex import IPS

from .voronoi import plot_polygon_like_obj

pjoin = os.path.join


def polygon_based_mask_processing(mask_dir: str):
    """
    Assume a directory structure like:
    - <mask_dir>
        - 0001
            - 003151.png
            - 003152.png
        - 0002
            - ...
    """

    pattern = pjoin(mask_dir, "*", "*.png")
    mask_files = glob.glob(pattern)

    color_cycler = cycle(("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"))

    for i in range(8):
        img = load_img(mask_files[i])
        pg = get_polygon_from_mask_img(img)
        plot_polygon_like_obj(
            None,
            pg,
            mode="plot",
            ls="",
            marker=".",
            markersize=10 * (0.5 + 1.5 / (i + 1)),
            color=next(color_cycler),
        )

    plt.show()


def normalize_polygon(polygon):
    minx, miny, _, _ = polygon.bounds
    return affinity.translate(polygon, xoff=-minx, yoff=-miny)


def get_polygon_from_mask_img(image: np.ndarray):

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate the contour
    epsilon = 0.001 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Convert to Shapely polygon
    pg0 = Polygon(approx.reshape(-1, 2))

    pg1 = normalize_polygon(pg0)

    # edges = np.array(pg.edges(mode="tuples")).astype(int)
    return pg1


def load_img(fpath, rgb=False):

    assert os.path.isfile(fpath), f"FileNotFound: {fpath}"
    image1  = cv2.imread(fpath)

    if rgb:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    else:
        # use BGR, do not convert
        pass

    return image1
