"""Tools for finding stellar clusters in Gaia data (RWTH Aachen PPT outreach project)."""

from .clustering import find_one_cluster
from .coordinates import add_distance, convert_to_galactic
from .files import load_fits, load_parquet, save_parquet
from .isochrones import add_absolute_magnitude, get_isochrone, load_isochrones
from .plotting import (
    plot_3d,
    plot_and_save,
    plot_dataframe,
    plot_ellipse,
    plot_histogram,
)
from .selection import select_ellipse, select_range

__version__ = "0.1.0"

__all__ = [
    "add_absolute_magnitude",
    "add_distance",
    "convert_to_galactic",
    "find_one_cluster",
    "get_isochrone",
    "load_fits",
    "load_isochrones",
    "load_parquet",
    "plot_3d",
    "plot_and_save",
    "plot_dataframe",
    "plot_ellipse",
    "plot_histogram",
    "save_parquet",
    "select_ellipse",
    "select_range",
]
