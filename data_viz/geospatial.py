import geopandas as gpd
import contextily as cx
from scipy import ndimage
import matplotlib as mpl
import matplotlib.colors as colors
from shapely.geometry import Polygon
import os
from typing import Tuple
import numpy as np

os.environ['USE_PYGEOS'] = '0'

def get_bounds(gdf: gpd.GeoDataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Get the bounding box of a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]:
            A tuple containing two tuples:
            - The first tuple represents the min and max x-values (longitude).
            - The second tuple represents the min and max y-values (latitude).
    """
    miny = gdf.geometry.bounds.min()['miny']
    minx = gdf.geometry.bounds.min()['minx']
    maxy = gdf.geometry.bounds.max()['maxy']
    maxx = gdf.geometry.bounds.max()['maxx']
    return (minx, maxx), (miny, maxy)

def build_choropleth(gdf: gpd.GeoDataFrame,
                  bins: int,
                  mask: int = 0,
                  name: str = "value") -> gpd.GeoDataFrame:
    """
    Build a 2D histogram and represent it as a GeoDataFrame with polygonal bins.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame with point geometries.
        bins (int): The number of bins for the 2D histogram in both x and y directions.
        mask (int, optional): The value to mask in the histogram. Defaults to 0.
        name (str, optional): The name of the column to store the histogram values. Defaults to "value".

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing polygon geometries (bins) and the corresponding values.
    """
    x = gdf.geometry.x
    y = gdf.geometry.y
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    H[H == mask] = np.nan
    # Step 2: Create the grid of points
    polygons = []
    values = []
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            # Create the polygon for each bin
            polygons.append(Polygon([
                (xedges[i], yedges[j]),
                (xedges[i+1], yedges[j]),
                (xedges[i+1], yedges[j+1]),
                (xedges[i], yedges[j+1])
            ]))
            values.append(H[i, j])
    # Step 3: Convert to GeoDataFrame
    gdf_hist = gpd.GeoDataFrame({'geometry': polygons, name: values})
    return gdf_hist

def zoom_district(ax: mpl.axes.Axes,
                  district: str,
                  gdf: gpd.GeoDataFrame,
                  column: str,
                  zoom: float = 1) -> None:
    """
    Adjust the plot's x and y limits to zoom into a specific district.

    Args:
        ax (mpl.axes.Axes): The matplotlib axis to apply the zoom.
        district (str): The name or value of the district to zoom into.
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the geometry data.
        column (str): The column in the GeoDataFrame that contains district identifiers.
        zoom (float, optional): The zoom factor to apply. Defaults to 1.
    """
    gdf_subset = gdf[gdf[column] == district]
    miny = gdf_subset.geometry.bounds.min()['miny'] - .01 * zoom
    minx = gdf_subset.geometry.bounds.min()['minx'] - .01 * zoom
    maxy = gdf_subset.geometry.bounds.max()['maxy'] + .01 * zoom
    maxx = gdf_subset.geometry.bounds.max()['maxx'] + .01 * zoom
    ax.set(xlim = (minx, maxx), ylim = (miny, maxy))

def clean_map(ax: mpl.axes.Axes) -> None:
    """
    Clean a matplotlib map by removing ticks, labels, and spines.

    Args:
        ax (mpl.axes.Axes): The matplotlib axis to clean.
    """
    ax.tick_params(axis='both', which='both', bottom=False,
                   left=False, labelbottom=False, labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def label_polygon(ax: mpl.axes.Axes,
                  label: str,
                  row: gpd.GeoSeries,
                  idx: int,
                  iter_df: gpd.GeoDataFrame,
                  textsize: int = 10) -> None:
    """
    Label the centroid of a polygon or multi-polygon geometry on a map.

    Args:
        ax (mpl.axes.Axes): The matplotlib axis on which to place the label.
        label (str): The text to place at the centroid of the polygon.
        row (gpd.GeoSeries): The row containing the polygon geometry.
        idx (int): The index of the row in the iterating DataFrame.
        iter_df (gpd.GeoDataFrame): The GeoDataFrame being iterated over.
        textsize (int, optional): The font size of the label. Defaults to 10.
    """
    if row.geometry.geom_type == 'MultiPolygon':
        for polygon in iter_df['geometry'][idx].geoms:
            ax.annotate(text=label,
                        size=textsize,
                        xy=(polygon.centroid.x, polygon.centroid.y),
                        horizontalalignment='center', verticalalignment='center')
    else:
        ax.annotate(text=label,
                    size=textsize,
                    xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    horizontalalignment='center', verticalalignment='center')
