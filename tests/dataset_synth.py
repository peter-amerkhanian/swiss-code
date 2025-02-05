import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

def simulate_gdf(num_points):
    # Define the bounding box for random points (xmin, xmax, ymin, ymax)
    bounding_box = (-120, -119, 35, 36)  # Example: Somewhere in California
    # Number of points to generate
    # Generate random coordinates within the bounding box
    x_coords = np.random.uniform(bounding_box[0], bounding_box[1], num_points)
    y_coords = np.random.uniform(bounding_box[2], bounding_box[3], num_points)

    # Create Point geometries
    geometry = [Point(x, y) for x, y in zip(x_coords, y_coords)]

    # Add some random attributes (e.g., "value")
    data = {
        "id": range(1, num_points + 1),
        "value": np.random.randint(0, 100, num_points),
    }

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=geometry)
    # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf