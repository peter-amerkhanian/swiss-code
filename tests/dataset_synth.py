import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

def simulate_df(num_transactions=100):
    customer_ids = np.random.randint(1000, 5000, num_transactions)
    transaction_amounts = np.round(np.random.uniform(5, 500, num_transactions), 2)
    payment_methods = np.random.choice(["Credit Card", "Debit Card", "PayPal", "Cash"], num_transactions)
    categories = np.random.choice(["Electronics", "Clothing", "Groceries", "Entertainment"], num_transactions)
    transaction_dates = pd.date_range(start="2024-01-01", periods=num_transactions, freq="D")
    data = {
        "transaction_id": range(1, num_transactions + 1),
        "customer_id": customer_ids,
        "amount": transaction_amounts,
        "payment_method": payment_methods,
        "category": categories,
        "date": transaction_dates
    }

    df = pd.DataFrame(data)
    return df


def simulate_gdf(num_points=100):
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