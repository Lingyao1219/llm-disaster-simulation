import osmnx as ox
import geopandas as gpd
import time
import random
from functools import wraps
from tqdm import tqdm
from requests.exceptions import RequestException, Timeout

# Global rate limiter
def rate_limit(calls_per_minute=15):
    """Decorator to limit call frequency"""
    interval = 60.0 / calls_per_minute
    last_call = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result
        return wrapper
    return decorator

# Apply the rate limit decorator
@rate_limit(calls_per_minute=15)  # Adjust this number based on OSM's limits
def get_building_info(lat, lon, radius=100, max_retries=2, sleep_time=2):
    """
    Get buildings near a location using OpenStreetMap data with retry mechanism.
    
    Args:
        lat: Latitude
        lon: Longitude
        radius: Search radius in meters (default: 100)
        max_retries: Maximum number of retry attempts (default: 2)
        sleep_time: Fixed sleep time between retries in seconds (default: 2)
    
    Returns:
        GeoDataFrame with building information or None if not found
    """
    center_point = (lat, lon)
    tags = {"building": True}

    for attempt in range(max_retries):
        try:
            # Attempt to fetch building features from OSM
            gdf = ox.features.features_from_point(center_point=center_point, tags=tags, dist=radius)

            if gdf is None or gdf.empty:
                print(f"No building information found for ({lat}, {lon})")
                return None

            # Ensure expected columns exist
            for col in ['height', 'building:material', 'seismic:resistance']:
                if col not in gdf.columns:
                    gdf[col] = None
            
            # Extract numerical heights
            gdf['height_m'] = gdf['height'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
            
            # Fill missing seismic resistance values
            gdf['seismic:resistance'] = gdf['seismic:resistance'].fillna("Unknown")

            return gdf

        except ox._errors.InsufficientResponseError as e:
            # Only retry for InsufficientResponseError (API/connection issues)
            if attempt < max_retries - 1:  # If not the last attempt
                print(f"Attempt {attempt + 1}/{max_retries} failed due to insufficient response: {str(e)}")
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to get building info after {max_retries} attempts")
                return None
                
        except Exception as e:
            # For other exceptions including "No matching features"
            error_msg = str(e)
            if "No matching features" in error_msg:
                # Don't retry for no matching features
                print(f"No matching features for ({lat}, {lon}): {error_msg}")
                return None
            else:
                # Other errors might be temporary, so retry
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Failed to get building info after {max_retries} attempts")
                    return None


def describe_buildings(gdf, radius=100):
    """
    Create a human-readable description of buildings.
    
    Args:
        gdf: GeoDataFrame with building information
        radius: Search radius used (default: 100)
    
    Returns:
        Text description of buildings
    """
    if gdf is None or gdf.empty:
        return "No building information can be found from the system."

    try:
        description = []

        # Count buildings
        total = len(gdf)
        description.append(f"A total of {total} buildings are found within a {radius}-meter radius")

        # Building types
        if 'building' in gdf.columns:
            building_types = gdf['building'].value_counts()

            # Replace 'yes' with 'general building' for display
            building_types.index = building_types.index.str.replace('^yes$', 'general building', regex=True)

            if not building_types.empty:
                types_desc = ", ".join(f"{btype} ({count})" for btype, count in building_types.items())
                description.append(f", including types such as: {types_desc}")

        # Heights
        if 'height_m' in gdf.columns:
            known_heights = gdf[gdf['height_m'].notna()]
            if not known_heights.empty:
                min_h = known_heights['height_m'].min()
                max_h = known_heights['height_m'].max()
                if min_h == max_h:
                    description.append(f". Building height is approximately {min_h:.1f} meters")
                else:
                    description.append(f". Building heights range from {min_h:.1f} to {max_h:.1f} meters")

        # Materials
        if 'building:material' in gdf.columns:
            material_counts = gdf['building:material'].dropna().value_counts()
            if not material_counts.empty:
                mats = ", ".join(f"{mat} ({cnt})" for mat, cnt in material_counts.items())
                description.append(f". Building materials include: {mats}")

        # Seismic resistance
        if 'seismic:resistance' in gdf.columns:
            seismic_known = gdf[gdf['seismic:resistance'].notna() & (gdf['seismic:resistance'] != 'Unknown')]
            if not seismic_known.empty:
                types = seismic_known['seismic:resistance'].value_counts()
                desc = ", ".join(f"{k} ({v})" for k, v in types.items())
                description.append(f". Some buildings have seismic resistance information: {desc}")
            else:
                description.append(" ")

        description.append(".")
        return "".join(description)
    
    except Exception as e:
        print(f"Error generating building description: {str(e)}")
        return "Building information is not available."


if __name__ == "__main__":
    radius = 100
    latitude = 36.7387	
    longitude = -119.7831

    result = get_building_info(latitude, longitude, radius)
    summary = describe_buildings(result, radius)
    print(summary)