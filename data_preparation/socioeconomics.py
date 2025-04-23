import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

socioeconomic_df = None
cbg_gdf = None


def load_data(cbg_shapefile_path='cbg_shapefile/US_blck_grp_2019_84.shp', 
              socioeconomic_path='cbg_information.csv', 
              force_reload=False):
    """
    Loads the required datasets into global variables if they haven't been loaded yet.
    
    Parameters:
    cbg_shapefile_path (str): Path to the Census Block Group shapefile
    socioeconomic_path (str): Path to the socioeconomic CSV file
    force_reload (bool): If True, reloads the data even if already loaded
    
    Returns:
    tuple: (socioeconomic_df, cbg_gdf) - The loaded dataframes
    
    Raises:
    FileNotFoundError: If the data files cannot be found
    """
    global socioeconomic_df, cbg_gdf
    
    if socioeconomic_df is None or force_reload:
        print(f"Loading socioeconomic data from {socioeconomic_path}...")
        try:
            socioeconomic_df = pd.read_csv(socioeconomic_path)
            print(f"Loaded socioeconomic data with {len(socioeconomic_df)} rows")
        except FileNotFoundError:
            raise FileNotFoundError(f"Socioeconomic data file not found at {socioeconomic_path}")

    if cbg_gdf is None or force_reload:
        print(f"Loading Census Block Group shapefile from {cbg_shapefile_path}...")
        try:
            cbg_gdf = gpd.read_file(cbg_shapefile_path)
            cbg_gdf = cbg_gdf.to_crs(epsg=4326)  # Ensure coordinate system is in WGS 84
            print(f"Loaded shapefile with {len(cbg_gdf)} block groups")
        except FileNotFoundError:
            raise FileNotFoundError(f"Shapefile not found at {cbg_shapefile_path}")
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            raise

    return socioeconomic_df, cbg_gdf


def get_sociodemographics(latitude, longitude, socioeconomic_df, cbg_gdf):
    """
    Returns sociodemographic factors for a given latitude and longitude.
    
    Parameters:
    latitude (float): Latitude coordinate
    longitude (float): Longitude coordinate
    
    Returns:
    dict: Dictionary containing the requested sociodemographic factors
    """
    
    # Create a point geometry from the input coordinates
    point = Point(longitude, latitude)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    
    # Perform spatial join to find which CBG contains the point
    joined_gdf = gpd.sjoin(point_gdf, cbg_gdf, how="left", predicate="within")
    
    # If no match is found, return empty results
    if joined_gdf.empty or joined_gdf['GEOID'].isna().all():
        return {
            "status": "no_cbg",
            "message": "No Census Block Group found for the provided coordinates."
        }
    
    # Find matching socioeconomic data
    geoid = joined_gdf['GEOID'].iloc[0]
    geoid_str = str(int(geoid))
    socio_row = socioeconomic_df[socioeconomic_df['BGFIPS'].astype(str) == geoid_str]
    
    # If no match
    if socio_row.empty:
        return {
            "status": "no_socio",
            "message": f"Census Block Group {geoid_str} found, but no socioeconomic data."
        }
    
    result = {
        "status": "success",
        'State_Name': socio_row['State_Name'].iloc[0],
        'County_Name': socio_row['County_Name'].iloc[0],
        'BGFIPS': socio_row['BGFIPS'].iloc[0],
        'Total_Population': socio_row['Total_Population'].iloc[0],
        'Population_Density': socio_row['Population_Density'].iloc[0],
        'Urbanized_Areas_Population_R': socio_row['Urbanized_Areas_Population_R'].iloc[0],
        'Over_65_R': socio_row['Over_65_R'].iloc[0],
        'Median_income': socio_row['Median_income'].iloc[0],
        'GINI': socio_row['GINI'].iloc[0],
        'Education_Degree_R': socio_row['Education_Degree_R'].iloc[0],
        'Health_care_R': socio_row['Health_care_R'].iloc[0],
        'Unemployed_R': socio_row['Unemployed_R'].iloc[0]
    }
    
    return result