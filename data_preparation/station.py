import math
import json
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth specified in decimal degrees.
    """
    # Convert decimal degrees to radians
    # First ensure all inputs are floats
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def find_closest_station(target_lat, target_lng, df):
    """
    Find the closest station from the given DataFrame based on latitude and longitude.
    
    Parameters:
    target_lat (float): Target latitude
    target_lng (float): Target longitude
    df (DataFrame): DataFrame with station data
    
    Returns:
    dict: The closest station's information
    """
    closest_station = None
    min_distance = float('inf')
    
    for idx, station in df.iterrows():
        try:
            lat = float(station['latitude'])
            lng = float(station['longitude'])
            
            distance = haversine_distance(target_lat, target_lng, lat, lng)
            
            if distance < min_distance:
                min_distance = distance
                closest_station = {
                    'index': idx,
                    'latitude': lat,
                    'longitude': lng,
                    'intensity': float(station['Intensity']),
                    'vs30': float(station['Vs30']),
                    'nresp': int(station['Nresp']),
                    'distance_km': min_distance
                }
        except (ValueError, TypeError) as e:
            # Skip entries that can't be converted to float
            print(f"Warning: Skipping entry {idx} due to error: {e}")
            continue
    
    return closest_station

def load_station_data(file_path):
    """
    Load station data from JSON file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract required data
    records = []
    for feature in data.get("features", []):  # Iterate through each feature
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # Check if network is "DYFI"
        if properties.get("network") == "DYFI":
            try:
                latitude = float(geometry.get("coordinates", [None, None])[1])  # Extract latitude
                longitude = float(geometry.get("coordinates", [None, None])[0])  # Extract longitude
                intensity = float(properties.get("intensity", 0))  # Extract intensity
                nresp = int(properties.get("nresp", 0))  # Extract nresp
                vs30 = float(properties.get("vs30", 0))

                # Ensure all values exist and are valid before adding to list
                if None not in (latitude, longitude, intensity, vs30, nresp):
                    records.append([latitude, longitude, intensity, vs30, nresp])
            except (ValueError, TypeError):
                # Skip entries that can't be converted to float/int
                continue

    # Convert to pandas DataFrame with explicit data types
    df = pd.DataFrame(records, columns=["latitude", "longitude", "Intensity", "Vs30", "Nresp"])
    
    # Ensure numeric data types
    df = df.astype({
        "latitude": float,
        "longitude": float,
        "Intensity": float,
        "Vs30": float,
        "Nresp": int
    })
    
    return df

if __name__ == "__main__":
    # Load data from JSON file
    file_path = '2014_napa/stationlist.json'
    try:
        stations_df = load_station_data(file_path)
        
        # Print info about loaded data
        print(f"Loaded {len(stations_df)} stations from {file_path}")
        print(stations_df.head())
        
        # Example usage - find closest station to a given location
        target_lat = 33.416167981408584	
        target_lng = -117.599027
        
        closest = find_closest_station(target_lat, target_lng, stations_df)
        
        if closest:
            print(f"\nClosest station to ({target_lat}, {target_lng}):")
            print(f"Location: ({closest['latitude']}, {closest['longitude']})")
            print(f"Distance: {closest['distance_km']:.2f} km")
            print(f"Intensity: {closest['intensity']}")
            print(f"Vs30: {closest['vs30']}")
            print(f"Nresp: {closest['nresp']}")
        else:
            print("No valid stations found to compare.")
    except Exception as e:
        print(f"Error processing file: {e}")