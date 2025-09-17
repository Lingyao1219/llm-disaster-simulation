import rasterio
import rasterio.windows

def get_raster_value(src, latitude, longitude):
    """
    Get the raster value at a specific latitude and longitude from an already opened raster.
    """
    # Transform latitude and longitude to pixel coordinates
    row, col = src.index(longitude, latitude)
    
    # Check if the pixel is within bounds
    if 0 <= row < src.height and 0 <= col < src.width:
        window = rasterio.windows.Window(col, row, 1, 1)
        data = src.read(1, window=window)
        return data[0][0]
    else:
        return None


if __name__ == "__main__":
    tif_path = "vs30_mosaic.tif"
    src = rasterio.open(tif_path)
    lat, lon = 35.59426706634436, -117.6945756
    value = get_raster_value(src, lat, lon)
    print(f"VS30 value at lat={lat}, lon={lon}: {value}")