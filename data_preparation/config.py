# Input Files
DYFI_DATA = '2019_ridgecrest/2019_ridgecrest_DYFI.csv'
STATION_DATA = '2019_ridgecrest/stationlist.json'
SOCIOECONOMIC_DATA = 'nhgis_shape/cbg_information.csv'
ZCTA_SHAPEFILE = 'nhgis_shape/US_zcta_2019.shp'
CBG_SHAPEFILE = 'nhgis_shape/US_blck_grp_2019_84.shp'
VS30_FILE = 'vs30_mosaic.tif'

# Output Files and Directories
OUTPUT_IMAGES_DIR = '2019_ridgecrest_images'
OUTPUT_IMAGES_CSV = '2019_ridgecrest_samples.csv'
CHECKPOINT_FILE = '2019_ridgecrest_checkpoint.pkl'
OUTPUT_SAMPLES_CSV = '2019_ridgecrest_samples_prompt.csv'
RAG_CHECKPOINT_FILE = '2019_ridgecrest_rag_checkpoint.pkl'
OUTPUT_RAG_SAMPLES_CSV = '2019_ridgecrest_rag_samples_prompt.csv'

# Earthquake Parameters
EARTHQUAKE_PARAMETERS = {
    "lat": 35.770,
    "lng": -117.599,
    "place": "Ridgecrest, CA",
    "magnitude": 7.1,
    "depth": 8.0,
}

# Sampling Parameters
NUM_SAMPLES = 100  # Number of sampled ZIP code
POINTS_PER_ZIP = 50  # Number of Street View samples per ZIP code
MAX_ATTEMPTS = 1000   # Maximum attempts to find valid points with Street View per ZIP code; setting a large number could result in significant cost