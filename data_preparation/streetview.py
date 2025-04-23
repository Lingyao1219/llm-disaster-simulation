import os
import requests
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple, Union, List
import io
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageService(Enum):
    """Enum for available image services"""
    GOOGLE = "google"
    MAPILLARY = "mapillary"
    OPENSTREETCAM = "openstreetcam"
    ALL = "all"

class StreetViewDownloader:
    """Class for downloading street view images from different services."""
    
    def __init__(self, secret_file: Optional[str] = 'secrets.txt', 
                 google_api_key: Optional[str] = None, 
                 mapillary_token: Optional[str] = None):
        """
        Initialize with either a secret file path or direct API keys/tokens.
        
        Args:
            secret_file (str, optional): Path to secrets file
            google_api_key (str, optional): Direct Google Maps API key
            mapillary_token (str, optional): Direct Mapillary API token
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        # Initialize API keys
        self.google_api_key = google_api_key
        self.mapillary_token = mapillary_token
        
        # If direct keys aren't provided, try to load from secret file
        if not (google_api_key and mapillary_token) and secret_file:
            self._load_secrets(secret_file)

    def _load_secrets(self, secret_file: str) -> None:
        """
        Load API keys and tokens from a secret file.
        """
        try:
            if not os.path.exists(secret_file):
                raise FileNotFoundError(f"Secret file not found: {secret_file}")
                
            with open(secret_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    try:
                        key, value = [part.strip() for part in line.split(',', 1)]
                        if key == "MAPILLARY_TOKEN" and not self.mapillary_token:
                            self.mapillary_token = value
                        elif key == "GOOGLE_API" and not self.google_api_key:
                            self.google_api_key = value
                    except ValueError:
                        logger.warning(f"Skipping invalid line in secrets file: {line}")
                        
            # Verify that we have the required keys
            if not self.google_api_key:
                logger.warning("Google API key not found in secrets file")
            if not self.mapillary_token:
                logger.warning("Mapillary token not found in secrets file")
                
        except Exception as e:
            logger.error(f"Error loading secrets file: {str(e)}")
            raise

    def _is_valid_image(self, image_content: bytes) -> bool:
        """
        Check if image content is valid and not a "no imagery" placeholder.
        
        Args:
            image_content: Raw image bytes
            
        Returns:
            bool: True if valid image, False if placeholder or invalid
        """
        try:
            # Read image data
            img = Image.open(io.BytesIO(image_content))
            
            # Convert to grayscale and calculate variance
            img_gray = img.convert('L')
            img_array = np.array(img_gray)
            variance = np.var(img_array)
            
            # If variance is very low, it's likely a plain image (placeholder)
            if variance < 500:
                #logger.info(f"Low variance image detected (variance: {variance}), likely a placeholder")
                return False
                
            # Additional check: Average brightness - placeholders often have high avg brightness
            brightness = np.mean(img_array)
            if brightness > 220:  # Very bright/white image
                #logger.info(f"Very bright image detected (brightness: {brightness}), likely a placeholder")
                return False
                
            return True
            
        except Exception as e:
            #logger.warning(f"Error checking image validity: {str(e)}")
            return False

    def _save_image(self, content: bytes, prefix: str, latitude: float, longitude: float, 
                   extra_id: Optional[str] = None) -> str:
        """
        Save image content to file with standardized naming.
        First validates that the image isn't a placeholder.
        
        Args:
            content: Raw image bytes
            prefix: Prefix for the filename
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            extra_id: Optional additional identifier
            
        Returns:
            str: Path to saved file or empty string if invalid
        """
        # Check if this is a valid image (not a placeholder)
        if not self._is_valid_image(content):
            #logger.info(f"Skipping placeholder image for coordinates: {latitude}, {longitude}")
            return ""
            
        filename_parts = [prefix, f"{latitude}_{longitude}"]
        if extra_id:
            filename_parts.insert(-1, extra_id)
        filename = f"{'_'.join(filename_parts)}.jpg"
        
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        filepath = os.path.join('images', filename)
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        return filepath

    def download_google_streetview(
        self,
        latitude: float,
        longitude: float,
        size: str = '600x300',
        heading: Optional[float] = None,
        pitch: Optional[float] = None,
        fov: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Download Street View image from Google Maps API.
        """
        if not self.google_api_key:
            return False, "Google Maps API key not provided"

        base_url = "https://maps.googleapis.com/maps/api/streetview"
        
        params = {
            'size': size,
            'location': f"{latitude},{longitude}",
            'key': self.google_api_key,
            'source': 'outdoor'
        }
        
        if heading is not None:
            params['heading'] = heading
        if pitch is not None:
            params['pitch'] = pitch
        if fov is not None:
            params['fov'] = fov
            
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Check if the response content is a valid image (not a placeholder)
            filename = self._save_image(
                response.content, 
                'google_streetview', 
                latitude, 
                longitude
            )
            
            # If filename is empty, it means it was a placeholder
            if not filename:
                return False, "No valid imagery available at this location"
                
            #logger.info(f"Successfully downloaded Google Street View image: {filename}")
            return True, filename
            
        except Exception as e:
            error_msg = f"Error downloading Google Street View image: {str(e)}"
            #logger.error(error_msg)
            return False, error_msg

    def download_mapillary_image(self, latitude: float, longitude: float) -> Tuple[bool, str]:
        """
        Download image from Mapillary for given coordinates.
        """
        if not self.mapillary_token:
            return False, "Mapillary token not provided"

        search_url = "https://graph.mapillary.com/images"
        
        params = {
            'access_token': self.mapillary_token,
            'fields': 'id,thumb_2048_url',
            'bbox': f'{longitude-0.001},{latitude-0.001},{longitude+0.001},{latitude+0.001}'
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                image = data['data'][0]
                thumb_url = image['thumb_2048_url']
                img_response = requests.get(thumb_url)
                img_response.raise_for_status()
                
                filename = self._save_image(
                    img_response.content,
                    'mapillary',
                    latitude,
                    longitude,
                    image['id']
                )
                
                # If filename is empty, it means it was a placeholder or invalid image
                if not filename:
                    return False, "No valid imagery available at this location"
                    
                return True, filename
            
            return False, "No images found at these coordinates"
                
        except Exception as e:
            return False, str(e)

    def download_openstreetcam_photo(self, latitude: float, longitude: float, radius: int = 100) -> Tuple[bool, str]:
        """
        Download photo from OpenStreetCam for given coordinates.
        """
        url = "https://api.openstreetcam.org/2.0/photo"
        
        params = {
            'lat': latitude,
            'lng': longitude,
            'radius': radius,
            'page_size': 1
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if (data.get('result') and 
                data['result'].get('data') and 
                len(data['result']['data']) > 0):
                photo_data = data['result']['data'][0]
                photo_url = photo_data.get('imageLthUrl')
                
                if photo_url:
                    photo_response = requests.get(photo_url, headers=self.headers)
                    photo_response.raise_for_status()
                    
                    filename = self._save_image(
                        photo_response.content,
                        'openstreetcam',
                        latitude,
                        longitude
                    )
                    
                    # If filename is empty, it means it was a placeholder or invalid image
                    if not filename:
                        return False, "No valid imagery available at this location"
                        
                    return True, filename
                
                return False, "No image URL found in the data"
            
            return False, "No photos found in the response data"
            
        except Exception as e:
            return False, str(e)

    def download_images(
        self,
        latitude: float,
        longitude: float,
        services: Union[ImageService, List[ImageService]] = ImageService.ALL,
        **kwargs
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Download images from specified services.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            services (Union[ImageService, List[ImageService]]): Service(s) to use
            **kwargs: Additional arguments for specific services:
                - size (str): Image size for Google Street View
                - heading (float): Camera heading for Google Street View
                - pitch (float): Camera pitch for Google Street View
                - fov (int): Field of view for Google Street View
                - radius (int): Search radius for OpenStreetCam
        
        Returns:
            Dict[str, Tuple[bool, str]]: Dictionary of results for each service
        """
        results = {}
        
        # Convert single service to list
        if isinstance(services, ImageService):
            services = [services]
            
        # If ALL is specified, use all services
        if ImageService.ALL in services:
            services = [ImageService.GOOGLE, ImageService.MAPILLARY, ImageService.OPENSTREETCAM]
            
        # Process each requested service
        for service in services:
            if service == ImageService.GOOGLE:
                results['google'] = self.download_google_streetview(
                    latitude=latitude,
                    longitude=longitude,
                    size=kwargs.get('size', '600x300'),
                    heading=kwargs.get('heading'),
                    pitch=kwargs.get('pitch'),
                    fov=kwargs.get('fov')
                )
            elif service == ImageService.MAPILLARY:
                results['mapillary'] = self.download_mapillary_image(
                    latitude=latitude,
                    longitude=longitude
                )
            elif service == ImageService.OPENSTREETCAM:
                results['openstreetcam'] = self.download_openstreetcam_photo(
                    latitude=latitude,
                    longitude=longitude,
                    radius=kwargs.get('radius', 100)
                )
                
        return results