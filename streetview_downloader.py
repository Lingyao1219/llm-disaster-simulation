#!/usr/bin/env python3
"""
Street View Image Downloader
This module provides functions to download street-level images from Mapillary 
and OpenStreetCam APIs based on geographic coordinates.
"""

import requests
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreetViewDownloader:
    """Class for downloading street view images from different services."""
    
    def __init__(self, mapillary_token: Optional[str] = None):
        """Initialize with optional Mapillary token."""
        self.mapillary_token = mapillary_token
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

    def download_mapillary_image(self, latitude: float, longitude: float) -> Tuple[bool, str]:
        """
        Download image from Mapillary for given coordinates.
        Returns: (success_status, filename or error_message)
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
                filename = f"mapillary_{latitude}_{longitude}_{image['id']}.jpg"
                with open(filename, 'wb') as f:
                    f.write(img_response.content)
                return True, filename
            
            return False, "No images found at these coordinates"
                
        except Exception as e:
            return False, str(e)


    def download_openstreetcam_photo(self, latitude: float, longitude: float, radius: int = 100) -> Tuple[bool, str]:
        """
        Download photo from OpenStreetCam for given coordinates.
        Returns: (success_status, filename or error_message)
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"openstreetcam_{latitude}_{longitude}_{timestamp}.jpg"
                    
                    with open(filename, 'wb') as f:
                        f.write(photo_response.content)
                    return True, filename
                
                return False, "No image URL found in the data"
            
            return False, "No photos found in the response data"
            
        except Exception as e:
            return False, str(e)