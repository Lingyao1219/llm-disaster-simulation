# Earthquake Data from USGS ShakeMap

## ShakeMap Instructions
For detailed information about ShakeMap, visit the [USGS ShakeMap documentation](https://code.usgs.gov/ghsc/esi/shakemap).

## Included Earthquakes
- **ci38457511**: [M 7.1, 2019, Ridgecrest Earthquake Sequence](https://earthquake.usgs.gov/earthquakes/eventpage/ci38457511/executive)  
- **ci14607652**: [M 7.2, 2010, The 2010 Sierra El Mayor, B.C., Mexico Earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/ci14607652/executive)  
- **nc21323712**: [M 6.5, 2003, 10 km NE of San Simeon, California](https://earthquake.usgs.gov/earthquakes/eventpage/nc21323712/executive)  
- **ci3144585**: [M 6.7, 1994, Northridge, California Earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/ci3144585/executive)  

## Useful Files and JSON Attributes for Our Project
- **`ID/current/products/info.json`**:  
  Contains `input/event_information` with basic earthquake details.  

- **`ID/current/products/stationlist.json`**:  
  Contains data under `root/feature/`, including:  
  - DYFI data  
  - Seismometer data  
  - Intensity  
  - PGA (Peak Ground Acceleration)  
  - PGV (Peak Ground Velocity)  
  - PSA (Peak Spectral Acceleration)  
  - Vs30 (average shear-wave velocity)  
  - GPS coordinates  

**Note:** DYFI data is available at a 1 km spatial resolution.
