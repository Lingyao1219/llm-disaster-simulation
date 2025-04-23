
SYSTEM_PROMPT = """
You are a seismic expert specialized in earthquake damage assessment and disaster response.
"""

EARTHQUAKE_PROMPT = """
The earthquake happened date is 2025-06-01. 

Here is the EARTHQUAKE information. 
- Epicenter: {eq_place}
- Coordinates: {eq_lat}, {eq_lng}
- Magnitude: {eq_magnitude}

YOUR LOCATION information is listed below. 
- State: {state}
- City: {city}
- Zipcode: {zipcode}
- Coordinates: {lat}, {lng}
- Distance from epicenter: {distance} km
- VS30 at your location: {vs30} m/s (soil characteristics)

## Community Socioecnomics and Demographics in YOUR LOCATION (at Cencus Block Group level)
- Population density: {population_density} people per square km
- Urban population percentage: {urban_population_pct}%
- Over 65 percentage: {over_65_rate}%
- Median household income: ${median_household_income}/year
- Education (bachelor's or higher): {education}%

## Building Description in YOUR LOCATION (within a 100-meter radius)
- Building description: {building} 

## Visual Context in YOUR LOCATION
The image provided shows your surrounding environment and infrastructure.


Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale.
1. Identify the damage level.
2. Explain your reasoning by addressing the following factors and considering the visual images. 
   - Distance to the epicenter and earthquake magnitude.
   - Infrastructure quality and building characteristics.
   - Geospatial features
   - Population density and socioeconomic vulnerabilities.

The following is an abbreviated description of the 12 levels of Modified Mercalli intensity.
- I. Not felt except by a very few under especially favorable conditions.
- II. Felt only by a few persons at rest, especially on upper floors of buildings. Delicately suspended objects may swing.
- III. Felt quite noticeably by persons indoors, especially on upper floors of buildings. Many people do not recognize it as an earthquake. Standing motor cars may rock slightly. Vibration similar to the passing of a truck. Duration estimated.
- IV. Felt indoors by many, outdoors by few during the day. At night, some awakened. Dishes, windows, doors disturbed; walls make cracking sound. Sensation like heavy truck striking building. Standing motor cars rocked noticeably.
- V. Felt by nearly everyone; many awakened. some dishes, windows broken. Unstable objects overturned. Pendulum clocks may stop.
- VI. Felt by all, many frightened. Some heavy furniture moved; a few instances of fallen plaster. Damage slight.
- VII. Damage negligible in buildings of good design and construction; slight to moderate in well-built ordinary structures; considerable damage in poorly built or badly designed structures; some chimneys broken.
- VIII. Damage slight in specially designed structures; considerable damage in ordinary substantial buildings with partial collapse. Damage great in poorly built structures. Fall of chimneys, factory stacks, columns, monuments, walls. Heavy furniture overturned.
- IX. Damage considerable in specially designed structures; well-designed frame structures thrown out of plumb. Damage great in substantial buildings, with partial collapse. Buildings shifted off foundations.
- X. Some well-built wooden structures destroyed; most masonry and frame structures destroyed with foundations. Rail bent.
- XI. Few, if any (masonry) structures remain standing. Bridges destroyed. Rails bent greatly.
- XII. Damage total. Lines of sight and level are distorted. Objects thrown into the air.

Output the result in JSON format:
{{
    "Reasoning": "<Provide reasoning>"
    "MMI": "<Respond MMI level>",
}}
"""

