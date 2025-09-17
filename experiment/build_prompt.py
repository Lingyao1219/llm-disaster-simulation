import pandas as pd

dataset_name = "2014_napa"


dataset = pd.read_csv("data/{}_samples_prompt.csv".format(dataset_name))
B_B_C_V_all, B_G_C_V_all, B_G_B_V_all, B_G_B_C_all = [], [], [], []
for _, item in dataset.iterrows():
    earthquake_prompt = item["earthquake_prompt"]
    date_info, earthquake_prompt = earthquake_prompt.split("Here is the EARTHQUAKE information.")
    basic_info, earthquake_prompt = earthquake_prompt.split("## Geospatial features in YOUR LOCATION")
    geospatial_info, earthquake_prompt = earthquake_prompt.split("## Building Description in YOUR LOCATION (within a 100-meter radius)")
    building_info, earthquake_prompt = earthquake_prompt.split('## Community Socioecnomics and Demographics in YOUR LOCATION (at Cencus Block Group level)')
    community_info, earthquake_prompt = earthquake_prompt.split("## Visual Context in YOUR LOCATION")
    visual_info, output_info = earthquake_prompt.split("Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale.")

    B_B_C_V = date_info + "Here is the EARTHQUAKE information." + basic_info +\
    "## Building Description in YOUR LOCATION (within a 100-meter radius)" + building_info +\
    "## Community Socioecnomics and Demographics in YOUR LOCATION (at Cencus Block Group level)" + community_info +\
    "## Visual Context in YOUR LOCATION" + visual_info +\
    "Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale." + output_info 
    B_G_C_V = "Here is the EARTHQUAKE information." + basic_info +\
    "## Geospatial features in YOUR LOCATION" + geospatial_info +\
    "## Community Socioecnomics and Demographics in YOUR LOCATION (at Cencus Block Group level)" + community_info +\
    "## Visual Context in YOUR LOCATION" + visual_info +\
    "Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale." + output_info 
    B_G_B_V = "Here is the EARTHQUAKE information." + basic_info +\
    "## Geospatial features in YOUR LOCATION" + geospatial_info +\
    "## Building Description in YOUR LOCATION (within a 100-meter radius)" + building_info +\
    "## Visual Context in YOUR LOCATION" + visual_info +\
    "Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale." + output_info 
    B_G_B_C = "Here is the EARTHQUAKE information." + basic_info +\
    "## Geospatial features in YOUR LOCATION" + geospatial_info +\
    "## Building Description in YOUR LOCATION (within a 100-meter radius)" + building_info +\
    "## Community Socioecnomics and Demographics in YOUR LOCATION (at Cencus Block Group level)" + community_info +\
    "Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale." + output_info

    B_B_C_V = B_B_C_V.replace("   - Geospatial features\n", "")
    B_G_C_V = B_G_C_V.replace("   - Infrastructure quality and building characteristics\n", "")
    B_G_B_V = B_G_B_V.replace("   - Population density and socioeconomic vulnerabilities\n", "")
    B_G_B_C = B_G_B_C.replace("   - Visual image of surroundings\n", "")

    B_B_C_V_all.append(B_B_C_V)
    B_G_C_V_all.append(B_G_C_V)
    B_G_B_V_all.append(B_G_B_V)
    B_G_B_C_all.append(B_G_B_C)


dataset_B_B_C_V = dataset.copy()
dataset_B_G_C_V = dataset.copy()
dataset_B_G_B_V = dataset.copy()
dataset_B_G_B_C = dataset.copy()


dataset_B_B_C_V["earthquake_prompt"] = B_B_C_V_all
dataset_B_G_C_V["earthquake_prompt"] = B_G_C_V_all
dataset_B_G_B_V["earthquake_prompt"] = B_G_B_V_all
dataset_B_G_B_C["earthquake_prompt"] = B_G_B_C_all


dataset_B_B_C_V.to_csv("data/{}_samples_prompt_B_B_C_V.csv".format(dataset_name), index=False)
dataset_B_G_C_V.to_csv("data/{}_samples_prompt_B_G_C_V.csv".format(dataset_name), index=False)
dataset_B_G_B_V.to_csv("data/{}_samples_prompt_B_G_B_V.csv".format(dataset_name), index=False)
dataset_B_G_B_C.to_csv("data/{}_samples_prompt_B_G_B_C.csv".format(dataset_name), index=False)
