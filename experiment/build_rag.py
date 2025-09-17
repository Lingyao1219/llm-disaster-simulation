import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin
import argparse

def main():

    # --- Argument parsing for user inputs ---
    parser = argparse.ArgumentParser(description="Retrieve top-K similar demonstrations for each test sample.")
    parser.add_argument('--dataset', type=str, default="2019_ridgecrest")
    parser.add_argument('--k', type=int, default=5, help='Number of most similar demonstration samples to retrieve per test sample.')
    parser.add_argument('--use_text', action='store_true', help='Include text embedding similarity in the similarity calculation.')
    args = parser.parse_args()
    K = args.k
    USE_TEXT = args.use_text

    # --- Load the CSV data into pandas DataFrames ---
    demo_file = "data/{}_rag_samples_prompt.csv".format(args.dataset)
    test_file = "data/{}_samples_prompt.csv".format(args.dataset)
    demo_df = pd.read_csv(demo_file)
    test_df = pd.read_csv(test_file)

    # (Optional) Drop or ignore the MMI column since we should not use it for similarity
    # We'll just ensure not to include it in feature lists, but keep it in dataframes if needed for other uses.
    if 'MMI' in demo_df.columns:
        pass  # not using it for similarity
    if 'MMI' in test_df.columns:
        pass  # not using it for similarity

    # Identify numeric feature columns to use for similarity (exclude non-numerical and label columns)
    numeric_features = [
        'distance',               # distance from epicenter (km) - provided as numeric in data
        'vs30',                   # soil shear-wave velocity
        'population_density', 
        'urban_population_pct', 
        'median_household_income', 
        'education', 
        'over_65_rate'
    ]
    # The demonstration data has a 'Responses' column (survey count) which is not in test data, so we skip it.
    # Also exclude latitude and longitude from this list because we'll handle those via haversine separately.
    # (The 'distance' here is distance to epicenter, not the same as direct lat/lon distance.)

    # Verify all chosen features exist in both dataframes (drop any that don't just in case)
    numeric_features = [col for col in numeric_features if col in demo_df.columns and col in test_df.columns]

    # --- Normalize the numeric features using min-max scaling ---
    # Compute global min and max for each feature across both demo and test for fairness
    mins = {}
    maxs = {}
    for col in numeric_features:
        combined_min = min(demo_df[col].min(), test_df[col].min())
        combined_max = max(demo_df[col].max(), test_df[col].max())
        mins[col] = combined_min
        maxs[col] = combined_max

    # Apply min-max normalization
    demo_norm = demo_df.copy()
    test_norm = test_df.copy()
    for col in numeric_features:
        denom = (maxs[col] - mins[col]) if (maxs[col] - mins[col]) != 0 else 1.0  # avoid division by zero
        demo_norm[col] = (demo_df[col] - mins[col]) / denom
        test_norm[col] = (test_df[col] - mins[col]) / denom

    # Now demo_norm and test_norm contain normalized values for the numeric features.

    # --- Define a function for Haversine distance (geospatial distance) ---
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate great-circle distance between two points (lat1, lon1) and (lat2, lon2) using Haversine formula.
        Returns distance in kilometers.
        """
        # convert degrees to radians
        phi1, lam1, phi2, lam2 = map(radians, [lat1, lon1, lat2, lon2])
        # haversine formula
        dphi = phi2 - phi1
        dlam = lam2 - lam1
        a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlam/2)**2
        c = 2 * asin(sqrt(a))
        R = 6371.0  # Earth radius in kilometers
        return R * c

    # Pre-compute a normalization factor for geospatial distance (max possible distance in dataset)
    all_lats = np.concatenate([demo_df['Latitude'].values, test_df['Latitude'].values])
    all_lons = np.concatenate([demo_df['Longitude'].values, test_df['Longitude'].values])
    min_lat, max_lat = all_lats.min(), all_lats.max()
    min_lon, max_lon = all_lons.min(), all_lons.max()
    # Compute distances between the extreme coordinate pairs as an upper bound
    corner1 = haversine_distance(min_lat, min_lon, max_lat, max_lon)
    corner2 = haversine_distance(min_lat, max_lon, max_lat, min_lon)
    max_geo_dist = max(corner1, corner2)  # use the larger of the two diagonal distances
    if max_geo_dist == 0:
        max_geo_dist = 1e-6  # avoid zero division if all points identical (unlikely)

    # --- (Optional) Text embedding for prompt similarity ---
    if USE_TEXT:
        from sentence_transformers import SentenceTransformer, util
        # Combine text fields to form the prompt content for embedding
        demo_texts = (demo_df['building'].fillna('') + " " + demo_df['earthquake_prompt'].fillna('')).tolist()
        test_texts = (test_df['building'].fillna('') + " " + test_df['earthquake_prompt'].fillna('')).tolist()
        # Load a sentence transformer model for embeddings (this may download a model if not already available)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Encode all demonstration prompts to embeddings
        demo_embeddings = model.encode(demo_texts, convert_to_numpy=True)
        # We will compute test embeddings on the fly in the loop (could also encode all upfront if memory allows).
        # Precompute norms of demo embeddings for cosine similarity calculation
        demo_norms = np.linalg.norm(demo_embeddings, axis=1)
    else:
        demo_embeddings = None  # not used
        demo_norms = None

    # --- Retrieve top-K similar demonstrations for each test sample ---
    results = {}  # dictionary to hold top-K indices for each test sample
    for idx, test_row in test_norm.iterrows():
        # Get test sample's normalized numeric feature vector as numpy array
        test_numeric_vec = test_row[numeric_features].to_numpy(dtype=float)
        # Compute numeric distances (Euclidean) to all demo samples
        demo_numeric_matrix = demo_norm[numeric_features].to_numpy(dtype=float)  # shape (num_demo, num_features)
        # Euclidean distance: sqrt(sum of squared differences for each feature)
        diff_matrix = demo_numeric_matrix - test_numeric_vec  # broadcast subtraction, shape (num_demo, num_features)
        numeric_dists = np.sqrt(np.sum(diff_matrix**2, axis=1))
        
        # Compute geospatial distances (haversine) to all demo samples
        test_lat = test_df.at[idx, 'Latitude']
        test_lon = test_df.at[idx, 'Longitude']
        # Vectorize haversine: compute distance for each demo coordinate pair
        demo_lats = demo_df['Latitude'].values
        demo_lons = demo_df['Longitude'].values
        # convert degrees to radians for arrays
        phi1 = radians(test_lat)
        lam1 = radians(test_lon)
        phi2 = np.radians(demo_lats)
        lam2 = np.radians(demo_lons)
        dphi = phi2 - phi1
        dlam = lam2 - lam1
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        geo_dists = 6371.0 * c  # Earth radius * angle = distance in km
        # Normalize geospatial distance by max_geo_dist to keep scale ~0-1
        geo_dists_norm = geo_dists / max_geo_dist

        # Compute text distance if using text similarity
        if USE_TEXT:
            # Get embedding for this test sample's prompt text
            test_text_embed = model.encode(test_texts[idx], convert_to_numpy=True)
            test_norm_val = np.linalg.norm(test_text_embed)
            # Cosine similarities: (A·B) / (||A|| * ||B||) for each demo embedding
            cos_sims = np.dot(demo_embeddings, test_text_embed) / (demo_norms * test_norm_val)
            # Some numeric stability: replace any NaN (if norms were zero) with 0
            cos_sims = np.nan_to_num(cos_sims, nan=0.0)
            text_dists = 1 - cos_sims  # convert similarity to a "distance" (range 0 to 2)
        else:
            text_dists = np.zeros(len(demo_df))  # no text difference if not using text

        # Combine distances: we simply add them up (treating each component equally after normalization)
        total_distance = numeric_dists + geo_dists_norm + text_dists

        # Get indices of the smallest distances
        top_k_idx = np.argsort(total_distance)[:K]
        results[test_df.at[idx, 'location_id']] = list(top_k_idx)

    # --- Output the results ---
    # Here we populate the results dict with each test sample's top-K similar demo indices.
    # You can format the output as needed. For demonstration, we'll print the first sample's result:
    # first_test_id = test_df.loc[0, 'location_id']
    # print(f"Top {K} similar demonstrations for test sample '{first_test_id}': {results[first_test_id]}")
    # (In practice, you might write the results to a file or use them in further processing instead of printing all.)

    retrieval_result_all = []
    for i in range(len(test_df)):
        test_id = test_df.loc[i, 'location_id']
        retrieval_result_ids = results[test_id]
        retrieval_result = [[demo_df["earthquake_prompt"][id], demo_df["MMI"][id]] for id in retrieval_result_ids]
        retrieval_result_all.append(retrieval_result)

    
    test_df["retrieval result"] = retrieval_result_all
    test_df.to_csv("data/{}_samples_prompt_rag_{}.csv".format(args.dataset, args.k), index=False)


if __name__ == "__main__":
    main()