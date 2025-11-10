# LLMs as World Models: Pre-event Simulation for Disaster Impact Assessment

Efficient simulation is essential for enhancing proactive preparedness for sudden-onset disasters such as earthquakes. Recent advancements in large language models (LLMs) as world models show promise in simulating complex scenarios. This project examines multiple LLMs to proactively estimate perceived earthquake impacts. To explore the potential, we develop an LLM-based framework to simulate how humans perceive seismic risks, as illustrated below. By integrating rich pre-event contextual information including geospatial, socioeconomic, building, and street-level imagery data, the LLMs are tasked with “reasoning” the likely severity of damage across spatial scales based on the Modified Mercalli Intensity (MMI) scale.

<img width="2996" height="1352" alt="simulation" src="https://github.com/user-attachments/assets/ca0daa9f-864b-441b-8026-1bf88e6c028a" />


## Overview

This project evaluates whether LLMs can effectively assess disaster impacts by analyzing satellite imagery, building characteristics, socioeconomic data, and seismic information. The system predicts earthquake damage severity using the Modified Mercalli Intensity (MMI) scale by combining:

- **Seismic Data**: USGS ShakeMap data (PGA, PGV, magnitude, depth, distance)
- **Building Information**: OpenStreetMap building characteristics
- **Geospatial Features**: Vs30 soil velocity data
- **Socioeconomic Data**: US Census demographics (population, income, education)
- **Visual Context**: Google Street View imagery

<img width="2986" height="1658" alt="framework" src="https://github.com/user-attachments/assets/79cbf905-a0de-4c61-b93b-607b74dd5333" />


## Project Structure

```
llm-disaster-simulation/
├── eq_data/                    # Earthquake ShakeMap data from USGS
├── data_preparation/           # Feature engineering and data ingestion
├── experiment/                 # Core ML experiment pipelines
├── analysis/                   # Post-processing and results analysis
└── README.md                   # Project documentation
```

## Earthquake Datasets

The project includes two major earthquake events with complete USGS ShakeMap products:

| Event ID | Name | Magnitude | Date | Location |
|----------|------|-----------|------|----------|
| ci38457511 | Ridgecrest | M 7.1 | 2019 | Ridgecrest, CA |
| nc72282711 | Napa | M 6.0 | 2014 | Napa, CA |

See [eq_data/README.md](eq_data/README.md) for detailed earthquake data structure.

## Pipeline Workflow

### 1. Data Preparation

**Location**: `data_preparation/`

Extract and aggregate multimodal features for sampled locations:

```bash
# Main workflow in data_preparation.ipynb
- Sample ZIP codes within earthquake impact zone
- Extract building features from OpenStreetMap
- Retrieve Vs30 soil characteristics from GeoTIFF
- Collect socioeconomic data from US Census
- Download Google Street View imagery
- Generate structured prompts with all features
```

**Key Modules**:
- `config.py` - Configuration parameters
- `prompt.py` - Prompt templates with MMI scale definitions
- `building.py` - OSM building feature extraction
- `station.py` - USGS ShakeMap station data processing
- `vs30.py` - Soil velocity data extraction
- `socioeconomics.py` - Census demographic data
- `streetview.py` - Google Street View image retrieval

**Output**: CSV files with prompts + street view images

### 2. Experiment Execution

**Location**: `experiment/`

Run LLM inference with various configurations:

```bash
# Single-request inference
python main.py --dataset 2019_ridgecrest --model gpt-4o --setting B+G+B+C+V

# Batch API inference (cost-efficient)
python main_batch_claude.py --dataset 2019_ridgecrest --model claude-3-5-sonnet

# With in-context learning (k-shot)
python main.py --dataset 2019_ridgecrest --model gpt-4o --ICL --k 5

# With RAG (retrieve k-nearest neighbors)
python main.py --dataset 2019_ridgecrest --model gpt-4o --RAG --k 5
```

**Supported Models**:
- **Closed-source**: GPT-4o, Claude 3.5 Sonnet/Haiku, Gemini 2.5 Pro
- **Open-source**: Qwen2.5-VL (3B/7B/32B/72B), Llama 3.2 Vision
- **Backends**: vLLM (local), Together.ai, Anthropic Batch API, OpenAI

**Modality Settings**:
- `B+G+B+C+V` - All modalities (Buildings, Geospatial, Buildings, Community, Visual)
- `B+G+C+V` - Without geospatial
- `B+G+B+V` - Without community/socioeconomic
- `B+G+B+C` - Without visual/street view

**Key Scripts**:
- `main.py` - Single-request inference with multiprocessing
- `main_batch_claude.py` - Anthropic batch API
- `main_batch.py` - OpenAI-compatible batch API
- `build_rag.py` - Prepare RAG demonstrations
- `response_sampling_vllm.py` - Local inference with vLLM
- `evaluation.py` - Compute metrics (RMSE, correlation)
- `utils.py` - Multi-LLM API utilities

### 3. Analysis

**Location**: `analysis/`

Analyze results and generate visualizations:

- `result_distance.ipynb` - Accuracy vs. distance from epicenter
- `result_text.ipynb` - NLP analysis of model reasoning
- `visual.ipynb` - Heatmaps and distribution plots
- `city_analysis.ipynb` - Geographic distribution of results

## Dependencies

### Required Python Libraries

```bash
# LLM APIs
openai, anthropic, together, google-generativeai

# Local Inference
vllm, torch

# Data Processing
pandas, numpy, geopandas

# Geospatial
osmnx, shapely, rasterio

# ML/Stats
scikit-learn, scipy

# Utilities
loguru, tqdm, requests
```

### External APIs

- **Google Street View API** - Image collection
- **USGS ShakeMap API** - Seismic data
- **OpenStreetMap** - Building characteristics
- **LLM APIs** - OpenAI, Anthropic, Together, Google Gemini

### Configuration

Create a `secrets.txt` file with your API keys:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## MMI Scale (Target Output)

The Modified Mercalli Intensity (MMI) scale measures earthquake impact:

- **I-II**: Not felt or barely felt
- **III-IV**: Felt indoors, minor effects
- **V-VI**: Felt widely, minor to moderate damage
- **VII-VIII**: Considerable to major damage
- **IX-X**: Heavy to severe damage
- **XI-XII**: Nearly total to total destruction

## Running Experiments

### Local Inference with vLLM

```bash
# Start vLLM server
bash experiment/run_vllm.sh

# Run experiments across datasets and settings
# (configured in shell script)
```

### API-based Inference

```bash
# Single model
bash experiment/run_api.sh

# Batch API (cost-optimized)
bash experiment/run_batch_api.sh

# With RAG
bash experiment/run_batch_api_rag.sh
```

### Evaluate Results

```bash
bash experiment/run_evaluation.sh
```

### Key Features

- **Multimodal Input**: Combines 5 modalities (seismic, visual, building, geospatial, socioeconomic)
- **Modality Ablation**: Tests impact of removing specific features
- **In-Context Learning**: Few-shot demonstrations (k-shot learning)
- **RAG Support**: k-NN retrieval of similar demonstrations
- **Data Leakage Testing**: Fairness evaluation with training data exposure
- **Multi-LLM Comparison**: Tests 10+ different models
- **Cost Optimization**: Batch API support for large-scale experiments


## Main Results

We evaluate multiple LLMs on simulating MMI predictions for two earthquake events: the 2014 Napa earthquake (magnitude 6.0) and the 2019 Ridgecrest earthquake (magnitude 7.1). The models are assessed at both zip code and county levels using RMSE and Pearson correlation metrics, as presented below.

| Model | Open Source | 2014 Napa |  |  |  | 2019 Ridgecrest |  |  |  |
|-------|-------------|-----------|------|-----------|------|-----------------|------|-----------|------|
|  |  | RMSE_Z ↓ | Corr_Z ↑ | RMSE_C ↓ | Corr_C ↑ | RMSE_Z ↓ | Corr_Z ↑ | RMSE_C ↓ | Corr_C ↑ |
| **Closed-Source Models**  |
| GPT-4o-2024-08-06 | ✗ | 2.43 | **0.77** | 2.37 | 0.88 | 1.97 | **0.75** | 1.91 | 0.77 |
| GPT-4.1-mini | ✗ | 2.56 | 0.61 | 2.48 | 0.67 | **0.92** | 0.64 | **0.77** | 0.76 |
| Claude-3.5-haiku | ✗ | 2.11 | 0.58 | 2.05 | 0.70 | 1.35 | 0.59 | 1.38 | 0.71 |
| **Open-Source Models**  |
| Llama-3.2-11B-VI | ✓ | 3.19 | 0.44 | 3.05 | 0.86 | 3.22 | 0.33 | 3.22 | 0.27 |
| Llama-3.2-90B-VI | ✓ | 2.62 | 0.57 | 2.55 | 0.66 | 2.06 | 0.62 | 2.19 | 0.59 |
| Qwen2.5-VL-3B | ✓ | 3.63 | 0.29 | 3.59 | 0.15 | 3.88 | 0.01 | 4.08 | -0.20 |
| Qwen2.5-VL-7B | ✓ | 1.79 | 0.43 | 1.68 | 0.70 | 1.53 | 0.05 | 1.59 | -0.18 |
| Qwen2.5-VL-32B | ✓ | **1.59** | 0.70 | **1.56** | 0.79 | 0.99 | 0.71 | 0.96 | 0.80 |
| Qwen2.5-VL-72B | ✓ | 2.17 | 0.46 | 2.12 | 0.44 | 1.39 | 0.64 | 1.28 | **0.86** |

**Note:** RMSE_Z and Corr_Z refer to zip code-level metrics; RMSE_C and Corr_C refer to county-level metrics. Best per-column values are highlighted in **bold**.

### Key Findings

- **Strong Performance**: The best-performing models achieve reasonable correlations with USGS "Did You Feel It?" (DYFI) reports, showing alignment with human-perceived earthquake impacts.
- **Closed-Source Advantage**: Closed-source models (GPT-4o, GPT-4.1-mini) generally outperform open-source alternatives.
- **Correlation vs. RMSE Trade-off**: We observe that correlation and RMSE metrics don't always align, indicating that models can effectively rank relative severity (high correlation) while struggling with absolute MMI value prediction (high RMSE).


## Citation

If you use this code or methodology in your research, please cite:

```
@inproceedings{li-etal-2025-llms-world,
    title = "{LLM}s as World Models: Data-Driven and Human-Centered Pre-Event Simulation for Disaster Impact Assessment",
    author = "Li, Lingyao  and
      Li, Dawei  and
      Ou, Zhenhui  and
      Xu, Xiaoran  and
      Liu, Jingxiao  and
      Ma, Zihui  and
      Yu, Runlong  and
      Deng, Min",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.153/",
    doi = "10.18653/v1/2025.emnlp-main.153",
    pages = "3078--3096",
    ISBN = "979-8-89176-332-6"
}
```



