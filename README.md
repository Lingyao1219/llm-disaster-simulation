# LLMs as World Models: Pre-event Simulation for Disaster Impact Assessment

A research project that leverages Large Language Models (LLMs) as "world models" to simulate and assess earthquake disaster impacts at the location level using multimodal geospatial data.

<img width="2996" height="1352" alt="simulation" src="https://github.com/user-attachments/assets/ca0daa9f-864b-441b-8026-1bf88e6c028a" />


## Overview

This project evaluates whether LLMs can effectively assess disaster impacts by analyzing satellite imagery, building characteristics, socioeconomic data, and seismic information. The system predicts earthquake damage severity using the Modified Mercalli Intensity (MMI) scale by combining:

- **Seismic Data**: USGS ShakeMap data (PGA, PGV, magnitude, depth, distance)
- **Building Information**: OpenStreetMap building characteristics
- **Geospatial Features**: Vs30 soil velocity data
- **Socioeconomic Data**: US Census demographics (population, income, education)
- **Visual Context**: Google Street View imagery

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

## HPC Support

SLURM scripts are provided for running on high-performance computing clusters:

- GPU allocation: 5x A100 GPUs
- Memory: 48GB per job
- Tensor parallelism for large models

## Key Features

- **Multimodal Input**: Combines 5 modalities (seismic, visual, building, geospatial, socioeconomic)
- **Modality Ablation**: Tests impact of removing specific features
- **In-Context Learning**: Few-shot demonstrations (k-shot learning)
- **RAG Support**: k-NN retrieval of similar demonstrations
- **Data Leakage Testing**: Fairness evaluation with training data exposure
- **Multi-LLM Comparison**: Tests 10+ different models
- **Cost Optimization**: Batch API support for large-scale experiments

## Citation

If you use this code or methodology in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

[Contact information to be added]

