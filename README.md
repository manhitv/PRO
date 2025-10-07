# Reproducibility Guidelines

## 🛠️ Environment Setup

Create a conda environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate pro-env
```
Ensure that the appropriate model weights (e.g., `gemma-2b`) are available locally or accessible via Hugging Face.

## 🗂️ 1. Parse Datasets
Preprocess and cache input prompts for each dataset:
```bash
# TriviaQA
python parse_trivia_qa.py --model='gemma-2b'

# SciQ and Natural Questions
python parse_datasets.py --model='gemma-2b' --dataset='sciq'
python parse_datasets.py --model='gemma-2b' --dataset='nq'
```

## 🧠 2. Beam Search Generation
Generate multiple completions per prompt using beam search.

```bash
# TriviaQA (10% of data)
python generate_beam_search_save_all_triviaqa_coqa_cleaned_device.py \
  --num_generations_per_prompt=10 \
  --model='gemma-2b' \
  --fraction_of_data_to_use=0.1 \
  --num_beams=10 --top_p=1.0 \
  --dataset='trivia_qa'

# SciQ (full dataset)
python generate_beam_search_save_all_datasets_cleaned_device.py \
  --num_generations_per_prompt=10 \
  --model='gemma-2b' \
  --fraction_of_data_to_use=1.0 \
  --num_beams=10 --top_p=1.0 \
  --dataset='sciq'

# NQ (50% of data)
python generate_beam_search_save_all_datasets_cleaned_device.py \
  --num_generations_per_prompt=10 \
  --model='gemma-2b' \
  --fraction_of_data_to_use=0.5 \
  --num_beams=10 --top_p=1.0 \
  --dataset='nq'
```

## 📊 3. Evaluation and Analysis
🔍 Semantic Similarity and Likelihood
```bash
python get_semantic_similarities_beam_search_datasets.py \
  --generation_model='gemma-2b' \
  --dataset={dataset_name}

python get_likelihoods_beam_search_datasets_temperature.py \
  --evaluation_model='gemma-2b' \
  --generation_model='gemma-2b' \
  --dataset={dataset_name} \
  --temperature=1.0

python calculate_beam_search_rouge_datasets.py \
  --model='gemma-2b' \
  --dataset={dataset_name}
```

## ✅ PRO Score (Ours)
```bash
python pro.py --model='gemma-2b' --dataset={dataset_name}
```

## 📉 Baselines for Comparison
```bash
python get_semantic_density_full_beam_search_unique_datasets_temperature.py \
  --generation_model='gemma-2b' \
  --dataset={dataset_name} \
  --temperature=1.0

python compute_confidence_measure_beam_search_unique_temperature.py \
  --generation_model='gemma-2b' \
  --evaluation_model='gemma-2b' \
  --dataset={dataset_name} \
  --temperature=1.0

python analyze_results_semantic_density_full_datasets_temperature.py \
  --dataset={dataset_name} \
  --model='gemma-2b' \
  --temperature=1.0

python results_table_auroc.py \
  --dataset={dataset_name} \
  --temperature=1.0
```

## 📓 Example Workflow
```bash
bash run.sh
```
