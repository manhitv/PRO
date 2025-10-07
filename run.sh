### Parse dataset
python parse_trivia_qa.py --model='gemma-2b'
python parse_datasets.py --model='gemma-2b' --dataset='sciq'
python parse_datasets.py --model='gemma-2b' --dataset='nq'

### Generation
python generate_beam_search_save_all_triviaqa_coqa_cleaned_device.py --num_generations_per_prompt=10 --model='gemma-2b' --fraction_of_data_to_use=0.1 --num_beams=10 --top_p=1.0 --dataset='trivia_qa'
python generate_beam_search_save_all_datasets_cleaned_device.py --num_generations_per_prompt=10 --model='gemma-2b' --fraction_of_data_to_use=1.0 --num_beams=10 --top_p=1.0 --dataset='sciq'
python generate_beam_search_save_all_datasets_cleaned_device.py --num_generations_per_prompt=10 --model='gemma-2b' --fraction_of_data_to_use=0.5 --num_beams=10 --top_p=1.0 --dataset='nq'

### Likelihood and Semantic similarity
python get_semantic_similarities_beam_search_datasets.py --generation_model='gemma-2b' --dataset={dataset_name}
python get_likelihoods_beam_search_datasets_temperature.py --evaluation_model='gemma-2b' --generation_model='gemma-2b' --dataset={dataset_name} --temperature=1.0
python calculate_beam_search_rouge_datasets.py --model='gemma-2b' --dataset={dataset_name}

# -- Our score: PRO
python pro.py --model='gemma-2b' --dataset={dataset_name}

# -- Other baselines
python get_semantic_density_full_beam_search_unique_datasets_temperature.py --generation_model='gemma-2b' --dataset={dataset_name} --temperature=1.0
python compute_confidence_measure_beam_search_unique_temperature.py --generation_model='gemma-2b' --evaluation_model='gemma-2b' --dataset={dataset_name} --temperature=1
python analyze_results_semantic_density_full_datasets_temperature.py --dataset={dataset_name} --model='gemma-2b' --temperature=1.0
python results_table_auroc.py --dataset={dataset_name} --temperature=1.0