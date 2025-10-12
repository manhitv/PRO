import argparse
import os
import config
import pickle

experiment_id = os.getpid()
print('experiment_id: ', experiment_id)

import evaluate
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random

device = 'cuda'

print(torch.cuda.device_count())
print(torch.cuda.is_available())
cache_dir = f'/tmp/rouge_cache_{experiment_id}'
os.environ['HF_EVALUATE_CACHE'] = cache_dir
os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--cuda_device', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device


# Set a seed value
seed_value = 10
num_beams = 10

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 'Llama-3.1-8B', 'Llama-3.2-1B', 'Llama-3.2-3B']
opt_models = ['opt-6.7b', 'opt-13b', 'opt-30b', 'opt-2.7b', 'opt-1.3b']
gemma_models = ['gemma-2-9b', 'gemma-2b', 'gemma-7b', 'gemma-2-2b', 'gemma-2-27b']
falcon_models = ['falcon-7b-instruct', 'Falcon3-7B-Instruct', 'Falcon3-10B-Instruct', 'falcon-7b', 'falcon-40b', 'falcon-11b']
    
if f"{args.model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.model}"

if f"{args.model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.model}"

if f"{args.model}" in opt_models:
    hf_model_dir = 'facebook/' + f"{args.model}"

if f"{args.model}" in gemma_models:
    hf_model_dir = 'google/' + f"{args.model}"

if f"{args.model}" in falcon_models:
    hf_model_dir = 'tiiuae/' + f"{args.model}"


print("before loading model")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype="float16", 
    bnb_4bit_use_double_quant=True, 
)

tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir)
if args.model in ['opt-30b', 'gemma-2-27b']:
    model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                                load_in_8bit=True,
                                                cache_dir=config.hf_cache_dir, device_map="auto")
elif args.model in ['falcon-40b', 'Llama-2-70b-hf', 'Mixtral-8x22B-v0.1', 'Mixtral-8x7B-v0.1']:
    model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                                quantization_config=bnb_config,
                                                cache_dir=config.hf_cache_dir, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                                torch_dtype=torch.float16,
                                                cache_dir=config.hf_cache_dir, device_map="auto")

with open(f'{config.data_dir}/{args.dataset}_{args.model}.pkl', 'rb') as dataset_file:
    dataset = pickle.load(dataset_file)

if args.fraction_of_data_to_use < 1.0:
    dataset = dataset[0:int(len(dataset)*args.fraction_of_data_to_use)]
print(len(dataset))


eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
# Check if model does not support pad_token (Falcon)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Token ID for .
period_tokenized = tokenizer('. ')['input_ids']
period_token_id = period_tokenized[1] if len(period_tokenized) > 1 else period_tokenized[0]  # Avoid IndexError

# Process framing IDs
question_framing_ids = []
for eos_token in eos_tokens:
    tokenized = tokenizer(eos_token)['input_ids']
    if len(tokenized) > 1:
        question_framing_ids.append([tokenized[1]])  # Take the second token
    elif len(tokenized) == 1:
        question_framing_ids.append([tokenized[0]])  # Take the second token
    else:
        print(f"Warning: Tokenization issue for '{eos_token}', got {tokenized}")

squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
exact_match_metric = evaluate.load("exact_match", experiment_id=experiment_id, cache_dir=cache_dir)

print(args)
def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 32
        sequences = []
        id_counter = 0
        for batch in tqdm.tqdm(dataloader):
            batch['question_id'] = [args.dataset + f'_{id_counter}']
            id_counter += 1
            input_ids = torch.tensor(batch['input_ids']).to(device).reshape(1, -1)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(input_ids,
                                                        use_cache=True,
                                                        num_beams=num_beams,
                                                        num_return_sequences=num_beams,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids,
                                                        num_beam_groups=num_beams,
                                                        diversity_penalty=1.0)
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)

            input_length = input_ids.shape[1]
            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=device)
            for i in range(number_of_generations):

                generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                            eos_token_id=period_token_id,
                                            temperature=args.temperature,
                                            bad_words_ids=question_framing_ids,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                few_shot_question = tokenizer.decode(input_ids[0])
                question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                sequence_dict = {
                    'prompt': input_ids[0],
                    'generations': generations[i],
                    'id': batch['question_id'],
                    'few_shot_question': tokenizer.decode(input_ids[0]),
                    'question': question
                }

                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[input_length:], skip_special_tokens=True))

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][input_length:], skip_special_tokens=True)
                
                # Add token likelihoods of the most likely generation

                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[1][input_length:], skip_special_tokens=True)

                sequence_dict['beam_search_generation_{}_ids'.format(0)] = sequence_dict['most_likely_generation_ids']
                sequence_dict['beam_search_generation_{}'.format(0)] = sequence_dict['most_likely_generation']
                sequence_dict['beam_search_generation_{}_ids'.format(1)] = sequence_dict['second_most_likely_generation_ids']
                sequence_dict['beam_search_generation_{}'.format(1)] = sequence_dict['second_most_likely_generation']
                for beam_index in range(2, num_beams):
                    sequence_dict['beam_search_generation_{}_ids'.format(beam_index)] = most_likely_generation[beam_index].to('cpu')
                    sequence_dict['beam_search_generation_{}'.format(beam_index)] = tokenizer.decode(
                                                most_likely_generation[beam_index][input_length:], skip_special_tokens=True)
                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0
                    sequence_dict[rouge_type + '_to_target_second'] = 0.0
                    for j in range(len(generated_texts)):
                        sequence_dict[rouge_type + '_to_target_{}'.format(j)] = 0.0

                sequence_dict['answer'] = batch['answer']
                sequence_dict['additional_answers'] = None

                sequence_dict['exact_match'] = 0.0
                sequence_dict['exact_match_second'] = 0.0
                for j in range(len(generated_texts)):
                    sequence_dict['exact_match_{}'.format(j)] = 0.0

                reference_answers = batch['answer']

                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target'])
                    predictions = [sequence_dict['second_most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match_second'] = max(results['exact_match'], sequence_dict['exact_match_second'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target_second'] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target_second'])
                    for j in range(len(generated_texts)):
                        predictions = [sequence_dict['generated_texts'][j].lstrip()]
                        references = [answer]
                        results = exact_match_metric.compute(predictions=predictions,
                                                             references=references,
                                                             ignore_case=True,
                                                             ignore_punctuation=True)
                        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match_{}'.format(j)])
                        rouge_results = rouge.compute(predictions=predictions, references=references)
                        for rouge_type in rouge_types:
                            sequence_dict[rouge_type + '_to_target_{}'.format(j)] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                           sequence_dict[rouge_type + '_to_target_{}'.format(j)])

                sequences.append(sequence_dict)

            del most_likely_generation
            del generations
            torch.cuda.empty_cache()


    return sequences


sequences = get_generations(model, dataset, args.num_generations_per_prompt)

cleaned_sequences = []

for sample in tqdm.tqdm(sequences):
    cleaned_generations = torch.ones_like(sample['generations'])
    question = sample['question']
    generated_texts = sample['generated_texts']
    cleaned_generated_texts = []
    max_len_of_generations = cleaned_generations.shape[-1]

    strings_to_filter_on = [
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:', ':', '000'
    ]
    if 'Meta-Llama-3' in f"{args.model}":
        strings_to_filter_on.append('000000')

    for i, generated_text in enumerate(generated_texts):
        for string in strings_to_filter_on:
            if string in generated_text:
                generated_text = generated_text.split(string)[0].rstrip()
        cleaned_generated_texts.append(generated_text)
        clean_ids = torch.cat(
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])
        cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

    sample['cleaned_generated_texts'] = cleaned_generated_texts
    sample['cleaned_generations'] = cleaned_generations

    for i in range(num_beams):
        sample['cleaned_beam_search_generation_{}'.format(i)] = sample['beam_search_generation_{}'.format(i)]
        for string in strings_to_filter_on:
            if string in sample['cleaned_beam_search_generation_{}'.format(i)]:
                sample['cleaned_beam_search_generation_{}'.format(i)] = sample['cleaned_beam_search_generation_{}'.format(i)].split(string)[0].rstrip()
        clean_ids = torch.cat(
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(sample['cleaned_beam_search_generation_{}'.format(i)])['input_ids'][1:], device=device)])
        sample['cleaned_beam_search_generation_{}_ids'.format(i)] = torch.ones_like(sample['beam_search_generation_{}_ids'.format(i)])
        max_len_of_generations = len(sample['beam_search_generation_{}_ids'.format(i)])
        sample['cleaned_beam_search_generation_{}_ids'.format(i)][:min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

    sample['most_likely_generation_ids'] = sample['cleaned_beam_search_generation_0_ids']
    sample['second_most_likely_generation_ids'] = sample['cleaned_beam_search_generation_1_ids']
    sample['most_likely_generation'] = sample['cleaned_beam_search_generation_0']
    sample['second_most_likely_generation'] = sample['cleaned_beam_search_generation_1']

    cleaned_sequences.append(sample)


with open(f'{config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl', 'wb') as outfile:
    pickle.dump(cleaned_sequences, outfile)