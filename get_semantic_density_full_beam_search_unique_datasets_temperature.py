import argparse
import os
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

import config
os.environ['HF_DATASETS_CACHE'] = config.hf_datasets_cache
import math

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--deg_codebase', type=str, default=True)
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_device



# Set a seed value
seed_value = 3
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
    
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to('cuda:0')

with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/{args.generation_model}_generations_{args.generation_model}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as infile:
    likelihoods = pickle.load(infile)


def clean_text(generated_text):
    strings_to_filter_on = [
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]
    for string in strings_to_filter_on:
        if string in generated_text:
            generated_text = generated_text.split(string)[0]
    return generated_text

num_beams = 10

average_contradict_prob_list_beam = []
semantic_density_list_beam = []

for beam_id in range(0, num_beams):
    average_contradict_prob_list = []
    semantic_density_list = []
    index = 0
    for sample in tqdm(sequences):
        question = sample['question']
        if 'cleaned_generated_texts' in sample:
            generated_texts = sample['cleaned_generated_texts']
        else:
            generated_texts = sample['generated_texts']

        id_ = sample['id']

        most_likely_text = clean_text(sample['cleaned_beam_search_generation_{}'.format(beam_id)])
        contradict_prob_list = []
        deg_contradict_prob_list = []

        likelihood_sum = 0
        semantic_density = 0
        # Evalauate semantic similarity
        unique_cleaned_beam_search_generation = set()
        deg_unique_cleaned_beam_search_generation = set()
        unique_beam_index = []
        deg_unique_beam_index = []
        for beam_index in range(num_beams):
            beam_search_generation_text = sample['cleaned_beam_search_generation_{}'.format(beam_index)]
            
            if beam_search_generation_text not in unique_cleaned_beam_search_generation:
                unique_cleaned_beam_search_generation.add(beam_search_generation_text)
                unique_beam_index.append(beam_index)
            
            if args.deg_codebase: 
                # Follow multinomial sampling from https://github.com/zlin7/UQ-NLG
                deg_generation_text = generated_texts[beam_index] 
                if deg_generation_text not in deg_unique_cleaned_beam_search_generation:
                    deg_unique_cleaned_beam_search_generation.add(deg_generation_text)
                    deg_unique_beam_index.append(beam_index)

        if args.deg_codebase:      
            for beam_index in deg_unique_beam_index:
                qa_1 = question + ' ' + generated_texts[beam_index]
                qa_2 = question + ' ' + most_likely_text
                origin_input = qa_1 + ' [SEP] ' + qa_2
                encoded_input = tokenizer.encode(origin_input, padding=True)
                prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda:0'))['logits'][0]
                prediction_softmax = softmax(prediction.cpu().detach().numpy())
                contradict_prob_1 = 1-prediction_softmax[2]

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda:0'))['logits'][0]
                reverse_prediction_softmax = softmax(reverse_prediction.cpu().detach().numpy())
                contradict_prob_2 = 1-reverse_prediction_softmax[2]

                deg_contradict_prob_list.append((contradict_prob_1+contradict_prob_2)/2.0)

        # Semantic Density codebase
        for beam_index in unique_beam_index:
            qa_1 = question + ' ' + sample['cleaned_beam_search_generation_{}'.format(beam_index)]
            qa_2 = question + ' ' + most_likely_text
            average_likelihood = math.exp(-likelihoods[index]['average_neg_log_likelihood_of_beam_search_gen_{}'.format(beam_index)])
            origin_input = qa_1 + ' [SEP] ' + qa_2
            encoded_input = tokenizer.encode(origin_input, padding=True)
            prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda:0'))['logits'][0]
            prediction_softmax = softmax(prediction.cpu().detach().numpy())
            contradict_prob_1 = 1-prediction_softmax[2]
            semantic_distance = prediction_softmax[0] + 0.5*prediction_softmax[1]
            semantic_density += 0.5*(1.0-semantic_distance)*average_likelihood

            reverse_input = qa_2 + ' [SEP] ' + qa_1
            encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
            reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda:0'))['logits'][0]
            reverse_prediction_softmax = softmax(reverse_prediction.cpu().detach().numpy())
            contradict_prob_2 = 1-reverse_prediction_softmax[2]
            reverse_semantic_distance = reverse_prediction_softmax[0] + 0.5*reverse_prediction_softmax[1]
            semantic_density += 0.5*(1.0-reverse_semantic_distance)*average_likelihood
            likelihood_sum += average_likelihood

            contradict_prob_list.append((contradict_prob_1+contradict_prob_2)/2.0)
        
        if args.deg_codebase:
            average_contradict_prob_list.append(np.mean(deg_contradict_prob_list))
        else:
            average_contradict_prob_list.append(np.mean(contradict_prob_list))
            
        semantic_density_list.append(semantic_density/likelihood_sum)
        index += 1

    average_contradict_prob_list_beam.append(average_contradict_prob_list)
    semantic_density_list_beam.append(semantic_density_list)

with open(f'{config.output_dir}/{args.generation_model}_generations_average_contradict_prob_beam_all_{args.dataset}_temperature{args.temperature}.pkl', 'wb') as outfile:
    pickle.dump(average_contradict_prob_list_beam, outfile)

with open(f'{config.output_dir}/{args.generation_model}_generations_semantic_density_beam_all_{args.dataset}_temperature{args.temperature}.pkl', 'wb') as outfile:
    pickle.dump(semantic_density_list_beam, outfile)
