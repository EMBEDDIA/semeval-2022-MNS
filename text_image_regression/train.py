# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import spacy
from utils import evaluate, split_data
from nltk.tokenize import sent_tokenize
import torch
import random
import logging
import argparse
import json
import math
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from sentence_transformers import (SentenceTransformer,
                                   CrossEncoder)
from sentence_transformers import losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from main import get_text
from torch.utils.data import DataLoader

#nlp = spacy.load("xx_ent_wiki_sm")

logging.basicConfig(level=logging.ERROR)
manualSeed = 2021

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


##model = SentenceTransformer('paraphrase-mpnet-base-v2')
##sentence_model = SentenceTransformer('stsb-bert-large')
##cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
#cross_encoder = CrossEncoder('castorini/monobert-large-msmarco')
# paraphrase-multilingual-mpnet-base-v2
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_path',
    type=str,
    default='../data/train_dir/batch1',
    help='batch folder')
parser.add_argument(
    '--train_csv',
    type=str,
    default='../data/semeval-2022_task8_train-data_batch1.csv',
    help='data_batch1.csv')
parser.add_argument(
    "--model_name_or_path",
    default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list ")
parser.add_argument(
    "--model_save_path",
    default='runs',
    type=str,
    help="Path to pre-trained model or shortcut name selected in the list ")

args = parser.parse_args()
data_path = args.data_path
train_csv = args.train_csv
model_name_or_path = args.model_name_or_path
model_save_path = args.model_save_path

#sentence_model = SentenceTransformer(model_name_or_path)


def transform(model, text, entities=None):
    encoding = model.encode([str(text)], convert_to_tensor=True)[0]
    return encoding.cpu().numpy()

import pandas as pd

def add_negatives(train_df):
    extra_df = pd.DataFrame(columns = train_df.columns)
    many = 5
    for _, row in train_df.iterrows():
        
        random_10 = train_df[train_df.link1 != row.link1].sample(many)
        
        random_10['link1'] = many * [row.link1]
        random_10['Overall'] = many * [0.0]
        random_10['pair_id'] = random_10['pair_id'].apply(lambda pair: pair.replace(pair.split('_')[0], row.pair_id.split('_')[0]))
        
        extra_df = pd.concat([extra_df, random_10], ignore_index=True)
        
    train_df = pd.concat([extra_df, train_df], ignore_index=True)
    return train_df

if __name__ == '__main__':

#    train_df, test_df = split_data.split_train_data_batch1(train_csv=train_csv)
    
    train_df = pd.read_csv('../data/train_split_batch1.csv')
    print(len(train_df))
    test_df = pd.read_csv('../data/test_split_batch1.csv')
    
    train_df = add_negatives(train_df)
    print(len(train_df))
    
    train_df['text1'] = train_df['pair_id'].apply(
        lambda pair: get_text(pair.split("_")[0]))
    train_df['text2'] = train_df['pair_id'].apply(
        lambda pair: get_text(pair.split("_")[1]))
    test_df['text1'] = test_df['pair_id'].apply(
        lambda pair: get_text(pair.split("_")[0]))
    test_df['text2'] = test_df['pair_id'].apply(
        lambda pair: get_text(pair.split("_")[1]))

    print(test_df.head())
    train_samples = [InputExample(texts=[entry[0], entry[1]], label=float(entry[2])) \
                    for entry in zip(train_df.text1, train_df.text2, train_df.Overall)]
    
    test_samples = [InputExample(texts=[entry[0], entry[1]], label=float(entry[2])) \
                    for entry in zip(test_df.text1, test_df.text2, test_df.Overall)]


    print(train_samples[:5])

    train_batch_size = 4
    num_epochs = 4
    
    word_embedding_model = models.Transformer(model_name_or_path)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_data_loader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='valid')

    # warm-up step (skip? guess not cuz skip = worse performance)
    warmup_steps = math.ceil(len(train_samples) * num_epochs + 0.1)

    model.fit(train_objectives=[(train_data_loader, train_loss)],
                evaluator=evaluator,
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                evaluation_steps=80,
                output_path=model_save_path)
    
    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    test_evaluator(model, output_path=model_save_path)

    true_similarity = np.array(test_df.Overall)

    pred_similarity = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
#        try:
            text1_embeddings = model.encode([row.text1], convert_to_tensor=True, show_progress_bar=False)
            text2_embeddings = model.encode([row.text2], convert_to_tensor=True, show_progress_bar=False)
            
            cosine_sim = util.pytorch_cos_sim(text1_embeddings, text2_embeddings)[0]
#            import pdb;pdb.set_trace()
            cosine_sim = cosine_sim.cpu().numpy()[0]

#            cosine_sim = distance.cosine(transform(model, row.text1),
#                                         transform(model, row.text2))

            pred_similarity.append(cosine_sim)
#            print(row.pair_id, cosine_sim)
#        except BaseException:
#            pred_similarity.append(0.5)
#            print('Issues', row.pair_id, len(row.text1), len(row.text2))

    


    print('pred_similarity:', pred_similarity)
    print('true_similarity:', true_similarity)
    pearson, p_val = evaluate.evaluate_scores(pred_similarity, true_similarity)

    print(pearson, p_val)




















