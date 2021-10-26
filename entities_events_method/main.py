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
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from sentence_transformers import (SentenceTransformer,
                                   CrossEncoder)


nlp = spacy.load("en_core_web_sm")

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
    default='stsb-mpnet-base-v2',
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list ")

args = parser.parse_args()
data_path = args.data_path
train_csv = args.train_csv
model_name_or_path = args.model_name_or_path

sentence_model = SentenceTransformer(model_name_or_path)


def transform(model, text, entities=None):
    sentences = sent_tokenize(text)  # [:3]
    for idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        for entity in entities:
            sentences[idx] = sentences[idx].replace(
                entity[0], '<' + entity[1] + '> ' + entity[0] +
                ' </' + entity[1] + '>')
#        print(sentences[idx])
    encodings = model.encode(
        sentences,
        convert_to_tensor=True,
        output_value="token_embeddings")
    sentences = [torch.sum(x, 0) for x in encodings]
    return torch.mean(torch.stack(sentences, 1), 1).cpu().numpy()


def get_text(pair_id):
    path = os.path.join(data_path, pair_id[-2:], pair_id + ".json")

    article_text = ''
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        article_text = data['text']
    else:
        print(path)
    return article_text


if __name__ == '__main__':

    train_df, test_df = split_data.split_train_data_batch1(train_csv=train_csv)

    test_df['text1'] = test_df['pair_id'].apply(
        lambda pair: get_text(pair.split("_")[0]))
    test_df['text2'] = test_df['pair_id'].apply(
        lambda pair: get_text(pair.split("_")[1]))

    print(test_df.head())

    true_similarity = np.array(test_df.Overall)

    pred_similarity = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            cosine_sim = distance.cosine(transform(sentence_model, row.text1),
                                         transform(sentence_model, row.text2))

            pred_similarity.append(cosine_sim)
#            print(row.pair_id, cosine_sim)
        except BaseException:
            pred_similarity.append(0.5)
            print('Issues', row.pair_id, len(row.text1), len(row.text2))

    print('pred_similarity:', pred_similarity)
    print('true_similarity:', true_similarity)
    pearson, p_val = evaluate.evaluate_scores(pred_similarity, true_similarity)

    print(pearson, p_val)
