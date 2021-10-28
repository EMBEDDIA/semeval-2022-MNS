import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from evaluate import evaluate_scores, compute_jsd, compute_kld2

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', default='results/', type=str)
argparser.add_argument('--model_name', default='contextualized_topic_model', type=str)
argparser.add_argument('--vocab_size', default=13465, type=int)
argparser.add_argument('--data_path', default='data/', type=str)
argparser.add_argument('--test_articles_file', default='test_articles.csv', type=str)
argparser.add_argument('--test_pairs_file', default='test.csv', type=str)
args = argparser.parse_args()

print("\n" + "-"*5, "Test zero-shot Contextualized Topic Model", "-"*5)
print("model_dir:", args.model_dir)
print("model_name:", args.model_name)
print("vocab_size:", args.vocab_size)
print("data_path:", args.data_path)
print("test_articles_file:", args.test_articles_file)
print("test_pairs_file:", args.test_pairs_file)
print("-"*40 + "\n")

print("Loading saved model")
#model_path = os.path.join(args.model_dir, args.model_name)
model_path = "/proj/zosa/bin/results/semeval-mns/trained_models/contextualized_topic_model_nc_100_tpm_0.0_tpv_0.99_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99"

ctm = ZeroShotTM(bow_size=13465,
                 contextual_size=768)

ctm.load(model_path, epoch=249)
topic_words = ctm.get_topics()
num_topics = len(topic_words)
print("topics:", num_topics)

# load test articles
#test_path = os.path.join(args.data_path, args.test_file)
test_path = "/proj/zosa/data/semeval-multilingual-news/test_split_batch1_articles.csv"
test_df = pd.read_csv(test_path)
test_articles = list(test_df['text'])
test_ids = list(test_df['id'])
print("Test articles:", len(test_articles))

# get document-topic distribution
qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")
testing_dataset = qt.transform(test_articles)
doc_topics = ctm.get_doc_topic_distribution(testing_dataset, n_samples=100)
print("doc_topics:", doc_topics.shape)

# make topic distributions more sparse and normalise
doc_topics[doc_topics < 1/num_topics] = 0
doc_topics = doc_topics/doc_topics.sum(axis=1)[:, np.newaxis]

# compute JSD or cosine sim between topic distributions
test_pairs_path = "/proj/zosa/data/semeval-multilingual-news/test_split_batch1.csv"
test_pairs_df = pd.read_csv(test_pairs_path)
pair_id = list(test_pairs_df['pair_id'])
true_scores = list(test_pairs_df['Overall'])
cosine_pred_scores = []
jsd_pred_scores = []
for i in range(len(pair_id)):
    id1 = int(pair_id[i].split("_")[0])
    id2 = int(pair_id[i].split("_")[1])
    if id1 in test_ids and id2 in test_ids:
        topics1 = doc_topics[test_ids.index(id1)]
        topics2 = doc_topics[test_ids.index(id2)]
        jsd = compute_jsd(topics1, topics2)
        cosine_dist = distance.cosine(topics1, topics2)
        cosine_pred_scores.append(cosine_dist)
        jsd_pred_scores.append(jsd)
    else:
        cosine_pred_scores.append(0.5)
        jsd_pred_scores.append(0.5)

# get Pearson-corr between true similarity and predicted
print("\ntrue_scores:", true_scores)
print("\ncosine_pred_scores:", cosine_pred_scores)
print("\njsd_pred_scores:", jsd_pred_scores)


print("\n--- cosine distance ---")
pearson_r, p_val = evaluate_scores(true_scores, cosine_pred_scores)

print("\n--- JSD ---")
pearson_r, p_val = evaluate_scores(true_scores, jsd_pred_scores)
