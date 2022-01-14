import os
import pandas as pd
import torch
from scipy.spatial import distance
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

from data import prune_vocabulary2
from evaluate import evaluate_scores, compute_jsd, compute_kld2

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', default='project_dir/datasets/semeval-multilingual-news', type=str)
argparser.add_argument('--articles_file', default='test_split_batch2_translated_mbart.csv', type=str)
argparser.add_argument('--sbert_model', default='multi-qa-mpnet-base-dot-v1', type=str)
argparser.add_argument('--save_dir', default='bin/results/ctm', type=str)
argparser.add_argument('--num_topics', default=100, type=int)
argparser.add_argument('--num_epochs', default=200, type=int)
#argparser.add_argument('--test_articles_file', default='test_articles.csv', type=str)
#argparser.add_argument('--test_pairs_file', default='test.csv', type=str)
args = argparser.parse_args()

print("\n" + "-"*5, "Train Combined CTM - monolingual only", "-"*5)
print("data_path:", args.data_path)
print("articles_file:", args.articles_file)
print("sbert_model:", args.sbert_model)
print("save_dir:", args.save_dir)
print("num_topics:", args.num_topics)
print("num_epochs:", args.num_epochs)
print("-"*50 + "\n")

df = pd.read_csv(os.path.join(args.data_path, args.articles_file))
df = df.dropna()
if 'trg_text' in df.columns:
    documents_raw = list(df.trg_text)
else:
    documents_raw = list(df.text)
print('documents_raw:', len(documents_raw))

# ----- Preprocessing -----
# articles_unproc, articles_proc = prune_vocabulary2(documents_raw)
# text_for_contextual = articles_unproc
# text_for_bow = articles_proc

# preprocess documents
preproc_pipeline = WhiteSpacePreprocessing(documents=documents_raw, vocabulary_size=5000)
preprocessed_docs, unpreprocessed_docs, vocab = preproc_pipeline.preprocess()

text_for_bow = preprocessed_docs
text_for_contextual = unpreprocessed_docs

print('text_for_contextual:', len(text_for_contextual))
print('text_for_bow:', len(text_for_bow))
print('vocab:', len(vocab))

qt = TopicModelDataPreparation(args.sbert_model)

training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)
#print("-"*10, "final vocab size:", len(qt.vocab), "-"*10)

# ----- Training -----
# initialize model
ctm = CombinedTM(bow_size=len(qt.vocab),
                 contextual_size=768,
                 n_components=args.num_topics,
                 num_epochs=args.num_epochs)

# run model
ctm.fit(train_dataset=training_dataset,
        save_dir=args.save_dir)

# see topics
ctm.get_topics()

# -----  Inference -----
# load test articles
test_art_file = "test_split_batch2_translated_mbart.csv"
test_path = os.path.join(args.data_path, test_art_file)

test_df = pd.read_csv(test_path)

if 'text' in test_df.columns:
    test_articles = list(test_df['text'])
else:
    test_articles = list(test_df['trg_text'])

test_ids = list(test_df['id'])

print("Test articles:", len(test_articles))

#  process test docs using the same DataPrep pipeline from training
testing_dataset = qt.transform(text_for_contextual=test_articles, text_for_bow=test_articles)

# get document-topic distribution
doc_topics = ctm.get_doc_topic_distribution(testing_dataset, n_samples=50)
print("doc_topics:", doc_topics.shape)

encdf = pd.DataFrame(doc_topics)
encdf['id'] = test_ids
topics_outfile = os.path.join(args.data_path, "combinedCTM_K" + str(args.num_topics) + "_" + args.sbert_model + ".csv")
encdf.to_csv(topics_outfile, index=False)
print("Saved  topic distributions to", topics_outfile, "!")

# make topic distributions more sparse and normalise
#doc_topics[doc_topics < 1/num_topics] = 0
#doc_topics = doc_topics/doc_topics.sum(axis=1)[:, np.newaxis]

# compute JSD or cosine sim between topic distributions
test_pairs_file = "test_split_batch2.csv"
test_pairs_path = os.path.join(args.data_path, test_pairs_file)


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
# print("\ntrue_scores:", true_scores)
# print("\ncosine_pred_scores:", cosine_pred_scores)
# print("\njsd_pred_scores:", jsd_pred_scores)

print("\n--- cosine distance ---")
pearson_r, p_val = evaluate_scores(true_scores, cosine_pred_scores)

#print("\n--- JSD ---")
#pearson_r, p_val = evaluate_scores(true_scores, jsd_pred_scores)


