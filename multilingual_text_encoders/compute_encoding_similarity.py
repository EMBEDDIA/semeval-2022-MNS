import numpy as np
import pandas as pd
import os
import json
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import paired_cosine_distances


from evaluate import evaluate_scores

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--encoded_file', default='articles_encoded.csv', type=str)
argparser.add_argument('--test_file', default='test.csv', type=str)
argparser.add_argument('--data_path', default='', type=str)

args = argparser.parse_args()

print("\n" + "-"*5, "Compute similarity of encoded articles", "-"*5)
print("data_path:", args.data_path)
print("encoded_file:", args.encoded_file)
print("test_file:", args.test_file)
print("-"*50 + "\n\n")


# save predicted scores to csv
def write_scores_to_file(pair_ids, pred_scores, save_path):
    df = {'pair_id': pair_ids, 'Overall': pred_scores}
    df = pd.DataFrame.from_dict(df)
    df.to_csv(save_path, index=False)
    print("Saved predicted scores to dataframe", save_path, "!")


# SemEval scores are in the range [1-4] where 1=most similar, 4=least
def renormalise_similarity_score_semeval(scores):
    # reverse the normalisation I did for SBERT finetuning
    scores = np.array(scores)
    renormalised_scores = (3 - (scores * 3)) + 1
    return renormalised_scores


def compute_similarity_for_article_pairs(encoded_df, article_pairs):
    similarity = {}
    art_ids = list(encoded_df.id)
    art_embeddings = np.array(encoded_df.drop(columns=['id']))
    for pair in article_pairs:
        #print('pair:', pair)
        id1 = int(pair.split("_")[0])
        id2 = int(pair.split("_")[1])
        if id1 in art_ids and id2 in art_ids:
            index1 = art_ids.index(id1)
            index2 = art_ids.index(id2)
            embedding1 = art_embeddings[index1]
            embedding2 = art_embeddings[index2]
            cosine_sim = 1-distance.cosine(embedding1, embedding2)
            # cosine_sim = util.cos_sim(embedding1, embedding2)
            #print('cosine_sim:', cosine_sim)
            similarity[pair] = cosine_sim
        # else:
        #     print('missing pair:', pair)
    print('pairs with scores:', len(similarity))
    print('missing pairs:', len(article_pairs)-len(similarity))
    return similarity


if __name__ == "__main__":
    encoded_df = pd.read_csv(os.path.join(args.data_path, args.encoded_file))
    test_df = pd.read_csv(os.path.join(args.data_path, args.test_file))
    test_pairs = list(test_df.pair_id)
    print('total pairs:', len(test_pairs))
    # in the real test data, true scores are not available
    if 'Overall' in test_df:
        true_scores = np.array(test_df.Overall)
    else:
        true_scores = np.array([])
    similarity = compute_similarity_for_article_pairs(encoded_df, test_pairs)
    # baseline: sample values from Gaussian
    #pred_similarity = np.random.normal(loc=0.5, scale=0.5, size=true_similarity.shape[0])
    pred_similarity = []
    for pair in test_pairs:
        if pair in similarity:
            pred_similarity.append(similarity[pair])
        else:
            pred_similarity.append(0.5)
    # print('pred_similarity:', pred_similarity)
    #print('true_similarity:', true_similarity)
    final_scores = renormalise_similarity_score_semeval(pred_similarity)
    save_filename = args.encoded_file[:-4] + "-predictions.csv"
    save_path = os.path.join(args.data_path, save_filename)
    write_scores_to_file(test_pairs, final_scores, save_path)
    # only get Pearson correlation when true scores exist
    if true_scores.shape[0] != 0:
        print("-"*5, "Evaluation", "-"*5)
        pearson, p_val = evaluate_scores(final_scores, true_scores)






