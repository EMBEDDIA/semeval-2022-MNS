import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import distance
from scipy.stats import pearsonr

argparser = argparse.ArgumentParser()

argparser.add_argument('--csv_file', default='test_split_batch2.csv', type=str)
argparser.add_argument('--data_path', default='/scratch/project_2003138/datasets/semeval-multilingual-news/', type=str)
argparser.add_argument('--image_dir', default='train_dir/batch2', type=str)

args = argparser.parse_args()



if __name__ == "__main__":
    csv_file = os.path.join(args.data_path, args.csv_file)
    print("Loading data from", csv_file)
    df  = pd.read_csv(csv_file)
    pairs_ids = list(df.pair_id)
    scores = list(df.Overall)
    
    ids = []
    for pair in tqdm(pairs_ids, "Collecting ids"):
        frst, scnd = pair.split("_")
        ids.append(frst)
        ids.append(scnd)
    ids = list(set(ids))
    
    image_dir = os.path.join(args.data_path, args.image_dir)
    idx2file = {}
    print("Collecting files")
    for subdir, dirs, files in os.walk(image_dir):
        for f in files:
            idx, ext = os.path.splitext(f)
            if (ext not in (".html", ".json")) and (idx in ids):
                idx2file[idx] = os.path.join(subdir,f)

    print("Found %d out of %d" %(len(idx2file), len(ids)))

    images = []
    image_ids = [] 
    for i in tqdm(idx2file, "Loading images"):
        try:
            img = Image.open(idx2file[i])
            img = img.convert("RGB")
            images.append(img)
            image_ids.append(i)
        except KeyError:
            continue
    

            
    model = SentenceTransformer('clip-ViT-B-32')
    
    img_emb = model.encode(images, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
    img_emb = img_emb
    print("img_emb", type(img_emb), img_emb.shape)

    idx2emb = {}
    for i in range(len(image_ids)):
        idx2emb[image_ids[i]] = img_emb[i,:]

    pred_similarity = []
    true_similarity = []
    for pair, score in tqdm(zip(pairs_ids, scores), "Computing similarities"):
        frst, scnd = pair.split("_")
        if frst in idx2emb and scnd in idx2emb:
            pred_similarity.append(distance.cosine(idx2emb[frst], idx2emb[scnd]))
            true_similarity.append(score)

    pearsonr_corr = pearsonr(np.array(pred_similarity), np.array(true_similarity))
    print(pearsonr_corr)
