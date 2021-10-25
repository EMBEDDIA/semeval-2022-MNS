from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import json
import time
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--csv_file', default='keywords_lemmatized_2018.csv', type=str)
argparser.add_argument('--data_path', default='/proj/zosa/newseye_data/stt/stt_articles_2004_2018_lemmatized/', type=str)

args = argparser.parse_args()

print("\n" + "-"*5, "Encode articles using SBERT", "-"*5)
print("csv_file:", args.csv_file)
print("data_path:", args.data_path)
print("-"*30 + "\n\n")


def create_batch1_csv(csv_filename):
    #data_path = "/users/zosaelai/project_dir/datasets/semeval-multilingual-news"
    batch1_path = os.path.join(args.data_path, "train_dir/batch1")
    dataset_df = pd.read_csv(os.path.join(args.data_path, csv_filename))
    #print("Dataset df:", dataset_df)
    pair_ids = list(dataset_df.pair_id)
    article_df = {'text': [], 'id': []}
    for pair in pair_ids:
        id1 = pair.split("_")[0]
        id2 = pair.split("_")[1]
        path1 = os.path.join(batch1_path, id1[-2:], id1 + ".json")
        path2 = os.path.join(batch1_path, id2[-2:], id2 + ".json")
        if os.path.exists(path1) is True:
            data = json.load(open(path1, 'r'))
            article_text = data['text']
            article_df['text'].append(article_text)
            article_df['id'].append(id1)
        if os.path.exists(path2) is True:
            data = json.load(open(path2, 'r'))
            article_text = data['text']
            article_df['text'].append(article_text)
            article_df['id'].append(id2)
    article_df = pd.DataFrame.from_dict(article_df)
    print("article_df:", article_df.shape)
    out_filename = os.path.join(args.data_path, csv_filename[:-4]+"_articles.csv")
    article_df.to_csv(out_filename)
    print("Done! Saved articles df to", out_filename, "!")


def encode(df, save_filepath):
    try:
        documents = list(df.text)
        documents = [" ".join(doc.split()) for doc in documents]
        model_name = 'distiluse-base-multilingual-cased'
        model = SentenceTransformer(model_name)
        print(f"[!] Encoding", len(documents), "articles")

        now = time.time()
        enc = model.encode(documents)
        encdf = pd.DataFrame(enc)
        if 'id' in df.columns:
            encdf['id'] = list(df.id)

        print(f"[!] df shape: {df.shape}")
        print(f"[!] Took {time.time() - now}s")
        encdf.to_csv(save_filepath, index=False)
        # encdf.to_feather(filename+".ft")

        print(f"[+] Written to {save_filepath}")

    except Exception as ex:
        print(ex)



if __name__ == "__main__":
    #create_batch1_csv(args.csv_file)
    csv_path = os.path.join(args.data_path, args.csv_file)
    print("Loading df from", csv_path)
    articles_df = pd.read_csv(csv_path)
    save_filename = csv_path[:-4] + "_encoded.csv"
    encode(articles_df, save_filename)





