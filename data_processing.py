import pandas as pd
import os
import json

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', default='/users/zosaelai/project_dir/datasets/semeval-multilingual-news/', type=str)
argparser.add_argument('--data_file', default='semeval-2022_task8_eval_data_202201.csv', type=str)
args = argparser.parse_args()


print("-"*5, "SemEval Task 8 - Data processing", "-"*5)
print("data_path:", args.data_path)
print("data_file:", args.data_file)
print("-"*20)

def create_batch1_csv(csv_filename, lang='all'):
    #data_path = "/users/zosaelai/project_dir/datasets/semeval-multilingual-news"
    batch1_path = os.path.join(args.data_path, "train_dir/batch1")
    dataset_df = pd.read_csv(os.path.join(args.data_path, csv_filename))
    if lang != 'all':
        dataset_df = dataset_df[dataset_df['url1_lang'] == lang]
    print("Dataset df:", dataset_df.shape)
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
    if lang != 'all':
        out_filename = os.path.join(args.data_path, csv_filename[:-4] + "_" + lang + "_articles.csv")
    else:
        out_filename = os.path.join(args.data_path, csv_filename[:-4]+"_articles.csv")
    article_df.to_csv(out_filename, index=False)
    print("Done! Saved articles df to", out_filename, "!")


def create_batch2_csv(csv_filename, lang='all'):
    #data_path = "/users/zosaelai/project_dir/datasets/semeval-multilingual-news"
    print("Getting articles for", csv_filename)
    batch_path = os.path.join(args.data_path, "train_dir/batch2")
    dataset_df = pd.read_csv(os.path.join(args.data_path, csv_filename))
    if lang != 'all':
        dataset_df = dataset_df[dataset_df['url1_lang'] == lang]
    print("Dataset df:", dataset_df.shape)
    pair_ids = list(dataset_df.pair_id)
    article_df = {'id': [], 'title': [], 'text': []}
    for pair in pair_ids:
        id1 = pair.split("_")[0]
        id2 = pair.split("_")[1]
        path1 = os.path.join(batch_path, id1[-2:], id1 + ".json")
        path2 = os.path.join(batch_path, id2[-2:], id2 + ".json")
        if os.path.exists(path1) is True:
            data = json.load(open(path1, 'r'))
            article_text = data['text']
            article_title = data['title']
            article_df['title'].append(article_title)
            article_df['text'].append(article_text)
            article_df['id'].append(id1)
        if os.path.exists(path2) is True:
            data = json.load(open(path2, 'r'))
            article_text = data['text']
            article_title = data['title']
            article_df['text'].append(article_text)
            article_df['title'].append(article_title)
            article_df['id'].append(id2)
    article_df = pd.DataFrame.from_dict(article_df)
    print("article_df:", article_df.shape)
    print(article_df.head())
    if lang != 'all':
        out_filename = os.path.join(args.data_path, csv_filename[:-4] + "_" + lang + "_articles.csv")
    else:
        out_filename = os.path.join(args.data_path, csv_filename[:-4]+"_articles.csv")
    article_df.to_csv(out_filename, index=False)
    print("Done! Saved articles df to", out_filename, "!")



# 90/10 split into train/test data
def split_train_data_batch1():
    train_csv = 'semeval-2022_task8_train-data_batch1.csv'
    df = pd.read_csv(train_csv)
    train_split = df[df.url1_lang != 'fr']
    # 28 rows of fr data
    test_split = df[df.url1_lang == 'fr']
    # shuffle rows of train split
    train_split = train_split.sample(frac=1)
    # take the first 265 rows: 265+28 = 293 (10% 0f 2932)
    test_split = pd.concat([test_split, train_split[:265]])
    train_split = train_split[265:]
    print('train size:', train_split.shape[0])
    print('test size:', test_split.shape[0])
    #train size: 2646
    #test size: 293
    #train_split.to_csv("train_split_batch1.csv", index=False)
    #test_split.to_csv("test_split_batch1.csv", index=False)
    return train_split, test_split


# 90/10 split into train/test data
def split_train_data_batch2():
    train_csv = 'semeval-2022_task8_train-data_batch2.csv'
    df = pd.read_csv(train_csv)
    train_split = df[df.url1_lang != 'fr']
    # 72 rows of fr data
    test_split = df[df.url1_lang == 'fr']
    print('test size:', test_split.shape[0])
    # shuffle rows of train split
    train_split = train_split.sample(frac=1)
    # take the first 265 rows: 424+72 = 496 (10% 0f 4964)
    test_split = pd.concat([test_split, train_split[:424]])
    train_split = train_split[424:]
    print('train size:', train_split.shape[0])
    print('test size:', test_split.shape[0])
    #train size: 4468
    #test size: 496
    train_split.to_csv("train_split_batch2.csv", index=False)
    test_split.to_csv("test_split_batch2.csv", index=False)
    return train_split, test_split


# check if all articles in the train csv file have been downloaded
def sanity_check():
    # get ids of articles
    train_csv = os.path.join(args.data_path, args.data_file) #'semeval-2022_task8_train-data_batch.csv'
    df = pd.read_csv(train_csv)
    art_ids = list(df.pair_id)
    art_ids = [pair.split('_') for pair in art_ids]
    art_ids = [i for sublist in art_ids for i in sublist]
    art_ids = list(set(art_ids))
    # get filenames of downloaded articles
    train_dir = 'train_dir/batch1'
    train_subdir = sorted(os.listdir(train_dir))
    html_files = []
    json_files = []
    for subdir in train_subdir:
        #print('subdir', subdir, 'of', len(train_subdir))
        path = os.path.join(train_dir, subdir)
        files = os.listdir(path)
        h = [f for f in files if 'html' in f]
        j = [f for f in files if 'json' in f]
        html_files.extend(h)
        json_files.extend(j)
    art_ids_json = [i+".json" for i in art_ids]
    diff = set(art_ids_json).difference(set(json_files))
    print("missing articles:", len(diff))
    print("missing article ids:", diff)


if __name__ == '__main__':
    #create_batch2_csv(args.data_file)
    sanity_check()