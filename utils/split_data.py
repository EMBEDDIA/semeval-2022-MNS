import pandas as pd
import os


# 90/10 split into train/test data
def split_train_data_batch1(train_csv='semeval-2022_task8_train-data_batch.csv'):
    
    
    df = pd.read_csv(train_csv)
    train_split = df[df.url1_lang != 'fr']
    
    # 28 rows of fr data
    test_split = df[df.url1_lang == 'fr']
    
    # shuffle rows of train split
    train_split = train_split.sample(frac=1, random_state=2021)
    
    # take the first 265 rows: 265+28 = 293 (10% 0f 2932)
    test_split = pd.concat([test_split, train_split[:265]])
    train_split = train_split[265:]
    
    print('train size:', train_split.shape[0])
    print('test size:', test_split.shape[0])
    
    # train size: 2646
    # test size: 293
    #train_split.to_csv("train_split_batch1.csv", index=False)
    #test_split.to_csv("test_split_batch1.csv", index=False)
    return train_split, test_split


# check if all articles in the train csv file have been downloaded
def sanity_check():
    
    # get ids of articles
    train_csv = 'semeval-2022_task8_train-data_batch.csv'
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
    
    art_ids_json = [i + ".json" for i in art_ids]
    diff = set(art_ids_json).difference(set(json_files))
    
    print("missing files:", len(diff))
