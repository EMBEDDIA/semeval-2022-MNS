# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModel, get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup)
from modules.transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer
import json
import argparse
from tqdm import tqdm
from torch import cuda
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import os
import random
manualSeed = 2022
from torch import nn 
from sklearn.svm import SVR
from nltk.tokenize import sent_tokenize, word_tokenize

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Creating the customized model
n_heads = 6
head_dims = 128
d_model = n_heads * head_dims
num_layers = 2
feedforward_dim = int(2 * d_model)
trans_dropout = 0.45
attn_type = 'transformer'
after_norm = 1
fc_dropout = 0.4
scale = attn_type == 'transformer'
dropout_attn = None
pos_embed = 'fix'
from nltk.tokenize import word_tokenize

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
LEARNING_RATE = 1e-05


parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, default="../data/train_split_batch2_entities.csv",
                    help='Train file (train.csv)')
parser.add_argument('--test_file', type=str, default="../data/test_split_batch2_entities.csv",
                    help='Test file (test.csv)')
parser.add_argument('--dev_file', type=str, default="../data/test_split_batch2_entities.csv",
                    help='Test file (test.csv)')
parser.add_argument('--test_not_labeled', type=str, default="../../DATA/TREC-IS/test_not_labeled.csv",
                    help='Test file (test.csv)')
parser.add_argument('--out', type=str, default='../data/runs',
                    help='The path of the directory where the *.csv data will be saved')
parser.add_argument('--val_steps', type=int, default=1000,
                    help='val_steps')
parser.add_argument('--batch_size', type=int, default=8,
                    help='val_steps')
parser.add_argument(
    "--adam_beta1",
    default=0.9,
    type=float,
    help="BETA1 for Adam optimizer.")
parser.add_argument(
    "--adam_beta2",
    default=0.999,
    type=float,
    help="BETA2 for Adam optimizer.")
parser.add_argument(
    "--do_lower_case",
    action="store_true",
    help="Set this flag if you are using an uncased model.")
parser.add_argument(
    "--weight_decay",
    default=0.0,
    type=float,
    help="Weight decay if we apply some.")
parser.add_argument("--local_rank", type=int, default=-
                    1, help="For distributed training: local_rank")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--num_train_epochs",
    default=3.0,
    type=float,
    help="Total number of training epochs to perform.")
parser.add_argument(
    "--no_cuda",
    action="store_true",
    help="Avoid using CUDA when available")
parser.add_argument(
    "--do_train",
    action="store_true",
    help="Whether to run training.")
parser.add_argument(
    "--do_eval",
    action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do_predict",
    action="store_true",
    help="Whether to run predictions on the test set.")
parser.add_argument(
    "--learning_rate",
    default=1e-05,
    type=float,
    help="The initial learning rate for Adam.")
parser.add_argument(
    "--cache_dir",
    default="temp",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list ",
)
parser.add_argument(
    "--adam_epsilon",
    default=1e-8,
    type=float,
    help="Epsilon for Adam optimizer.")
parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--save_steps",
    type=int,
    default=50,
    help="Save checkpoint every X updates steps.")

args = parser.parse_args()
train_file = args.train_file
test_file = args.test_file
output_dir = args.out
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.batch_size

import re
from nltk.corpus import stopwords
STOPWORDS = []
for language in ['hungarian',
                 'swedish',
                 'kazakh',
                 'norwegian',
                 'finnish',
                 'arabic',
                 'indonesian',
                 'portuguese',
                 'turkish',
                 'azerbaijani',
                 'slovene',
                 'spanish',
                 'danish',
                 'nepali',
                 'romanian',
                 'greek',
                 'dutch',
                 'tajik',
                 'german',
                 'english',
                 'russian',
                 'french',
                 'italian']:

    STOPWORDS += list(stopwords.words(language))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
    
args.device = device 

print('device', device)


#mlb = MultiLabelBinarizer()
lb = LabelBinarizer()
from ast import literal_eval

train_df = pd.read_csv(train_file, sep=',', converters={"entities1": literal_eval, "entities2": literal_eval,
                                                        "meta_keywords1": literal_eval, "meta_keywords2": literal_eval,
                                                        "keywords1": literal_eval, "keywords2": literal_eval,
                                                        "tags1": literal_eval, "tags2": literal_eval})
test_df = pd.read_csv(test_file, sep=',', converters={"entities1": literal_eval, "entities2": literal_eval,
                                                        "meta_keywords1": literal_eval, "meta_keywords2": literal_eval,
                                                        "keywords1": literal_eval, "keywords2": literal_eval,
                                                        "tags1": literal_eval, "tags2": literal_eval})

dev_df = pd.read_csv(args.dev_file, sep=',', converters={"entities1": literal_eval, "entities2": literal_eval,
                                                        "meta_keywords1": literal_eval, "meta_keywords2": literal_eval,
                                                        "keywords1": literal_eval, "keywords2": literal_eval,
                                                        "tags1": literal_eval, "tags2": literal_eval})

print(train_df.head(10))
train_df['text1'] = train_df['text1'].apply(lambda x: str(x))
train_df['text2'] = train_df['text2'].apply(lambda x: str(x))
test_df['text1'] = test_df['text1'].apply(lambda x: str(x))
test_df['text2'] = test_df['text2'].apply(lambda x: str(x))
dev_df['text1'] = dev_df['text1'].apply(lambda x: str(x))
dev_df['text2'] = dev_df['text2'].apply(lambda x: str(x))

print('Before:', len(train_df))
train_df['length1'] = train_df.text1.str.len()
train_df['length2'] = train_df.text2.str.len()
print(len(train_df[train_df.length1 < 4]), len(train_df[train_df.length1 < 4]))

test_df['length1'] = test_df.text1.str.len()
test_df['length2'] = test_df.text2.str.len()
print(len(test_df[test_df.length1 < 4]), len(test_df[test_df.length1 < 4]))

test_df['id1'] = test_df['pair_id'].apply(
    lambda pair: int(pair.split("_")[0]))
test_df['id2'] = test_df['pair_id'].apply(
    lambda pair: int(pair.split("_")[1]))
train_df['id1'] = train_df['pair_id'].apply(
    lambda pair: int(pair.split("_")[0]))
train_df['id2'] = train_df['pair_id'].apply(
    lambda pair: int(pair.split("_")[1]))
dev_df['id1'] = dev_df['pair_id'].apply(
    lambda pair: int(pair.split("_")[0]))
dev_df['id2'] = dev_df['pair_id'].apply(
    lambda pair: int(pair.split("_")[1]))


with open('../data/train_split_batch2_images_clip.csv', 'r') as f:
    train_image_embeddings = f.readlines()[1:]
    train_ids = [int(x.split(',')[-1].strip()) for x in train_image_embeddings]
    train_image_embeddings = [np.array([float(x) for x in y.split(',')][:-1]) for y in train_image_embeddings]
with open('../data/test_split_batch2_images_clip.csv', 'r') as f:
    test_image_embeddings = f.readlines()[1:]
    test_ids = [int(x.split(',')[-1].strip()) for x in test_image_embeddings]
    test_image_embeddings = [np.array([float(x) for x in y.split(',')][:-1]) for y in test_image_embeddings]

google_train_df = pd.read_csv('../data/train_split_batch2_google_translated.csv')
google_test_df = pd.read_csv('../data/test_split_batch2_google_translated.csv')

google_train_df['id'] = google_train_df['id'].apply(lambda x: int(x))
google_test_df['id'] = google_test_df['id'].apply(lambda x: int(x))

bart_df = pd.read_csv('../data/semeval-2022_task8_train-data_batch2_retry_translated_mbart.csv')
#bart_train_df = pd.read_csv('../data/train_split_batch2_translated_mbart.csv')
#bart_test_df = pd.read_csv('../data/test_split_batch2_translated_mbart.csv')

embeddings1, embeddings2 = [], []
google_translations1, bart_translations1 = [], []
google_translations2, bart_translations2 = [], []
for id1, id2 in zip(train_df.id1, train_df.id2):
    try:
      google_translations1.append(google_train_df[google_train_df['id'] == id1].trg_text.tolist()[0])
    except:
#      print(train_df[train_df['id1'] == id1])
      google_translations1.append(" ")
    try:
      google_translations2.append(google_train_df[google_train_df['id'] == id2].trg_text.tolist()[0])
    except:
#      print(train_df[train_df['id2'] == id2])
      google_translations2.append(" ")
    try:
      bart_translations1.append(bart_df[bart_df['id'] == id1].trg_text.tolist()[0])
    except:
#      print(train_df[train_df['id1'] == id1])
      bart_translations1.append(" ")
    try:
      bart_translations2.append(bart_df[bart_df['id'] == id2].trg_text.tolist()[0])
    except:
#      print(train_df[train_df['id2'] == id2])
      bart_translations2.append(" ")
    if id1 in train_ids:
        embeddings1.append(train_image_embeddings[train_ids.index(id1)])
    else:
        embeddings1.append(np.zeros(512))
    if id2 in train_ids:
        embeddings2.append(train_image_embeddings[train_ids.index(id2)])
    else:
        embeddings2.append(np.zeros(512))

#print(embeddings2[:4])
train_df['embeddings1'] = embeddings1
train_df['embeddings2'] = embeddings2
train_df['google1'] = google_translations1
train_df['google2'] = google_translations2
train_df['bart1'] = bart_translations1
train_df['bart2'] = bart_translations2

embeddings1, embeddings2 = [], []
google_translations1, bart_translations1 = [], []
google_translations2, bart_translations2 = [], []
for id1, id2 in zip(test_df.id1, test_df.id2):
    try:
      google_translations1.append(google_test_df[google_test_df['id'] == id1].trg_text.tolist()[0])
    except:
      google_translations1.append(" ")
#      print(test_df[test_df['id1'] == id1])
    try:
      google_translations2.append(google_test_df[google_test_df['id'] == id2].trg_text.tolist()[0])
    except:
      google_translations2.append(" ")
#      print(test_df[test_df['id2'] == id2])
    try:
      bart_translations1.append(bart_df[bart_df['id'] == id1].trg_text.tolist()[0])
    except:
      bart_translations1.append(" ")
#      print(test_df[test_df['id1'] == id1])
    try:
      bart_translations2.append(bart_df[bart_df['id'] == id2].trg_text.tolist()[0])
    except:
      bart_translations2.append(" ")
#      print(test_df[test_df['id2'] == id2])
    if id1 in test_ids:
        embeddings1.append(test_image_embeddings[test_ids.index(id1)])
    else:
        embeddings1.append(np.zeros(512))
    if id2 in test_ids:
        embeddings2.append(test_image_embeddings[test_ids.index(id2)])
    else:
        embeddings2.append(np.zeros(512))

test_df['embeddings1'] = embeddings1
test_df['embeddings2'] = embeddings2
test_df['google1'] = google_translations1
test_df['google2'] = google_translations2
test_df['bart1'] = bart_translations1
test_df['bart2'] = bart_translations2

LABELS = ['Geography', 'Entities', 'Time', 'Narrative', 'Overall', 'Style', 'Tone']
#LABELS = ['Overall']
#for LABEL in LABELS:
#    train_df[LABEL] = train_df[LABEL].apply(lambda x: round(x)-1)
#    test_df[LABEL] = test_df[LABEL].apply(lambda x: round(x)-1)
#
print(train_df['Overall'].unique())
print(test_df['Overall'].unique())
#lb.fit(train_df['Overall'])
#for LABEL in LABELS:    
#    train_df[LABEL] = [y for y in lb.transform(train_df[LABEL])]
#    test_df[LABEL] = [y for y in lb.transform(test_df[LABEL])]

def text_cleaner(tweet):
#    import pdb;pdb.set_trace()
    tweet = re.sub(r"@\w*", " ", str(tweet)).strip() #removing username
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', " ", str(tweet)).strip() #removing links
    tweet = re.sub(r'[^a-zA-Z]', " ", str(tweet)).strip() #removing sp_char
    tw = []
    
    for text in tweet.split():
        if text not in STOPWORDS:
            if not tw.startwith('@') and tw != 'RT':
                tw.append(text)
#    print( " ".join(tw))
    tw = re.sub(r"\s+", '-', ' '.join(tw))
    return tw

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

#IMAGE_MODEL = 'google/vit-large-patch32-224-in21k'
IMAGE_MODEL = 'google/vit-base-patch16-224'
feature_extractor = ViTFeatureExtractor.from_pretrained(IMAGE_MODEL)
empty_image_url = 'https://m.media-amazon.com/images/I/51UW1849rJL._AC_SX679_.jpg'
empty_image = Image.open(requests.get('https://m.media-amazon.com/images/I/51UW1849rJL._AC_SX679_.jpg', stream=True).raw)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        
#        print('self.data.text1', self.data.text1)
#        print('self.data.text2', self.data.text2)
#        self.comment_text1 = ' '.join(sent_tokenize(self.data.text1)) 
#        self.comment_text2 = ' '.join(sent_tokenize(self.data.text2))
        
        self.comment_text1 = self.data.text1
        self.comment_text2 = self.data.text2
        self.entities1 = self.data.entities1
        self.entities2 = self.data.entities2
        self.images1 = self.data.embeddings1
        self.images2 = self.data.embeddings2
        self.google1 = self.data.google1
        self.google2 = self.data.google2
        self.bart1 = self.data.bart1
        self.bart2 = self.data.bart2
        self.meta_keywords1 = self.data.meta_keywords1
        self.meta_keywords2 = self.data.meta_keywords2
        self.title1 = self.data.title1
        self.title2 = self.data.title2
        self.topic1 = self.data.topic1
        self.topic2 = self.data.topic2
        self.pair_id = self.data.pair_id
        
        self.top_images1 = self.data.top_images1
        self.top_images2 = self.data.top_images2
        self.tags1 = self.data.tags1
        self.tags2 = self.data.tags2
        self.keywords1 = self.data.keywords1
        self.keywords2 = self.data.keywords2


        self.targets = {}
        for LABEL in LABELS:
            if LABEL not in self.targets:
                self.targets[LABEL] = []
            if LABEL not in self.data:
                self.targets[LABEL] = [0] * len(self.data)
            else:
                self.targets[LABEL] = self.data[LABEL]
        
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text1+self.comment_text2)
    
    def _search_image(self, id_image):
        
        if os.path.exists('../data/batch2_images/' + str(id_image) + '.jpg'):
            return '../data/batch2_images/' + str(id_image) + '.jpg'
        elif os.path.exists('../data/batch2_images/' + str(id_image) + '.jpeg'):
            return '../data/batch2_images/' + str(id_image) + '.jpeg'
        elif os.path.exists('../data/batch2_images/' + str(id_image) + '.png'):
            return '../data/batch2_images/' + str(id_image) + '.png'
        elif os.path.exists('../data/batch2_images/img_' + str(id_image)):
            return '../data/batch2_images/img_' + str(id_image)
        else:
            print(id_image, 'not_found')
            return '../data/batch2_images/img_000000'
        
    def __getitem__(self, index):

        #import pdb;pdb.set_trace()
#        entities1 = ' '.join([ent_type + ' ' + ent_text for ent_type, ent_text in self.entities1[index]])
        comment_text1 = str(self.comment_text1[index]) #str(self.comment_text1[index])
        google1 = str(self.google1[index]) #str(self.comment_text1[index])
        google1 = " ".join(google1.split()) #+ #' ' + entities1
        bart1 = str(self.bart1[index]) #str(self.comment_text1[index])
        bart1 = " ".join(bart1.split()) #+ #' ' + entities1
        
        comment_text1 = str(self.title1[index]) + ' ' + " ".join(comment_text1.split()) + ' ' +  \
                str(' '.join(self.meta_keywords1[index])) + ' ' + str(self.topic1[index]) + ' '+ \
                ' ' + str(self.tags1[index]) + ' ' + str(self.keywords1[index])

        image1 = self.images1[index]
        image2 = self.images2[index]
        
        #f_name = 'data/batch2_images/img_{}'.format(id_article)
        id1 = self.pair_id[index].split("_")[0]
        id2 = self.pair_id[index].split("_")[1]
        
        
        path1 = self._search_image(id1)
        path2 = self._search_image(id2)
        
#        print(path1, path2)
#        print(self.top_images1[index], self.top_images2[index])
        try:
            top_image1 = Image.open(path1)
        except:
#            print('--', path1)
            top_image1 = Image.open('../data/batch2_images/img_000000')
          
        try:
            top_image2 = Image.open(path2)
        except:
#            print('--', path2)
#            import pdb;pdb.set_trace()
            top_image2 = Image.open('../data/batch2_images/img_000000')
        
        try:
            top_image1 = feature_extractor(images=top_image1, return_tensors="pt")['pixel_values'].squeeze()
        except:
            top_image1 = Image.open('../data/batch2_images/img_000000')
            top_image1 = torch.zeros((3, 224, 224), dtype=torch.float)#feature_extractor(images=top_image1, return_tensors="pt")['pixel_values'].squeeze()

        try:
            top_image2 = feature_extractor(images=top_image2, return_tensors="pt")['pixel_values'].squeeze()          
        except:
            top_image2 = Image.open('../data/batch2_images/img_000000')
            top_image2 = torch.zeros((3, 224, 224), dtype=torch.float)#feature_extractor(images=top_image2, return_tensors="pt")['pixel_values'].squeeze()          

#        entities2 = ' '.join([ent_type + ' ' + ent_text for ent_type, ent_text in self.entities2[index]])
        comment_text2 = str(self.comment_text2[index]) #str(self.comment_text2[index])
        google2 = str(self.google2[index]) #str(self.comment_text1[index])
        google2 = " ".join(google2.split()) #+ #' ' + entities1
        bart2 = str(self.bart2[index]) #str(self.comment_text1[index])
        bart2 = " ".join(bart2.split()) #+ #' ' + entities1
        
        comment_text2 = str(self.title2[index]) + ' ' + " ".join(comment_text2.split()) + ' ' + \
                str(' '.join(self.meta_keywords2[index])) + ' ' + str(self.topic2[index]) + ' '+ \
                ' ' + str(self.tags1[index]) + ' ' + str(self.keywords2[index])

#        entities = []
#        for ent_text, ent_type in self.entities1[index]:
#            if ent_text not in entities:
#               bart1 = bart1.replace(ent_text, '[' + ent_type + '] ' + ent_text)
#               entities.append(ent_text)
#        #print(entities)
#        entities = []
#        for ent_text, ent_type in self.entities2[index]:
#            if ent_text not in entities:
#                bart2 = bart2.replace(ent_text, '[' + ent_type + '] ' + ent_text)             
#                entities.append(ent_text)
        
        #print(entities)
        
#        import pdb;pdb.set_trace()
        
        inputs = self.tokenizer.encode_plus(
            comment_text1,
            comment_text2,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
#        import pdb;pdb.set_trace()

        return_result = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'image1': torch.tensor(image1, dtype=torch.float),
            'image2': torch.tensor(image2, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'top_image1': top_image1,
            'top_image2': top_image2
        }

        for LABEL in LABELS:
            return_result[LABEL] = torch.tensor(self.targets[LABEL][index], dtype=torch.float)

        return return_result


    def get_tokenizer(self):
        return self.tokenizer

# Creating the dataset and dataloader for the neural network

#train_size = 0.8
#train_dataset=new_df.sample(frac=train_size, random_state=200)
# test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
#train_dataset = train_dataset.reset_index(drop=True)


print("TRAIN Dataset: {}".format(train_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

#def create_optimizer(model):
#    named_parameters = list(model.named_parameters())    
#    
#    roberta_parameters = named_parameters[:197]    
#    attention_parameters = named_parameters[199:203]
#    regressor_parameters = named_parameters[203:]
#        
#    attention_group = [params for (name, params) in attention_parameters]
#    regressor_group = [params for (name, params) in regressor_parameters]
#
#    parameters = []
#    parameters.append({"params": attention_group})
#    parameters.append({"params": regressor_group})
#
#    for layer_num, (name, params) in enumerate(roberta_parameters):
#        weight_decay = 0.0 if "bias" in name else 0.01
#
#        lr = LEARNING_RATE
#
#        if layer_num >= 69:        
#            lr = LEARNING_RATE * 2.5
#
#        if layer_num >= 133:
#            lr = LEARNING_RATE * 5
#
#        parameters.append({"params": params,
#                           "weight_decay": weight_decay,
#                           "lr": lr})
#
#    return torch.optim.AdamW(parameters)

class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class CFG:
    debug = False
    captions_path = "."
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "bert-base-multilingual-uncased"
    text_embedding = 768
    text_tokenizer = "bert-base-multilingual-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 512 
    dropout = 0.1
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

from sadice import SelfAdjDiceLoss
criterion = SelfAdjDiceLoss()

#def loss_fn(outputs, targets):
#    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def loss_fn(outputs, targets):
    return torch.nn.MSELoss()(outputs, targets.view(-1, 1))

import clip
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig

class BERTClass(torch.nn.Module):

    config_class = AutoConfig

    def __init__(self, config, tokenizer):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        self.bert = AutoModel.from_pretrained(args.model_name_or_path,
                                                              config=config)
        
        self.bert.resize_token_embeddings(len(tokenizer))
        
        configuration = ViTConfig(IMAGE_MODEL)
        self.vit = ViTModel.from_pretrained(IMAGE_MODEL) 

        self.dropout = torch.nn.Dropout(0.3)
        self.transformer = TransformerEncoder(num_layers, d_model, n_heads,
                                              feedforward_dim, trans_dropout,
                                              after_norm=after_norm,
                                              attn_type=attn_type,
                                              scale=scale,
                                              dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
#        self.len_bert = 768
#        self.image_model, self.preprocess = clip.load(" ViT-H/14 ", device=device)
#        
#        self.preprocess = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
#        self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.in_fc = torch.nn.Linear(self.bert.config.hidden_size, d_model)
        self.fc_dropout = torch.nn.Dropout(fc_dropout)
#        self.pooler = ContextPooler(config)
#        output_dim = self.pooler.output_dim

        self.in_fc = torch.nn.Linear(self.bert.config.hidden_size, d_model)
        self.fc_dropout = torch.nn.Dropout(fc_dropout)
        

        self.self_attn = MultiHeadAttn(d_model, n_heads)
        self.head = AttentionHead(self.bert.config.hidden_size)

        self.drop = nn.Dropout(0.3)
        
        # convolutional layer
        self.conv1 = nn.Conv1d(MAX_LEN, 128, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=3)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=3)
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool1d(3)
        #self.fc_conv = nn.Linear(257,2)
        self.fc_conv = nn.Linear(30, 2)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        
        self.temperature = CFG.temperature
        # dense layer 2 (Output layer)
        MAX_LABEL = 4
        self.linear_types = nn.ModuleList()
        for LABEL in LABELS:
#            self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size, MAX_LABEL))
#            self.linear_types.append(torch.nn.Linear(512*2, MAX_LABEL))
#            self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size + 512*2, MAX_LABEL))
          
#            self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size *(MAX_LEN+1), MAX_LABEL))
#            self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size *(MAX_LEN+1), 1))
#            self.linear_types.append(torch.nn.Linear(512, 1))
#            self.linear_types.append(torch.nn.Linear(768*2, 1))
            self.linear_types.append(torch.nn.Linear(1024*(MAX_LEN+1), 1))
#            self.linear_types.append(torch.nn.Linear(768*(MAX_LEN+1), 1))
#            self.linear_types.append(torch.nn.Linear(768*(MAX_LEN+3), 1))
#            self.linear_types.append(torch.nn.Linear(768*(MAX_LEN+2), 1))
            
#            self.linear_types.append(torch.nn.Linear(263680, 1))
#            self.linear_types.append(torch.nn.Linear(132608, MAX_LABEL))
        self.image_projection = ProjectionHead(embedding_dim=512*2)
#        self.image_projection = ProjectionHead(embedding_dim=512)
        self.text_projection = ProjectionHead(embedding_dim=self.bert.config.hidden_size)
#        self.text_projection = ProjectionHead(embedding_dim=512)
#        self.text_doc_projection = ProjectionHead(embedding_dim=self.bert.config.hidden_size)
        self.image_projection1 = torch.nn.Linear(512*2, 768)
        self.image_projection2 = torch.nn.Linear(512*2, 768)

#        self._init_weights(self)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, ids, mask, image1, image2, top_image1, top_image2, token_type_ids):

        element = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        doc, bert_outputs = element[0], element[1]
        
#        chars_doc = self.in_fc(doc)
#        mask_doc = mask.ne(0)
#        chars_doc = self.transformer(chars_doc, mask_doc)
#        doc = self.fc_dropout(chars_doc)
##        doc = self.attn(doc)
        output = self.dropout(bert_outputs)
##        output1 = self.dropout(bert_outputs1)
#        output = torch.cat([output, image1, image2], 1)
#        output = self.image_projection(torch.cat([image1, image2], 1))
#        image_embeddings2 = self.image_projection(image2)
#        text_embeddings = self.text_projection(output)
#        text_doc_embeddings = self.text_projection(doc)

#        output = torch.cat([image_embeddings1.unsqueeze(1), image_embeddings2.unsqueeze(1)], 1).view(output.shape[0], -1)
#        top_image1 = self.vit(top_image1)
#        top_image2 = self.vit(top_image2)
        
#        image_embeddings1 = self.image_projection1(top_image1['pooler_output'])
#        image_embeddings2 = self.image_projection2(top_image2['pooler_output'])
#        image_embeddings1 = top_image1['pooler_output']
#        image_embeddings2 = top_image2['pooler_output']
        
#        image_embedding = torch.add(image_embeddings1.unsqueeze(1), image_embeddings2.unsqueeze(1))
        
#        import pdb;pdb.set_trace()
#        output = torch.cat([image_embeddings1.unsqueeze(1), image_embeddings2.unsqueeze(1)], 1).view(output.shape[0], -1)
#        output = torch.cat([output.unsqueeze(1), doc, image_embeddings1.unsqueeze(1), image_embeddings2.unsqueeze(1)], 1).view(output.shape[0], -1)
        output = torch.cat([output.unsqueeze(1), doc], 1).view(output.shape[0], -1)
        
#        import pdb;pdb.set_trace()
#        output = torch.cat([image_embeddings1.unsqueeze(1), image_embeddings2.unsqueeze(1), text_embeddings.unsqueeze(1), text_doc_embeddings], 1).view(output.shape[0], -1)
#        output = torch.cat([image1.unsqueeze(1), image2.unsqueeze(1), text_embeddings.unsqueeze(1), text_doc_embeddings], 1).view(output.shape[0], -1)
#        import pdb;pdb.set_trace()
#        output = torch.cat([output.unsqueeze(1), doc], 1).view(output.shape[0], -1)
#        output = torch.cat([self.attn(doc).unsqueeze(1), output.unsqueeze(1), doc], 1).view(output.shape[0], -1)
        # Getting Image and Text Embeddings (with same dimension)
#        
#        output = torch.cat([image_embeddings, text_embeddings], 1)
        # Calculating the Loss
#        logits = (text_embeddings @ image_embeddings.T) / self.temperature
#        images_similarity = image_embeddings @ image_embeddings.T
#        texts_similarity = text_embeddings @ text_embeddings.T
#        targets = torch.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
#
#        import pdb;pdb.set_trace()

        output_types = []
        for idx, LABEL in enumerate(LABELS):
#            import pdb;pdb.set_trace()
            output_types.append(self.linear_types[idx](output))
        
#        doc = self.head(doc)
        
        return output_types



def evaluation(model, epoch):
    model.eval()
#    
#    fin_targets_probs = []
    
    fin_targets_types = {}
    fin_output_types = {}
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader), total=len(testing_loader)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            image1 = data['image1'].to(device, dtype=torch.float)
            image2 = data['image2'].to(device, dtype=torch.float)
            top_image1 = data['top_image1'].to(device, dtype=torch.float)
            top_image2 = data['top_image2'].to(device, dtype=torch.float)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            
            for LABEL in LABELS:
                if LABEL not in fin_targets_types:
                    fin_targets_types[LABEL] = []
                fin_targets_types[LABEL] += data[LABEL].cpu().detach().numpy().tolist()

            output_types = model(ids, mask, image1, image2, top_image1, top_image2, token_type_ids)

#            fin_targets_types.extend(targets.cpu().detach().numpy().tolist())
            
#            fin_targets_priorities.extend(torch.argmax(
#                priorities, -1).cpu().detach().numpy().tolist())
            
#            fin_targets_priorities.extend(priorities.cpu().detach().numpy().tolist())
            
            for idx, LABEL in enumerate(LABELS):
                if LABEL not in fin_output_types:
                    fin_output_types[LABEL] = []
#                import pdb;pdb.set_trace()
#                fin_output_types[LABEL] += torch.argmax(torch.softmax(output_types[idx], dim=1), dim=1).cpu().detach().numpy().tolist()
                fin_output_types[LABEL] += output_types[idx].reshape(-1).cpu().detach().numpy().tolist()
            
#            fin_output_types.extend(torch.sigmoid(output_types).cpu().detach().numpy().tolist())
         
#    for threshold in [0.5]:
    max_f1 = 0.0
    for LABEL in LABELS:
        if 'Over' in LABEL:
          print('Evaluation', LABEL, 'Epoch:', epoch)
          output_types, target_types = fin_output_types[LABEL], fin_targets_types[LABEL]
          
#          accuracy = metrics.accuracy_score(target_types, output_types)
#          f1_score_micro = metrics.f1_score(
#              target_types, output_types, average='micro')
#          f1_score_macro = metrics.f1_score(
#              target_types, output_types, average='macro')
#          
#          if f1_score_micro > max_f1:
#              max_f1 = f1_score_micro
#  #            thresh = threshold
#              
#          print(f"Accuracy Score {LABEL} = {accuracy}")
#          print(f"Best F1 Score (Micro) {LABEL} = {max_f1}")
#          print(f"F1 Score (Micro) {LABEL} = {f1_score_micro}")
#          print(f"F1 Score (Macro) {LABEL} = {f1_score_macro}")
#          print(
#              metrics.classification_report(
#                  target_types,
#                  output_types,
#                  digits=4))
    
    from scipy.stats import pearsonr

#    pred_scores = np.array([float(x) + 1 for x in fin_output_types['Overall']])
    pred_scores = np.array([float(x) for x in fin_output_types['Overall']])
    true_scores = np.array(fin_targets_types['Overall'])
    print(pred_scores[:10])
    print(true_scores[:10])
    pearsonr_corr = pearsonr(pred_scores, true_scores)
    print("Pearson-r:", pearsonr_corr[0])
    print("p-value:", pearsonr_corr[1])
    
#    import pdb;pdb.set_trace()
    test_df['PredictedOverall'] = pred_scores
    
#    test_df.drop(columns=['embeddings1', 'embeddings2'], inplace=True)
    
    test_df.to_csv(os.path.join(output_dir, 'eval_predictions.csv'))

    return fin_output_types, fin_targets_types


def train(model, optimizer, scheduler, epoch):
    model.train()
    
    for step, data in tqdm(enumerate(training_loader),
                        total=len(training_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        image1 = data['image1'].to(device, dtype=torch.float)
        image2 = data['image2'].to(device, dtype=torch.float)
        top_image1 = data['top_image1'].to(device, dtype=torch.float)
        top_image2 = data['top_image2'].to(device, dtype=torch.float)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        
        output_types = model(ids, mask, image1, image2, top_image1, top_image2, token_type_ids)

        losses = []
        targets = {}
        for idx, LABEL in enumerate(LABELS):
            if LABEL not in targets:
                targets[LABEL] = []
            targets[LABEL] = data[LABEL].to(device, dtype=torch.float)
            losses.append(loss_fn(output_types[idx], targets[LABEL]))

        
        optimizer.zero_grad()
#        losses = [loss_fn(output_types, targets), loss_fct(output_priorities, priorities)]
#        loss = loss_fn(output_types, targets)
        loss = sum(losses)
        
        if step % args.val_steps == 0 and step > 0:
            evaluation(model, "train_step{}_epoch{}".format(step, epoch))
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
#            
        if step % args.save_steps == 0:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
#            model_to_save.save_pretrained(output_dir)
            
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pth'))
            
            tokenizer.save_pretrained(output_dir)
            torch.save(
                args, os.path.join(
                    output_dir, "training_args.bin"))
            torch.save(
                model.state_dict(), os.path.join(
                    output_dir, "model.pt"))
            torch.save(
                optimizer.state_dict(), os.path.join(
                    output_dir, "optimizer.pt"))
            torch.save(
                scheduler.state_dict(), os.path.join(
                    output_dir, "scheduler.pt"))
            print(
                "Saving optimizer and scheduler states to %s", output_dir)


        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return model
    
if args.do_train:
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                              do_lower_case=args.do_lower_case, truncation=True)
    
    import string
    punctuation = list(string.punctuation)
    punctuation.append('``')
    punctuation.append("'s")
    punctuation.append("''")
    punctuation.append("--")
    punctuation.append('’') 
    punctuation.append('“')
    punctuation.append('”')

    added_tokens = {}
#    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#        for text in [row.bart1, row.bart2]: 
#            if len(str(text).strip()) > 5:
#                for word in word_tokenize(text):
#                    word_pieces = tokenizer.tokenize(word)
#                    if len(word_pieces) > 1:
#                        if word not in punctuation:
#                            if word not in added_tokens:
#                                added_tokens[word] = 1
#                            else:
#                                added_tokens[word] += 1
#                            print(word)
#    print(len(added_tokens))
#    if len(added_tokens) >  1:added_tokens
#        print("[ BEFORE ] tokenizer vocab size:", len(tokenizer)) 
##        import pdb;pdb.set_trace()
#        added_tokens = tokenizer.add_tokens(added_tokens)
#        print("[ AFTER ] tokenizer vocab size:", len(tokenizer)) 
#        print('added_tokens:', added_tokens)
                            
#    import pdb;pdb.set_trace()
    
    from nltk import FreqDist
    freq_dist = FreqDist(added_tokens)
    print(freq_dist.most_common(100))
    #freq_dist = FreqDist(added_tokens)
    print(len(added_tokens)) 
    if len(added_tokens) >  1:
        print("[ BEFORE ] tokenizer vocab size:", len(tokenizer)) 
        added_tokens = tokenizer.add_tokens([x[0] for x in freq_dist.most_common(100)])
        print("[ AFTER ] tokenizer vocab size:", len(tokenizer)) 
        print('added_tokens:', added_tokens)


    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }
    
    training_set = CustomDataset(train_df, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_df, tokenizer, MAX_LEN)

#    train_sampler = RandomSampler(train_data)
#    val_sampler = SequentialSampler(val_data)
#    
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)


    config = AutoConfig.from_pretrained(args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = BERTClass(config, tokenizer)
#    model.resize_token_embeddings(len(tokenizer))

#    model = BERTClass.from_pretrained(
#        args.model_name_or_path,
#        from_tf=bool(".ckpt" in args.model_name_or_path),
#        config=config,
#        cache_dir=args.cache_dir if args.cache_dir else None,
#    )

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], "weight_decay": args.weight_decay, }, {
            "params": [
                p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0}, ]
    
    t_total = len(training_loader) // args.gradient_accumulation_steps * args.num_train_epochs
#                
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(
            args.adam_beta1,
            args.adam_beta2))
    
#    optimizer = create_optimizer(model)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

#    from schedulers import CyclicLR
#    scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)

    for epoch in range(int(args.num_train_epochs)):
        _ = train(model, optimizer, scheduler, epoch)
        _, _ = evaluation(model, epoch)

if args.do_predict:

    config = AutoConfig.from_pretrained(args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        do_lower_case=args.do_lower_case)

    testing_set = CustomDataset(test_df, tokenizer, MAX_LEN)
   
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 10
                   }
    
    testing_loader = DataLoader(testing_set, **test_params)
    model = BERTClass(config)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pth')))
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    evaluation(model, 'final', None, None)
    
# for epoch in range(EPOCHS):
#    import pdb;pdb.set_trace()
