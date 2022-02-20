"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--sbert_model', default='paraphrase-multilingual-mpnet-base-v2', type=str)
argparser.add_argument('--sts_train_data', default='semeval_sts_translated_mbart.csv', type=str)
argparser.add_argument('--num_epochs', default=50, type=int)
args = argparser.parse_args()


print("\n" + "-"*5, "Fine-tuning SBERT", "-"*5)
print("sbert_model:", args.sbert_model)
print("sts_train_data:", args.sts_train_data)
print("num_epochs:", args.num_epochs)
print("-"*40 + "\n")

def create_sts_dataset():
    articles_files = ["batch2_google_translated_train.csv", "batch2_google_translated_test.csv"]
    pairs_files = ["train_split_batch2.csv", "test_split_batch2.csv"]
    sts_dataset = {'split': [],
                   'sentence1': [],
                   'sentence2': [],
                   'score': []}
    for file_index in range(len(articles_files)):
        articles_df = pd.read_csv(articles_files[file_index])
        articles_df = articles_df.dropna()
        pairs_df = pd.read_csv(pairs_files[file_index])
        pairs_df = pairs_df.dropna()
        pairs = list(pairs_df.pair_id)
        overall_scores = list(pairs_df.Overall)
        # overall_scores = [float(5.0 - score) for score in overall_scores]
        # reverse and normalise scores from [0, 1] (least to most similar)
        overall_scores = [float(3.0 - (score - 1.0))/3.0 for score in overall_scores]
        for index, pair in enumerate(pairs):
            id1 = int(pair.split('_')[0])
            id2 = int(pair.split('_')[1])
            res1 = articles_df[articles_df.id == id1]
            res2 = articles_df[articles_df.id == id2]
            if res1.shape[0] > 0 and res2.shape[0] > 0:
                sent1 = " ".join(res1.iloc()[0]["trg_text"].lower().split())
                sent2 = " ".join(res2.iloc()[0]["trg_text"].lower().split())
                # normalised_score = (overall_scores[index] - 1.0) / 3.0
                normalised_score = overall_scores[index]
                split_name = 'train' if 'train' in articles_files[file_index] else 'test'
                sts_dataset['split'].append(split_name)
                sts_dataset['sentence1'].append(sent1)
                sts_dataset['sentence2'].append(sent2)
                sts_dataset['score'].append(normalised_score)
    sts_dataset = pd.DataFrame.from_dict(sts_dataset)
    print("STS dataset rows:", sts_dataset.shape[0])
    outfilename = "semeval_sts_translated_google.csv"
    sts_dataset.to_csv(outfilename, index=False)
    print("Saved STS dataset as", sts_dataset)
    return sts_dataset


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
# sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
#
# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)




# Read the dataset
# model_name = 'nli-distilroberta-base-v2'
# train_batch_size = 16
# num_epochs = 4
# model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

data_path = "/users/zosaelai/project_dir/datasets/semeval-multilingual-news/"
sts_filename = args.sts_train_data
sts_dataset_path = os.path.join(data_path, sts_filename)

model_name = args.sbert_model
train_batch_size = 16
num_epochs = args.num_epochs

models_dir = "/users/zosaelai/project_dir/elaine/semeval-multilingual-news/trained_models"
model_save_name = "semeval_translated_sts-" + model_name
model_save_path = os.path.join(models_dir, model_save_name)


# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
df = pd.read_csv(sts_dataset_path)
for index, row in df.iterrows():
    score = float(row['score'])
    inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
    if row['split'] == 'dev':
        dev_samples.append(inp_example)
    elif row['split'] == 'test':
        test_samples.append(inp_example)
    else:
        train_samples.append(inp_example)


# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         #score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
#         score = float(row['score'])
#         inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
#
#         if row['split'] == 'dev':
#             dev_samples.append(inp_example)
#         elif row['split'] == 'test':
#             test_samples.append(inp_example)
#         else:
#             train_samples.append(inp_example)



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
#evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)