import os
import pandas as pd
import torch
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

from data import prune_vocabulary2

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', default='', type=str)
argparser.add_argument('--articles_file', default='keywords_lemmatized_2018.csv', type=str)
argparser.add_argument('--sbert_model', default='paraphrase-multilingual-mpnet-base-v2', type=str)
argparser.add_argument('--save_dir', default='bin/results/ctm', type=str)
argparser.add_argument('--num_topics', default=20, type=int)
args = argparser.parse_args()

print("\n" + "-"*5, "Train zero-shot Contextualized Topic Model", "-"*5)
print("data_path:", args.data_path)
print("articles_file:", args.articles_file)
print("sbert_model:", args.sbert_model)
print("save_dir:", args.save_dir)
print("num_topics:", args.num_topics)
print("-"*40 + "\n")

df = pd.read_csv(os.path.join(args.data_path, args.articles_file))
documents_raw = list(df.text)
articles_unproc, articles_proc = prune_vocabulary2(documents_raw)
text_for_contextual = articles_unproc
text_for_bow = articles_proc
#articles_unprocessed = [doc.lower() for doc in list(df.text)]
#articles_processed = [' '.join(clean_document(doc)) for doc in articles_unprocessed]


print('\ntext_for_contextual:', text_for_contextual[:3])
print('text_for_contextual:', len(text_for_contextual))
print('\ntext_for_bow:', text_for_bow[:3])
print('text_for_bow:', len(text_for_bow))

qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")

training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)
print("-"*10, "vocab size:", len(qt.vocab), "-"*10)

ctm = ZeroShotTM(bow_size=len(qt.vocab),
                 contextual_size=768,
                 n_components=args.num_topics,
                 num_epochs=100)

ctm.fit(train_dataset=training_dataset,
        save_dir=args.save_dir)

ctm.get_topics()
