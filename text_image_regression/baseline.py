# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('..'))


# , AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from multiprocessing import Pool
from deeppavlov import configs, build_model
import yake
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
from nltk import pos_tag
import math
from nltk.corpus import stopwords
from enchant.checker import SpellChecker
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np
import json
import argparse
import logging
import random
import torch
import spacy
import os
import sys
from sentence_transformers import (SentenceTransformer,
                                   CrossEncoder)
from sentence_transformers import losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from torch.utils.data import DataLoader

from deeppavlov import configs, build_model
import yake
from utils import evaluate
tqdm.pandas()

#nlp = spacy.load("xx_ent_wiki_sm")

logging.basicConfig(level=logging.ERROR)
manualSeed = 2021

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


##model = SentenceTransformer('paraphrase-mpnet-base-v2')
##sentence_model = SentenceTransformer('stsb-bert-large')
##cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
#cross_encoder = CrossEncoder('castorini/monobert-large-msmarco')
# paraphrase-multilingual-mpnet-base-v2
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_path',
    type=str,
    default='../data/train_dir/batch1',
    help='batch folder')
parser.add_argument(
    '--train_csv',
    type=str,
    default='../data/semeval-2022_task8_train-data_batch1.csv',
    help='data_batch1.csv')
parser.add_argument(
    "--model_name_or_path",
    default='stsb-mpnet-base-v2',
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list ")
parser.add_argument(
    '--train_split',
    type=str,
    default='../data/train_split_batch1.csv',
    help='data_batch1.csv')
parser.add_argument(
    '--test_split',
    type=str,
    default='../data/test_split_batch1.csv',
    help='data_batch1.csv')
parser.add_argument("--do_lower_case", action="store_true",
    help="Set this flag if you are using an uncased model.")
parser.add_argument(
    "--model_save_path",
    default='runs',
    type=str,
    help="Path to pre-trained model or shortcut name selected in the list ")

args = parser.parse_args()
data_path = args.data_path
train_csv = args.train_csv
model_name_or_path = args.model_name_or_path


#
def transform(model, text, add_entities=None):
#    print('---', text)
    sentences = sent_tokenize(text)  # [:3]
#    for idx, sentence in enumerate(sentences):
#        doc = nlp(sentence)
#        entities = [(ent.text, ent.label_) for ent in doc.ents]
#
#        for entity in entities:
#            sentences[idx] = sentences[idx].replace(
#                entity[0], '<' + entity[1] + '> ' + entity[0] +
#                ' </' + entity[1] + '>')
##        print(sentences[idx])
    encodings = model.encode(
        sentences,
        convert_to_tensor=True,
        output_value="token_embeddings")
    sentences = [torch.sum(x, 0) for x in encodings]
    return torch.mean(torch.stack(sentences, 1), 1).cpu().numpy()

#def transform(model, text, add_entities=None):
#    sentences = sent_tokenize(text)#[:3]
#
#    encoding = model.encode(sentences, convert_to_tensor=True)
#
#    return torch.mean(encoding, 0).cpu().numpy()
#

#def transform(model, text, add_entities=None):
#    encoding = model.encode([str(text)], convert_to_tensor=True)[0]
#    return encoding.cpu().numpy()


def get_text(pair_id):
    path = os.path.join(data_path, pair_id[-2:], pair_id + ".json")

    article_text = ''
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        article_text = data['text']
    else:
        print(path)
        
#    article_text = re.sub(r"\n+", ' ', article_text)
#    article_text = re.sub(r"\s+", ' ', article_text)
    
#    article_text = clean(article_text)
    if args.do_lower_case:
        article_text = article_text.lower()
        
    return article_text

ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

max_length = 512

def get_multilingual_entitites(text, max_length=512):
    entities = []
    for sent in sent_tokenize(text):
#        print(sent, len(word_tokenize(sent)))
        try:
            vector = ner_model([' '.join(word_tokenize(sent)[:max_length])])
        except:
#            import pdb;pdb.set_trace()
#            print(sent)
#            vector = ner_model([' '.join(word_tokenize(sent)[:340])])
#            print(max_length)
            max_length = max_length-1
            return get_multilingual_entitites(text, max_length=max_length)
            
        tokens = vector[0][0]
        tags = vector[1][0]

        pos_tags = [pos for token, pos in pos_tag(tokens)]
        conlltags = [(token, pos, tg)
                     for token, pos, tg in zip(tokens, pos_tags, tags)]
        ne_tree = conlltags2tree(conlltags)
        original_text = []
        for subtree in ne_tree:
            # skipping 'O' tags
            if isinstance(subtree, Tree):
                original_label = subtree.label()
                original_string = " ".join(
                    [token for token, pos in subtree.leaves()])
                original_text.append((original_string, original_label))
#        print(original_text)
        entities += original_text
#        import pdb;pdb.set_trace()

    return entities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained(
    'castorini/doc2query-t5-base-msmarco')
model.to(device)

def get_questions(text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_k=10,
        num_return_sequences=3)
    results = []
    for output in outputs[:5]:
        results.append(tokenizer.decode(output, skip_special_tokens=True))
#    for i in range(3):
#        print(
#            f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')
    print(results)
    return results

if __name__ == '__main__':

#    train_df, test_df = split_data.split_train_data_batch1(train_csv=train_csv)
    train_df = pd.read_csv(args.train_split)
    test_df = pd.read_csv(args.test_split)

#    test_df['text1'] = test_df['pair_id'].apply(
#        lambda pair: get_text(pair.split("_")[0]))
#    test_df['text2'] = test_df['pair_id'].apply(
#        lambda pair: get_text(pair.split("_")[1]))
#    train_df['text1'] = train_df['pair_id'].apply(
#        lambda pair: get_text(pair.split("_")[0]))
#    train_df['text2'] = train_df['pair_id'].apply(
#        lambda pair: get_text(pair.split("_")[1]))

    print(train_df.head())

#    from ast import literal_eval
#    test_df = pd.read_csv('../data/test_split_batch2_processed.csv', sep='\t', converters={"entities1": literal_eval, "entities2": literal_eval})
#    print(test_df.head())

#    test_df['questions1'] = test_df['text1'].progress_apply(lambda text : get_questions(text))
#    test_df['entities1'] = test_df['text1'].progress_apply(lambda text : get_multilingual_entitites(text))
#    test_df['entities2'] = test_df['text2'].progress_apply(lambda text : get_multilingual_entitites(text))
#    test_df.to_csv('../data/test_split_batch2_entities.csv')
    
#    train_df['entities1'] = train_df['text1'].progress_apply(lambda text : get_multilingual_entitites(text))
#    train_df['entities2'] = train_df['text2'].progress_apply(lambda text : get_multilingual_entitites(text))
#    train_df.to_csv('../data/train_split_batch2_entities.csv')
    
#
#    true_similarity = np.array(test_df.Overall)
#    
#    L = 0.5
#    for L in [0.1, 0.2, 0.3, 0.4, 0.5]:
#        print('Lambda:', L)
#        pred_similarity = []
#        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
#            if len(str(row.text1).strip()) < 5 or len(str(row.text2).strip()) < 5:
#                pred_similarity.append(0.5)
##                print('Issues', row.pair_id, len(str(row.text1).strip()), len(str(row.text2).strip()))
#            
##                pred_similarity.append(0.5)
#            else:
#                entities1 = ' '.join(['[' + x[1] + '] ' + x[0] for x in row.entities1])
#                entities2 = ' '.join(['[' + x[1] + '] ' + x[0] for x in row.entities2])
#        
##                cosine_sim_entities = distance.cosine(transform(sentence_model, entities1, add_entities=False),
##                                             transform(sentence_model, entities2, add_entities=False))
##                
#                cosine_sim_text = distance.cosine(transform(sentence_model, row.text1, add_entities=False),
#                                             transform(sentence_model, row.text2, add_entities=False))
#            
##                
##                cosine_sim = L * cosine_sim_entities + (1.0-L) * cosine_sim_text
#    
#                pred_similarity.append(cosine_sim_text)
#    #            print(row.pair_id, cosine_sim)
#    #        except BaseException:
#    #            pred_similarity.append(0.5)
##                if len(pred_similarity) > 3 and len(pred_similarity) % 100 == 0:
##                    pearson, p_val = evaluate.evaluate_scores(pred_similarity, true_similarity[:len(pred_similarity)])
##                    print(pearson, p_val)
#    
#    
##        print('pred_similarity:', pred_similarity)
##        print('true_similarity:', true_similarity)
#        pearson, p_val = evaluate.evaluate_scores(pred_similarity, true_similarity)
#    
#        print(pearson, p_val)
    
#    import pdb;pdb.set_trace()
    train_df['Overall'] = train_df['Overall'].progress_map(lambda x: (4.0 - x) / 4.0)
    test_df['Overall'] = test_df['Overall'].progress_map(lambda x: (4.0 - x) / 4.0)

    train_samples = [InputExample(texts=[str(entry[0]), str(entry[1])], label=float(entry[2])) \
                    for entry in zip(train_df.text1, train_df.text2, train_df.Overall)]
    print(len(train_samples))
    train_samples += [InputExample(texts=[str(entry[1]), str(entry[0])], label=float(entry[2])) \
                    for entry in zip(train_df.text1, train_df.text2, train_df.Overall)]
    print(len(train_samples))
    
    test_samples = [InputExample(texts=[str(entry[0]), str(entry[1])], label=float(entry[2])) \
                    for entry in zip(test_df.text1, test_df.text2, test_df.Overall)]
    print(len(test_samples))


    print(train_samples[:5])

    train_batch_size = 4
    num_epochs = 15
    
#    word_embedding_model = models.Transformer(model_name_or_path)

    # Apply mean pooling to get one fixed sized sentence vector
#    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
#                                   pooling_mode_mean_tokens=True,
#                                   pooling_mode_cls_token=False,
#                                   pooling_mode_max_tokens=False)
#    
#    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    model = SentenceTransformer(model_name_or_path)


    train_data_loader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='valid')

    # warm-up step (skip? guess not cuz skip = worse performance)
    warmup_steps = math.ceil(len(train_samples) * num_epochs + 0.1)

    model.fit(train_objectives=[(train_data_loader, train_loss)],
                evaluator=evaluator,
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                evaluation_steps=80,
                output_path=args.model_save_path)
    
    model = SentenceTransformer(args.model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    test_evaluator(model, output_path=args.model_save_path)

    true_similarity = np.array(test_df.Overall)

    pred_similarity = []
    idx = 0
    
    
    text1_embeddings = model.encode([str(x) for x in test_df.text1], convert_to_tensor=True, show_progress_bar=True)
    text2_embeddings = model.encode([str(x) for x in test_df.text2], convert_to_tensor=True, show_progress_bar=True)

    from sklearn.metrics.pairwise import cosine_similarity

    pred_similarity = cosine_similarity(text1_embeddings.cpu().numpy(), text2_embeddings.cpu().numpy())
    
    pearson, p_val = evaluate.evaluate_scores(np.diag(pred_similarity), true_similarity)

    print(pearson, p_val)

#    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
##        try:
#            text1_embeddings = model.encode([str(row.text1)], convert_to_tensor=True, show_progress_bar=False)[0]
#            text2_embeddings = model.encode([str(row.text2)], convert_to_tensor=True, show_progress_bar=False)[0]
#            
##            cosine_sim = util.pytorch_cos_sim(text1_embeddings, text2_embeddings)[0]
#            cosine_sim_text = distance.cosine(text1_embeddings.cpu().numpy(), text2_embeddings.cpu().numpy())
##            cosine_sim = cosine_sim.cpu().numpy()[0]
#
##            cosine_sim = distance.cosine(transform(model, row.text1),
##                                         transform(model, row.text2))
#
#            pred_similarity.append(cosine_sim_text)
#            print(row.pair_id, true_similarity[idx], cosine_sim_text)
#            
#            idx += 1
##            print(row.pair_id, cosine_sim)
##        except BaseException:
#            pred_similarity.append(0.5)
#            print('Issues', row.pair_id, len(row.text1), len(row.text2))

#    print('pred_similarity:', pred_similarity)
#    print('true_similarity:', true_similarity)
#    pearson, p_val = evaluate.evaluate_scores(pred_similarity, true_similarity)
    import pdb;pdb.set_trace()
#    print(pearson, p_val)
#

