# -*- coding: utf-8 -*-
import re
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
from utils import evaluate_scores
import spacy
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

#from nltk.tokenize import sent_tokenize
tqdm.pandas()
#nlp = spacy.load("en_core_web_sm")
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
    default='../data/train_dir/batch2',
    help='batch folder')
parser.add_argument(
    '--train_csv',
    type=str,
    default='../data/semeval-2022_task8_train-data_batch2.csv',
    help='data_batch1.csv')
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
parser.add_argument(
    "--model_name_or_path",
    default='stsb-mpnet-base-v2',
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list ")

args = parser.parse_args()
data_path = args.data_path
train_csv = args.train_csv
model_name_or_path = args.model_name_or_path


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained(
    'castorini/doc2query-t5-base-msmarco')
model.to(device)

##
#def transform(model, text, add_entities=False):
#    #    import pdb;pdb.set_trace()
#
#    text = str(text)
##    print('----', text)
#    if len(text.strip()) < 2:
#        text = 'empty'
#    text = text.replace('\n', ' ').strip()
##    sentences = sent_tokenize(text)[:3]+sent_tokenize(text)[-3:]
#    sentences = sent_tokenize(text)
##    if add_entities:
##        for idx, sentence in enumerate(sentences):
##            doc = nlp(sentence)
##            entities = [(ent.text, ent.label_) for ent in doc.ents]
##
###            sentences[idx] = ''
##            for entity in entities:
## sentences[idx] = sentences[idx].replace(
## entity[0], '<' + entity[1] + '> ' + entity[0] +
## ' </' + entity[1] + '>')
##                sentences[idx] += entity[0] + ' ' + entity[1] + ' '
##            if len(sentences[idx]) < 2:
##                sentences[idx] = 'No entities'
##            print(sentences[idx])
#
#    encodings = model.encode(
#        sentences,
#        convert_to_tensor=True,
#        output_value="token_embeddings")
#    sentences = [torch.sum(x, 0) for x in encodings]
#
#    return torch.mean(torch.stack(sentences, 1), 1).cpu().numpy()
####

def transform(model, text, add_entities=None):
    encoding = model.encode([str(text)], convert_to_tensor=True)[0]
    return encoding.cpu().numpy()


#        tokenizer= AutoTokenizer.from_pretrained("t5-large")

# build and load model, it take time depending on your internet connection
#        model= AutoModelForSequenceClassification.from_pretrained("t5-large")
summarizer = pipeline("summarization", model="t5-large", tokenizer="t5-large")


def get_summary(text):
    summary = summarizer(str(text), min_length=5, max_length=128)[
        0]['summary_text']
    print(summary)
    return summary


def tweet_cleaner(tweet):
    #    import pdb;pdb.set_trace()
    tweet = re.sub(r"@\w*", " ", str(tweet)).strip()  # removing username
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', " ",
                   str(tweet)).strip()  # removing links
    tweet = re.sub(r'[^a-zA-Z]', " ", str(tweet)).strip()  # removing sp_char
#    tw = []
#
#    for text in tweet.split():
#        if text not in stopwords:
#            if not tw.startwith('@') and tw != 'RT':
#                tw.append(text)
#    print( " ".join(tw))
    tweet = re.sub(r"\s+", ' ', tweet)
    return tweet


REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#STOPWORDS = set(stopwords)


spell_checker = SpellChecker("en_US")
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


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the
    # matched string in REPLACE_BY_SPACE_RE with space.
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the
    # matched string in BAD_SYMBOLS_RE with nothing.
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and len(
        word) > 2)  # remove stopwors from text
    text = re.sub(r'\s+', ' ', text)

#    text += ' ' + extract_keywords(text)

#    correctwords = [w for w in  text.split() if spell_checker.check(w)]


#    entities = []
#    doc = nlp(text)
#    for ent in doc.ents:
#        if ent.text not in entities:
###            entities.append(' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> ')
# entities.append(ent.text)# + ' ' + ent.label_)
##            text = text.replace(ent.text, ' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> ')
#            text = text.replace(ent.text, ' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> ')
##
##
#    text = '[TEXT] ' + text + ' [ENTITIES] ' + ' '.join(entities)
#    text = ' '.join(correctwords)

    return text


def get_text(pair_id):
    path = os.path.join(data_path, pair_id[-2:], pair_id + ".json")

    article_text = ''
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        article_text = data['text']
    else:
        print(path)
        
    article_text = re.sub(r"\n+", ' ', article_text)
    article_text = re.sub(r"\s+", ' ', article_text)
    return article_text


kw_extractor = yake.KeywordExtractor(
    lan="en", n=1, top=300)  # , features=None)


def extract_keywords(text, to_lower=True):
    if to_lower:
        text = text.lower()
    keywords = dict(kw_extractor.extract_keywords(text))
    #processed_keywords = " ".join(keywords.keys())
    processed_keywords = {}
    for (keyword, score) in keywords.items():
        #score = 2-score
        score = -math.log(score)
        if score <= 0.0:
            continue
        processed_keywords[keyword] = score
    return ' '.join(processed_keywords.keys())


ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)


def get_multilingual_entitites(text):
    entities = []
    for sent in sent_tokenize(text):
#        print(sent, len(word_tokenize(sent)))
        vector = ner_model([sent])

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


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def add_entities(df):
    df['entities1'] = df['text1'].apply(lambda text: ' '.join(
        [' '.join(ents) for ents in get_multilingual_entitites(text)]))
    df['entities2'] = df['text2'].apply(lambda text: ' '.join(
        [' '.join(ents) for ents in get_multilingual_entitites(text)]))
    return df


if __name__ == '__main__':

    sentence_model = SentenceTransformer(model_name_or_path)

#    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#    model = AutoModel.from_pretrained(model_name_or_path)
#
#    doc_text = """Le cap du million de ventes a été dépassé l'an dernier. L'empressement des acheteurs à
#    devenir propriétaires, doublé d'une pénurie de biens à vendre, a entraîné de fortes hausses de prix dans
#    certaines villes. L'euphorie ne semblait pas devoir retomber... Mais la crise actuelle risque de changer
#    la donne.nnRecord battu ! En 2019, contre toute attente, le nombre de transactions a franchi pour la première
#    fois, en France, le cap du million, avec 1 059 000 ventes enregistrées sur tout le territoire. Cette excellente
#    conjoncture est surtout due aux taux de crédit historiquement bas, qui ont contribué à doper le pouvoir d'achat des acheteurs.
#    "En moyenne, les ménages ont pu emprunter à des taux compris entre 1,1 et 1,2% l'an dernier, soit moins que l'inflation ",
#    confirme Philippe Taboret, directeur associé du courtier Cafpi. Mais les particuliers ont aussi été nettement plus nombreux à
#    s'intéresser à l'investissement locatif. L'an passé près d'une transaction sur quatre a été réalisée dans cet objectif. La raison
#    : beaucoup de ménages voulaient garantir une stabilité à leur patrimoine et profiter d'une rentabilité moyenne de 2 à 5%.
#    Des taux largement plus intéressants que ceux offerts par les produits financiers à risque modéré.nnMais face à cette demande
#    en forte hausse, le nombre de vendeurs a continué de diminuer dans les zones les plus recherchées. A Paris, la pénurie qui
#    frappe le marché depuis plus de deux ans s'est accentuée. Le prix du mètre carré dans la capitale a donc flambé, avec une
#    hausse de 6,2% en moyenne l'an dernier selon les notaires. La barre psychologique des 10 000 €/m2 a même été dépassée,
#    sans que l'appétit des acheteurs s'émousse. Le phénomène d'attraction de la capitale a aussi été constaté dans le haut de gamme.
#    "L'an dernier, nous avons noté beaucoup plus de transactions entre 3 et 7 millions d'euros qu'en 2018", confie Alexander Kraft,
#    président-directeur général de Sotheby's Realty France-Monaco. Phénomène nouveau : la frénésie immobilière s'est largement diffusée
#    dans toutes les grandes métropoles de Province, avec les mêmes conséquences. "De mois en mois, le manque de biens à vendre s'accentue
#    dans tous les quartiers de l'hypercentre et leur proche couronne ", observe Philippe Descampiaux, directeur des agences Descampiaux-Dudicourt à Lille."""
#
#    input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
#    outputs = model.generate(
#        input_ids=input_ids,
#        max_length=64,
#        do_sample=True,
#        top_k=10,
#        num_return_sequences=3)

#    for i in range(3):
#        print(
#            f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')
    from ast import literal_eval

    test_df = pd.read_csv('../data/test_split_batch2_processed.csv', sep='\t', converters={"entities1": literal_eval, "entities2": literal_eval})

#    train_df = pd.read_csv(args.train_split)
#    test_df = pd.read_csv(args.test_split)
#
#    test_df['text1'] = test_df['pair_id'].apply(
#        lambda pair: get_text(pair.split("_")[0]))
#    test_df['text2'] = test_df['pair_id'].apply(
#        lambda pair: get_text(pair.split("_")[1]))
    
    print(test_df.head())
#
# test_df['text1_summary'] = test_df['text1'].apply(
# lambda text: get_summary(text))
# test_df['text2_summary'] = test_df['text2'].apply(
# lambda text: get_summary(text))
# print(test_df.head('../data/test_split_batch1_text_summary.csv'))
    
#    import pdb;pdb.set_trace()
##    test_df = parallelize_dataframe(test_df, add_entities)
#    test_df['entities1'] = test_df['text1'].progress_apply(lambda text : get_multilingual_entitites(text))
#    test_df['entities2'] = test_df['text2'].progress_apply(lambda text : get_multilingual_entitites(text))

# pd.DataFrame.to_csv(test_df)
#    test_df.to_csv('../data/test_split_batch2_processed.csv')

    true_similarity = np.array(test_df.Overall)

    pred_similarity = []
    idx = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        #        try:
        #            entities1 = get_multilingual_entitites(row.text1)
        #            entities2 = get_multilingual_entitites(row.text2)

#        import pdb;pdb.set_trace()
        
#        entities1 = ' '.join(['[' + x[1] + '] ' + x[0] for x in row.entities1])
#        entities2 = ' '.join(['[' + x[1] + '] ' + x[0] for x in row.entities2])

#        cosine_sim = distance.cosine(transform(sentence_model, entities1, add_entities=False),
#                                     transform(sentence_model, entities2, add_entities=False))
        
        cosine_sim_text = distance.cosine(transform(sentence_model, row.text1, add_entities=False),
                                     transform(sentence_model, row.text2, add_entities=False))
        
        
        cosine_sim = cosine_sim_text#(cosine_sim+cosine_sim_text)/2
#            cosine_sim = distance.cosine(transform(sentence_model, extract_keywords(row.text1), add_entities=False),
#                                         transform(sentence_model, extract_keywords(row.text2), add_entities=False))
#            cosine_sim_entities = distance.cosine(transform(sentence_model, row.text1, add_entities=True),
# transform(sentence_model, row.text2, add_entities=True))

#            pred_similarity.append((cosine_sim + cosine_sim_entities)/2.0)
        print(true_similarity[idx], '-', cosine_sim,
              '-', round((4 - 0) * cosine_sim))
        pred_similarity.append(cosine_sim)
        idx += 1
#            print(row.pair_id, cosine_sim)
#        except BaseException:
#            pred_similarity.append(0.5)
#            print('Issues', row.pair_id, len(row.text1), len(row.text2))

#    import pdb;pdb.set_trace()
#    pred_similarity = [round(0 + (4 - 0) * x) for x in pred_similarity]

#            pred_similarity = [lower + (upper - lower) * x for x in pred_similarity]

    print('pred_similarity:', pred_similarity)
    print('true_similarity:', true_similarity)
    pearson, p_val = evaluate_scores(pred_similarity, true_similarity)

    print(pearson, p_val)
