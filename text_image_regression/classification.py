# -*- coding: utf-8 -*-
from torch import nn
import os.path
from ast import literal_eval
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import random
import os
from tqdm import tqdm
import argparse
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModel, get_linear_schedule_with_warmup)
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)
manualSeed = 2022


np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
LEARNING_RATE = 1e-05

IMAGE_MODEL = 'google/vit-large-patch32-224-in21k'
IMAGE_MODEL = 'google/vit-base-patch16-224'
# IMAGE_MODEL = 'openai/clip-vit-base-patch32'
# IMAGE_MODEL = 'openai/clip-vit-large-patch14'
feature_extractor = ViTFeatureExtractor.from_pretrained(IMAGE_MODEL)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_file',
    type=str,
    default="../data/train_split_batch2_entities.csv",
    help='Train file (train.csv)')
parser.add_argument(
    '--test_file',
    type=str,
    default="../data/test_split_batch2_entities.csv",
    help='Test file (test.csv)')
parser.add_argument(
    '--dev_file',
    type=str,
    default="../data/test_split_batch2_entities.csv",
    help='Test file (test.csv)')
parser.add_argument(
    '--out',
    type=str,
    default='../data/runs',
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

train_df = pd.read_csv(
    train_file,
    sep=',',
    converters={
        "entities1": literal_eval,
        "entities2": literal_eval,
        "meta_keywords1": literal_eval,
        "meta_keywords2": literal_eval,
        "keywords1": literal_eval,
        "keywords2": literal_eval,
        "tags1": literal_eval,
        "tags2": literal_eval})

test_df = pd.read_csv(
    test_file,
    sep=',',
    converters={
        "entities1": literal_eval,
        "entities2": literal_eval,
        "meta_keywords1": literal_eval,
        "meta_keywords2": literal_eval,
        "keywords1": literal_eval,
        "keywords2": literal_eval,
        "tags1": literal_eval,
        "tags2": literal_eval})

dev_df = pd.read_csv(
    args.dev_file,
    sep=',',
    converters={
        "entities1": literal_eval,
        "entities2": literal_eval,
        "meta_keywords1": literal_eval,
        "meta_keywords2": literal_eval,
        "keywords1": literal_eval,
        "keywords2": literal_eval,
        "tags1": literal_eval,
        "tags2": literal_eval})

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


LABELS = [
    'Geography',
    'Entities',
    'Time',
    'Narrative',
    'Overall',
    'Style',
    'Tone']

# test_df['Overall'] = list(true_results['Overall'])

print(train_df['Overall'].unique())
print(dev_df['Overall'].unique())


# train_ids = pd.read_csv('../data/KB/train_i2idx.csv')
# train_ids['id'] = train_ids['id'].astype('Int64')
# test_ids = pd.read_csv('../data/KB/eval_i2idx.csv')
# test_ids['id'] = test_ids['id'].astype('Int64')
# dev_ids = pd.read_csv('../data/KB/test_i2idx.csv')
# dev_ids['id'] = dev_ids['id'].astype('Int64')

# files_KB = []
# for path, directories, files in os.walk('../data/KB/'):
#     for file in files:
#         print('found %s' % os.path.join(path, file))
#         files_KB.append(os.path.join(path, file))

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe

        self.comment_text1 = self.data.text1
        self.comment_text2 = self.data.text2
        self.entities1 = self.data.entities1
        self.entities2 = self.data.entities2
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
        return len(self.comment_text1 + self.comment_text2)

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
            # print(id_image, 'not_found')
            return '../data/batch2_images/img_000000'

    def __getitem__(self, index):

        id1 = self.pair_id[index].split("_")[0]
        id2 = self.pair_id[index].split("_")[1]

        path1 = self._search_image(id1)
        path2 = self._search_image(id2)

#        google1 = str(self.google1[index]) #str(self.comment_text1[index])
#        google1 = " ".join(google1.split()) #+ #' ' + entities1
#        bart1 = str(self.bart1[index]) #str(self.comment_text1[index])
#        bart1 = " ".join(bart1.split()) #+ #' ' + entities1

        entities1 = ' '.join(
            [ent_type + ' ' + ent_text for ent_type, ent_text in self.entities1[index]])
        # str(self.comment_text1[index])
        comment_text1 = str(self.comment_text1[index])
        comment_text1 = str(self.title1[index]) + ' ' + " ".join(comment_text1.split()) + ' ' +  \
            str(' '.join(self.meta_keywords1[index])) + ' ' + \
            str(self.tags1[index]) + ' ' + str(self.keywords1[index]) + ' ' + str(self.topic1[index])

        try:
            top_image1 = Image.open(path1)
        except BaseException:
            # print('--', path1)
            top_image1 = Image.open('../data/batch2_images/img_000000')

        try:
            top_image2 = Image.open(path2)
        except BaseException:
            top_image2 = Image.open('../data/batch2_images/img_000000')

        try:
            top_image1 = feature_extractor(images=top_image1, return_tensors="pt")[
                'pixel_values'].squeeze()
        except BaseException:
            top_image1 = torch.zeros((3, 224, 224), dtype=torch.float)

        try:
            top_image2 = feature_extractor(images=top_image2, return_tensors="pt")[
                'pixel_values'].squeeze()
        except BaseException:
            # feature_extractor(images=top_image2, return_tensors="pt")['pixel_values'].squeeze()
            top_image2 = torch.zeros((3, 224, 224), dtype=torch.float)

        # top_image1 = torch.zeros((3, 224, 224), dtype=torch.float)
        # top_image2 = torch.zeros((3, 224, 224), dtype=torch.float)
#        google2 = str(self.google2[index]) #str(self.comment_text1[index])
#        google2 = " ".join(google2.split()) #+ #' ' + entities1
#        bart2 = str(self.bart2[index]) #str(self.comment_text1[index])
#        bart2 = " ".join(bart2.split()) #+ #' ' + entities1

        entities2 = ' '.join(
            [ent_type + ' ' + ent_text for ent_type, ent_text in self.entities2[index]])
        # str(self.comment_text2[index])
        comment_text2 = str(self.comment_text2[index])
        comment_text2 = str(self.title2[index]) + ' ' + " ".join(comment_text2.split()) + ' ' + \
            str(' '.join(self.meta_keywords2[index])) + ' ' + \
            str(self.tags2[index]) + ' ' + str(self.keywords2[index]) + ' ' + str(self.topic2[index])
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

        return_result = {
            'ids': torch.tensor(ids, dtype=torch.long),
            #            'image1': torch.tensor(image1, dtype=torch.float),
            #            'image2': torch.tensor(image2, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'top_image1': top_image1,
            'top_image2': top_image2
        }

        for LABEL in LABELS:
            return_result[LABEL] = torch.tensor(
                self.targets[LABEL][index], dtype=torch.float)

        return return_result

    def get_tokenizer(self):
        return self.tokenizer

# Creating the dataset and dataloader for the neural network


print("TRAIN Dataset: {}".format(train_df.shape))
print("TEST Dataset: {}".format(test_df.shape))


def loss_fn(outputs, targets):
    return torch.nn.MSELoss()(outputs, targets.view(-1, 1))


class BERTClass(torch.nn.Module):

    config_class = AutoConfig

    def __init__(self, config, tokenizer):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        self.bert = AutoModel.from_pretrained(args.model_name_or_path,
                                              config=config)

        self.bert.resize_token_embeddings(len(tokenizer))

        self.vit = ViTModel.from_pretrained(IMAGE_MODEL)

        self.dropout = torch.nn.Dropout(0.3)
        self.drop = nn.Dropout(0.3)

        self.linear_types = nn.ModuleList()
        for LABEL in LABELS:
            # self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size*2, 1))
            # self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size*(MAX_LEN+1), 1))
            self.linear_types.append(torch.nn.Linear(
                self.bert.config.hidden_size * (MAX_LEN + 3), 1))
            # self.linear_types.append(torch.nn.Linear(self.bert.config.hidden_size*(MAX_LEN+2), 1))

        self.image_projection1 = torch.nn.Linear(512 * 2, 768)
        self.image_projection2 = torch.nn.Linear(512 * 2, 768)

#        self._init_weights(self)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            ids,
            mask,
            image1,
            image2,
            top_image1,
            top_image2,
            token_type_ids):

        element = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)

        doc, bert_outputs = element[0], element[1]

        output = self.dropout(bert_outputs)

        top_image1 = self.vit(top_image1)
        top_image2 = self.vit(top_image2)

        image_embeddings1 = top_image1['pooler_output']
        image_embeddings2 = top_image2['pooler_output']

        # output = torch.cat([image_embeddings1.unsqueeze(1), image_embeddings2.unsqueeze(1)], 1).view(output.shape[0], -1)
        output = torch.cat([output.unsqueeze(1), doc, image_embeddings1.unsqueeze(
            1), image_embeddings2.unsqueeze(1)], 1).view(output.shape[0], -1)
        # output = torch.cat([output.unsqueeze(1), doc], 1).view(output.shape[0], -1)

        output_types = []
        for idx, LABEL in enumerate(LABELS):
            output_types.append(self.linear_types[idx](output))

        return output_types


def evaluation(model, epoch, loader, mode):
    model.eval()
    print('MODE:', mode)
    fin_targets_types = {}
    fin_output_types = {}

    with torch.no_grad():
        for _, data in tqdm(enumerate(loader), total=len(loader)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            image1 = data['top_image1'].to(device, dtype=torch.float)
            image2 = data['top_image2'].to(device, dtype=torch.float)
            top_image1 = data['top_image1'].to(device, dtype=torch.float)
            top_image2 = data['top_image2'].to(device, dtype=torch.float)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)

            for LABEL in LABELS:
                if LABEL not in fin_targets_types:
                    fin_targets_types[LABEL] = []
                fin_targets_types[LABEL] += data[LABEL].cpu().detach().numpy().tolist()

            output_types = model(
                ids,
                mask,
                image1,
                image2,
                top_image1,
                top_image2,
                token_type_ids)

            for idx, LABEL in enumerate(LABELS):
                if LABEL not in fin_output_types:
                    fin_output_types[LABEL] = []
                fin_output_types[LABEL] += output_types[idx].reshape(-1).cpu(
                ).detach().numpy().tolist()

    from scipy.stats import pearsonr

    pred_scores = np.array([float(x) for x in fin_output_types['Overall']])
    true_scores = np.array(fin_targets_types['Overall'])
    print(pred_scores[:10])
    print(true_scores[:10])
    pearsonr_corr = pearsonr(pred_scores, true_scores)
    print("Pearson-r:", pearsonr_corr[0])
    print("p-value:", pearsonr_corr[1])

    if 'test' in mode:
        test_df['PredictedOverall'] = pred_scores
        test_df.to_csv(
            os.path.join(
                output_dir,
                mode +
                '_predictions_' +
                str(epoch) +
                '.csv'))
    else:
        train_df['PredictedOverall'] = pred_scores
        train_df.to_csv(
            os.path.join(
                output_dir,
                mode +
                '_predictions_' +
                str(epoch) +
                '.csv'))

    return fin_output_types, fin_targets_types


def train(model, optimizer, scheduler, epoch):
    model.train()

    for step, data in tqdm(enumerate(training_loader),
                           total=len(training_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        image1 = data['top_image1'].to(device, dtype=torch.float)
        image2 = data['top_image2'].to(device, dtype=torch.float)
        top_image1 = data['top_image1'].to(device, dtype=torch.float)
        top_image2 = data['top_image2'].to(device, dtype=torch.float)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        output_types = model(
            ids,
            mask,
            image1,
            image2,
            top_image1,
            top_image2,
            token_type_ids)

        losses = []
        targets = {}
        for idx, LABEL in enumerate(LABELS):
            if LABEL not in targets:
                targets[LABEL] = []
            targets[LABEL] = data[LABEL].to(device, dtype=torch.float)
            losses.append(loss_fn(output_types[idx], targets[LABEL]))

        optimizer.zero_grad()
        loss = sum(losses)

        if step % args.save_steps == 0:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
#            model_to_save.save_pretrained(output_dir)

            torch.save(
                model_to_save.state_dict(),
                os.path.join(
                    output_dir,
                    'model.pth'))

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        truncation=True)

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

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = BERTClass(config, tokenizer)

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

    t_total = len(
        training_loader) // args.gradient_accumulation_steps * args.num_train_epochs
#
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(
            args.adam_beta1,
            args.adam_beta2))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    for epoch in range(int(args.num_train_epochs)):
        _ = train(model, optimizer, scheduler, epoch)
        _, _ = evaluation(model, epoch, testing_loader, 'test')
        # _, _ = evaluation(model, epoch, training_loader_test, 'train')

if args.do_predict:

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
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
