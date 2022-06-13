# EMBEDDIA at SemEval 2022 Task 8: Multilingual News Similarity

This is the codebase for our experiments for Task 8: Multilingual news similarity

## Abstract
We cover several techniques and propose different methods for finding the multilingual news article similarity by exploring the dataset in its entirety. We take advantage of the textual content of the articles, the provided **metadata** (e.g., titles, keywords, topics), the **translated articles**, the **images** (those that were available), and **knowledge graph-based representations** for entities and relations present in the articles. We, then, compute the semantic similarity between the different features and predict through regression the similarity scores. 

Our findings show that, while our researched methods obtained promising results, exploiting the **semantic textual similarity** with sentence representations is unbeatable. Finally, in the official SemEval 2022 Task 8, we ranked **fifth in the overall team ranking** cross-lingual results, and **second in the English-only results**.

### Link to paper: TBD (To be presented as SemEval 2022 workshop on 14 July 2022, co-located with NAACL)


# Regression (Text+Image)

```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python classification.py  
--out runs/semeval_512_xlm_large_image --model_name_or_path xlm-roberta-large 
--train_file ../data/train_split_batch2.csv  --test_file ../data/test_split_batch2.csv
--do_train  --batch_size 16 --num_train_epochs 2 --learning_rate 1e-5 --do_lower_case
```


