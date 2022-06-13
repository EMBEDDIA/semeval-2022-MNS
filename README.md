# EMBEDDIA at SemEval 2022 Task 8

This is the codebase for our experiments for Task 8: Multilingual news similarity

Paper: TBD (To be presented as SemEval 2022 workshop on 14 July 2022, co-located with NAACL)


# Regression (Text+Image)

```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python classification.py  
--out runs/semeval_512_xlm_large_image --model_name_or_path xlm-roberta-large 
--train_file ../data/train_split_batch2.csv  --test_file ../data/test_split_batch2.csv
--do_train  --batch_size 16 --num_train_epochs 2 --learning_rate 1e-5 --do_lower_case
```


