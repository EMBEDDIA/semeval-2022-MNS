# semeval-2022-MNS


# Regression (Text+Image)

```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python classification.py  --out runs/semeval_512_xlm_large_image --model_name_or_path xlm-roberta-large --train_file ../data/train_split_batch2.csv  --test_file ../data/test_split_batch2.csv  --do_train  --batch_size 16 --num_train_epochs 2 --learning_rate 1e-5 --do_lower_case
```


