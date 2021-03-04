# Cross-Attention Based BERT4Rec


## Usage

**Requirements**

* python 3.6+
* Tensorflow 1.15 (GPU version)
* CUDA compatible with TF 1.15

**Run**

For simplicity, here we take ml-1m as an example:

``` bash
./run_ml-1m.sh
```
include two part command:
generated masked training data
``` bash
python -u gen_data_fin.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \
    --pool_size=${pool_size} \
```

train the model
``` bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4
```

### hyper-parameter settings
json in `bert_train` like `bert_config_ml-1m_64.json`

```json
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}
```


## Reference

```TeX
@inproceedings{Sun:2019:BSR:3357384.3357895,
 author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
 title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 isbn = {978-1-4503-6976-3},
 location = {Beijing, China},
 pages = {1441--1450},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3357384.3357895},
 doi = {10.1145/3357384.3357895},
 acmid = {3357895},
 publisher = {ACM},
 address = {New York, NY, USA}
} 
```
```@InProceedings{Yu_2020_CVPR,
author = {Yu, Yuechen and Xiong, Yilei and Huang, Weilin and Scott, Matthew R.},
title = {Deformable Siamese Attention Networks for Visual Object Tracking},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

```
ndcg@1:0.27334437086092717, hit@1:0.27334437086092717， ndcg@5:0.4350090955373317, hit@5:0.5786423841059603, ndcg@10:0.4701029856279757, hit@10:0.6867549668874172, ap:0.41375562287737444, valid_user:6040.0

INFO:tensorflow:Finished evaluation at 2021-02-07-00:23:19
I0207 00:23:19.007557 140642322712384 evaluation.py:275] Finished evaluation at 2021-02-07-00:23:19
INFO:tensorflow:Saving dict for global step 400000: global_step = 400000, loss = 6.100204, masked_lm_accuracy = 0.02615894, masked_lm_loss = 6.0999246
I0207 00:23:19.007871 140642322712384 estimator.py:2049] Saving dict for global step 400000: global_step = 400000, loss = 6.100204, masked_lm_accuracy = 0.02615894, masked_lm_loss = 6.0999246
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 400000: ./BERT4Rec-MultiSeq/ml-1m-mp1.0-sw0.5-mlp0.2-df10-mpps40-msl200-64/model.ckpt-400000
I0207 00:23:19.275999 140642322712384 estimator.py:2109] Saving 'checkpoint_path' summary for global step 400000: ./BERT4Rec-MultiSeq/ml-1m-mp1.0-sw0.5-mlp0.2-df10-mpps40-msl200-64/model.ckpt-400000
INFO:tensorflow:***** Eval results *****
I0207 00:23:19.293869 140642322712384 run.py:588] ***** Eval results *****
INFO:tensorflow:{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}

I0207 00:23:19.294348 140642322712384 run.py:589] {
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}
INFO:tensorflow:  global_step = 400000
```

```
ndcg@1:0.26903973509933776, hit@1:0.26903973509933776， ndcg@5:0.42775598147095745, hit@5:0.5695364238410596, ndcg@10:0.46340664581866337, hit@10:0.6791390728476822, ap:0.4077645664291678, valid_user:6040.0
INFO:tensorflow:Finished evaluation at 2021-02-24-10:05:13
I0224 10:05:13.598598 140640140453696 evaluation.py:275] Finished evaluation at 2021-02-24-10:05:13
INFO:tensorflow:Saving dict for global step 400000: global_step = 400000, loss = 6.0340858, masked_lm_accuracy = 0.02615894, masked_lm_loss = 6.034216
I0224 10:05:13.598937 140640140453696 estimator.py:2049] Saving dict for global step 400000: global_step = 400000, loss = 6.0340858, masked_lm_accuracy = 0.02615894, masked_lm_loss = 6.034216
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 400000: ../BERT-MultiSeq-Rec_Exp/ml-1m-mp1.0-sw0.5-mlp0.2-df10-mpps40-msl200-64/model.ckpt-400000
I0224 10:05:13.711844 140640140453696 estimator.py:2109] Saving 'checkpoint_path' summary for global step 400000: ../BERT-MultiSeq-Rec_Exp/ml-1m-mp1.0-sw0.5-mlp0.2-df10-mpps40-msl200-64/model.ckpt-400000
INFO:tensorflow:***** Eval results *****
I0224 10:05:13.728864 140640140453696 run.py:588] ***** Eval results *****
INFO:tensorflow:{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}

I0224 10:05:13.729411 140640140453696 run.py:589] {
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}

INFO:tensorflow:  global_step = 400000
I0224 10:05:13.729727 140640140453696 run.py:592]   global_step = 400000
INFO:tensorflow:  loss = 6.0340858
I0224 10:05:13.729836 140640140453696 run.py:592]   loss = 6.0340858
INFO:tensorflow:  masked_lm_accuracy = 0.02615894
I0224 10:05:13.729906 140640140453696 run.py:592]   masked_lm_accuracy = 0.02615894
INFO:tensorflow:  masked_lm_loss = 6.034216
I0224 10:05:13.729966 140640140453696 run.py:592]   masked_lm_loss = 6.034216
```