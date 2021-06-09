# MetaAdaptRank

This repository provides the implementation of meta-learning to reweight synthetic weak supervision data described in the paper [**Meta Adaptive Neural Ranking with Contrastive Synthetic Supervision**](https://arxiv.org/pdf/2012.14862.pdf).

- [1. Contrastive Supervision Synthesis (CTSyncSup)](#1-contrastive-supervision-synthesis)
- [2. Meta Learning to Reweight CTSyncSup](#2-meta-learning-to-reweight)

## CONTACT

For any question, please contact **Si Sun** by email s-sun17@mails.tsinghua.edu.cn (respond to emails more quickly), we will try our best to solve ：）


## QUICKSTART

```
python 3.7
Pytorch 1.5.0
```

### 0/ Data Preparation

First download and prepare the following data into the `data` folder:

* Download source-domain NLG training datasets and prepare them into the `data/source_data` folder.

  - MS MARCO Passage Ranking dataset [[download]](https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz)

  [>> example data format](./data/source_data/toy_triples.train.small.tsv)


* Download target-domain inference/evaluation datasets and prepare them into the `data/target_data` folder.
  - clueweb09
  - robust04
  - trec-covid

  [>> example data format](./data/target_data)


* Download the pre-trained language models and prepare them into the `data/pretrain_model` folder.
  - [t5-small](https://huggingface.co/t5-small) & [t5-base](https://huggingface.co/t5-base) (for Weak Supervision Synthesis)
  - [bert-base-uncased](https://huggingface.co/bert-base-uncased) (for ClueWeb09 and Robust04)
  - [BiomedNLP-PubMedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) (for TREC-COVID)


## 1 Contrastive Supervision Synthesis

### 1.1 Source-domain NLG training

- We train two query generators (QG & ContrastQG) with the MS MARCO dataset using `train_nlg.sh` in the `run_shells` folder:

  ```
  bash prepro_nlg_dataset.sh
  ```

- Optional arguments:

  ```
  --generator_mode            choices=['qg', 'contrastqg']
  --pretrain_generator_type   choices=['t5-small', 't5-base']
  --train_file                The path to the source-domain nlg training dataset
  --save_dir                  The path to save the checkpoints data; default: ../results
  ```

### 1.2 Target-domain NLG inference

- The whole nlg inference pipline contains five steps:
  - 1.2.1/ Data preprocess
  - 1.2.2/ Seed query generation
  - 1.2.3/ BM25 subset retrieval
  - 1.2.4/ Contrastive doc pairs sampling
  - 1.2.5/ Contrastive query generation

- 1.2.1/ **Data preprocess.** convert target-domain documents into the nlg format using `prepro_nlg_dataset.sh` in the `preprocess` folder:

  ```
  bash prepro_nlg_dataset.sh
  ```

- Optional arguments:

  ```
  --dataset_name          choices=['clueweb09', 'robust04', 'trec-covid']
  --input_path            The path to the target dataset
  --output_path           The path to save the preprocess data; default: ../data/prepro_target_data
  ```

- 1.2.2/ **Seed query generation.** utilize the trained QG model to generate seed queries for each target documents using `nlg_inference.sh` in the `run_shells` folder:

  ```
  bash nlg_inference.sh
  ```

- Optional arguments:
  ```
  --generator_mode            choices='qg'
  --pretrain_generator_type   choices=['t5-small', 't5-base']
  --target_dataset_name       choices=['clueweb09', 'robust04', 'trec-covid']
  --generator_load_dir        The path to the pretrained QG checkpoints.
  ```

- 1.2.3/ **BM25 subset retrieval.** utilize BM25 to retrieve document subset according to the seed queries using `do_subset_retrieve.sh` in the `bm25_retriever` folder:

  ```
  bash do_subset_retrieve.sh
  ```

- Optional arguments:
  ```
  --dataset_name          choices=['clueweb09', 'robust04', 'trec-covid']
  --generator_folder      choices=['t5-small', 't5-base']
  ```

- 1.2.4/ **Contrastive doc pairs sampling.** pairwise sample contrastive doc pairs from the BM25 retrieved subset using `sample_contrast_pairs.sh` in the `preprocess` folder:

  ```
  bash sample_contrast_pairs.sh
  ```

- Optional arguments:

  ```
  --dataset_name          choices=['clueweb09', 'robust04', 'trec-covid']
  --generator_folder      choices=['t5-small', 't5-base']
  ```

- 1.2.5/ **Contrastive query generation.** utilize the trained ContrastQG model to generate new queries based on contrastive document pairs using `nlg_inference.sh` in the `run_shells` folder:

  ```
  bash nlg_inference.sh
  ```

- Optional arguments:
  ```
  --generator_mode            choices='contrastqg'
  --pretrain_generator_type   choices=['t5-small', 't5-base']
  --target_dataset_name       choices=['clueweb09', 'robust04', 'trec-covid']
  --generator_load_dir        The path to the pretrained ContrastQG checkpoints.
  ```


## 2 Meta Learning to Reweight


### 2.1 Data Preprocess

- Prepare the contrastive synthetic supervision data (CTSyncSup) into the `data/synthetic_data` folder.
  - CTSyncSup_clueweb09
  - CTSyncSup_robust04
  - CTSyncSup_trec-covid

  [>> example data format](./data/synthetic_data/CTSyncSup_trec-covid)

- Preprocess the target-domain datasets into the 5-fold cross-validation format using `run_cv_preprocess.sh` in the `preprocess` folder:

  ```
  bash run_cv_preprocess.sh
  ```

- Optional arguments:

  ```
  --dataset_class         choices=['clueweb09', 'robust04', 'trec-covid']
  --input_path            The path to the target dataset
  --output_path           The path to save the preprocess data; default: ../data/prepro_target_data
  ```


### 2.2 Train and Test Models

- The whole process of training and testing MetaAdaptRank contains three steps:

  - 2.2.1/ **Meta-pretraining.** The model is trained on synthetic weak supervision data, where the synthetic data are reweighted using meta-learning. The training fold of the target dataset is considered as target data that guides meta-reweighting.

  - 2.2.2/ **Fine-tuning.** The meta-pretrained model is continuously fine-tuned on the training folds of the target dataset.

  - 2.2.3/ **Ensemble and Coor-Ascent.** Coordinate Ascent is used to combine the last representation layers of all fine-tuned models, as LeToR features, with the retrieval scores from the base retriever.


- 2.2.1/ **Meta-pretraining** using `train_meta_bert.sh` in the `run_shells` folder:
  ```
  bash train_meta_bert.sh
  ```
  Optional arguments for meta-pretraining:

  ```
  --cv_number             choices=[0, 1, 2, 3, 4]
  --pretrain_model_type   choices=['bert-base-cased', 'BiomedNLP-PubMedBERT-base-uncased-abstract']
  --train_dir             The path to the synthetic weak supervision data
  --target_dir            The path to the target dataset
  --save_dir              The path to save the output files and checkpoints; default: ../results
  ```
  Complete optional arguments can be seen in `config.py` in the `scripts` folder.

- 2.2.2/ **Fine-tuning** using `train_metafine_bert.sh` in the `run_shells` folder:
  ```
  bash train_metafine_bert.sh
  ```
  Optional arguments for fine-tuning:

  ```
  --cv_number             choices=[0, 1, 2, 3, 4]
  --pretrain_model_type   choices=['bert-base-cased', 'BiomedNLP-PubMedBERT-base-uncased-abstract']
  --train_dir             The path to the target dataset
  --checkpoint_folder     The path to the checkpoint of the meta-pretrained model
  --save_dir              The path to save output files and checkpoint; default: ../results
  ```

- 2.2.3/ **Testing** the fine-tuned model to collect LeToR features through `test.sh` in the `run_shells` folder:
  ```
  bash test.sh
  ```
  Optional arguments for testing:

  ```
  --cv_number             choices=[0, 1, 2, 3, 4]
  --pretrain_model_type   choices=['bert-base-cased', 'BiomedNLP-PubMedBERT-base-uncased-abstract']
  --target_dir            The path to the target evaluation dataset
  --checkpoint_folder     The path to the checkpoint of the fine-tuned model
  --save_dir              The path to save output files and the **features** file; default: ../results
  ```

- 2.2.4/ **Ensemble.** Train and test five models for each fold of the target dataset (5-fold cross-validation), and then ensemble and convert their output features to coor-ascent format using `combine_features.sh` in the `ensemble` folder:
  ```
  bash combine_features.sh
  ```
  Optional arguments for ensemble:
  ```
  --qrel_path             The path to the qrels of the target dataset
  --result_fold_1         The path to the testing result folder of the first fold model
  --result_fold_2         The path to the testing result folder of the second fold model
  --result_fold_3         The path to the testing result folder of the third fold model
  --result_fold_4         The path to the testing result folder of the fourth fold model
  --result_fold_5         The path to the testing result folder of the fifth fold model
  --save_dir              The path to save the ensembled `features.txt` file; default: ../combined_features
  ```

- 2.2.5/ **Coor-Ascent.** Run coordinate ascent using `run_ranklib.sh` in the `ensemble` folder:
  ```
  bash run_ranklib.sh
  ```
  Optional arguments for coor-ascent:
  ```
  --qrel_path             The path to the qrels of the target dataset
  --ranklib_path          The path to the ensembled features.
  ```
  The final evaluation results will be output in the `ranklib_path`.


## Results

All TREC files listed in this paper can be found in [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d02c9fee036448a2bc4d/).
