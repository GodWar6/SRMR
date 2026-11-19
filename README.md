# SRMR: Selective Reflection and Multi-Source Re-ranking for LLM-Based Sequential Recommendation

Implementation of our method SRMR for LLM-based sequential recommendation, featuring selective reflection gating and multi-source candidate re-ranking.

## Overview

SRMR builds upon large language models for sequential recommendation by introducing a risk-aware reflection gating mechanism and a PPO-based reflection selector. Instead of applying reflection uniformly, SRMR selectively triggers reflection based on user history consistency, and fuses base recommendations with reflection outputs via a joint reranking scheme incorporating LCS similarity, TF-IDF semantic similarity, and rank prior.

## Prerequisites

See ``./environment.yml`` for details.

## Dataset Prepare

### Download

Amazon 2018 5-core (Arts / Games / Instruments), already filtered and converted to RecBole format:

> **[Download from GitHub Release — data-v1.0](https://github.com/GodWar6/SRMR/releases/tag/data-v1.0)**
>
> Download `SRMR-datasets.zip` and extract it at the project root, so that
> `datasets/Amazon_Arts/`, `datasets/Amazon_Games/` and
> `datasets/Amazon_Instruments/` sit next to `refl_refining_iteration.py`.

To rebuild them from the raw 5-core files instead:

> [Datasets download](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
>
> Search for Amazon, and download the 2018 version which includes corresponding .item, .inter files; ensure all necessary files are included.

### prepare_cluster_user/prepare_data.py

This .py file is used to construct the CF model (DMF). To modify training configurations, adjust the settings in the `./prepare_cluster_user/DMF config/` for different datasets.

### prepare_cluster_user/sample_users.py

It samples the evaluation user subset (`users_<N>.json`, 1000 users by default). The subset is drawn from the same 5-core filtered interactions that the DMF backbone is trained on, so the sampled internal user ids match the CF user embedding table.

### prepare_cluster_user/pre_cu_sampled.py

It is used for constructing user clusters, with specifics aligning with the paper's internal details.

### prepare_cluster_user/get_samples.ipynb

It is used for constructing training, validation, and test samples, allowing lookup of corresponding user/item embeddings.

## Training & Evaluation

### ex_refl_refining_iteration.sh

This is the script for running offline reflection resource construction via the Generator-Critic-Reviser three-stage iterative process. You can refer to the corresponding parameters within and run according to your needs.


### ex_refl_selection_training.sh

This script is for training the PPO-based reflection selector. You can reference the relevant parameters from here and execute as required.

### ex_refl_re_recommend.sh

This is the script to run online inference: base recommendation → PPO arm selection → candidate fusion and deduplication → joint reranking, followed by evaluation on the test set. Relevant parameters can be referenced and executed according to your needs. To run on other datasets, change `dataset_name` (and the corresponding reflection pool / PPO checkpoint paths) in the script, e.g. to `Amazon_Games`.

## Others

### ./prompt/

This is the folder containing two types of prompts: recommendation prompts and reflection prompts.

### Meta-Llama-3-8B-Instruct-bf16

If you haven't downloaded the LLaMa-3 model, please download the LLaMa-3-8B model from huggingface at this link:
[LLaMa-3-8B](https://huggingface.co/gradientai/Meta-Llama-3-8B-bf16)

The model is loaded locally via vLLM; configure the path through the `--model` parameter in `refl_re_recommend.py` and `refl_refining_iteration.py`.
