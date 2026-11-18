import numpy as np
import json
import os
from logging import getLogger
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
import torch

def load_data_and_model(model_file, data_path, device):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    if data_path:
        config['data_path'] = data_path
    if device:
        config['device'] = device
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

def load_data_and_model_wo_device(model_file, data_path):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    if data_path:
        config['data_path'] = data_path
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

def load_CF_SAS(dataset_name, gpu_id='0'):
    if dataset_name == 'Amazon_Arts':
        model_path = '/root/autodl-tmp/SRMR-Rec/saved/DMF-Amazon_Arts-Aug-07-2026_19-21-04.pth'
        SASRec_model_path = '/root/autodl-tmp/SRMR-Rec/saved/SASRec-Amazon_Arts-Aug-08-2026_16-54-15.pth'
    elif dataset_name == "Amazon_Games":
        model_path = '/root/autodl-tmp/SRMR-Rec/saved/DMF-Amazon_Games-Jul-30-2026_15-53-34.pth'
        SASRec_model_path = '/root/autodl-tmp/SRMR-Rec/saved/SASRec-Amazon_Games-Aug-05-2026_15-45-27.pth'
    elif dataset_name == "Amazon_Instruments":
        model_path = '/root/autodl-tmp/SRMR-Rec/saved/DMF-Aug-06-2026_10-34-17.pth'
        SASRec_model_path = '/root/autodl-tmp/SRMR-Rec/saved/SASRec-Amazon_Instruments-Aug-06-2026_19-42-50.pth'
    data_path = f'/root/autodl-tmp/SRMR-Rec/datasets/{dataset_name}'
    
    config, model, dataset, train_data, valid_data, test_data  = load_data_and_model(
        model_file=model_path,
        data_path=data_path,
        device = int(gpu_id)
    )
    SAS_config, SAS_model, SAS_dataset, SAS_train_data, SAS_valid_data, SAS_test_data  = load_data_and_model(
        model_file=SASRec_model_path,
        data_path=data_path,
        device = int(gpu_id)
    )
    return model, dataset, SAS_model, SAS_dataset

def get_user_emb(user_id=None, CF_model=None, CF_dataset=None):
    user_emb = CF_model.get_user_embedding(user=torch.tensor([int(user_id)])).to('cpu').detach().numpy().reshape(-1, 1)
    return user_emb

def concat_item_list_emb(item_list=None, SAS_model=None, SAS_dataset=None):
    item_emb_size = SAS_model.item_embedding.weight[0].to('cpu').detach().numpy().shape[0]
    item_list_emb = np.zeros((item_emb_size, 1))
    for item in item_list:
        if item not in SAS_dataset.field2token_id['item_id'].keys():
            continue
        item_emb = SAS_model.item_embedding.weight[SAS_dataset.field2token_id['item_id'][item]].to('cpu').detach().numpy().reshape(-1, 1)
        item_list_emb = item_list_emb + item_emb
    # get average of item_list_emb
    item_list_emb /= len(item_list)
    return item_list_emb

def generate_candidate_emb(user_id=None, item_list=None, CF_model=None, CF_dataset=None, SAS_model=None, SAS_dataset=None):
    user_emb = get_user_emb(user_id, CF_model, CF_dataset)
    item_list_emb = concat_item_list_emb(item_list, SAS_model, SAS_dataset)
    final_candidate_emb = np.concatenate((user_emb, item_list_emb), axis=0)
    return final_candidate_emb.reshape(-1)