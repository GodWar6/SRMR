import os
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation


BASE_SAVE_DIR = '/root/autodl-tmp/SRMR-Rec/saved'
os.makedirs(BASE_SAVE_DIR, exist_ok=True)                    
os.makedirs(os.path.join(BASE_SAVE_DIR, 'logs'), exist_ok=True) 


# get CF model, current setting still needs completion. BASELINE: DMF
dataset = ['Amazon_Arts', 'Amazon_Games', 'Amazon_Instruments']
model_list = ['DMF']
filtered_dataset = []
for data in dataset:
    for model in model_list:
        curr_model_conf_path = "/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/DMF config/" + data + ".yaml"
        config = Config(model=model, dataset=data, config_file_list=[curr_model_conf_path])
        
        curr_dataset = create_dataset(config)
        curr_dataset.save()
        print("Dataset Saved OK.")
        print("Start Training.")
        run_recbole(model=model, dataset=data, config_file_list=[curr_model_conf_path])
        print("current model is: ", model)
        print("current dataset is: ", data)
        print("Finished running.")
        print("--------------------------------------")
print("All finished.")
