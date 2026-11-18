 # %%
import transformers
import torch
from prompt.reflect_prompt import *
from pprint import pprint
import json
import os
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken
from tqdm.contrib import tzip
import pandas as pd
from typing import List
import argparse
import random
import numpy as np
import pdb
from typing import List,Union
from recbole.quick_start import run_recbole
from utils.data import load_data_and_model
from recbole.data.interaction import Interaction
import setproctitle
import datetime
import html
from vllm import LLM, SamplingParams

random.seed(2024)
np.random.seed(2024)


parser = argparse.ArgumentParser(description=".")


parser.add_argument('-bs', '--batch_size', type=int, default=12, help='Batch size for rec')
parser.add_argument('-bs2', '--batch_size2', type=int, default=0, help='Batch size for fs')
parser.add_argument('-dn', '--dataset_name', type=str, default='Amazon_Games', help='Dataset name') # key param
parser.add_argument('-mil', '--max_inter_len', type=int, default=20, help='')
parser.add_argument('-ged', '--gold_examples_dir', type=str, default='', help='')
parser.add_argument('-mi', '--max_iter', type=int, default=2, help='')
parser.add_argument('-ex', type=str, default='TRY', help='name of experiment this time')
parser.add_argument('-od', '--output_dir', type=str, default='iter_pool/{dataset_name}_{ex}', help='')
parser.add_argument('-fst', '--fs_temperature', type=float, default=1.0, help='')
parser.add_argument('-fsds', '--fs_do_sample', action='store_true', help='')
parser.add_argument('-rt', '--refl_type', type=str, default='IP', choices=['EP', 'IP', 'CF'], help='')
parser.add_argument('-fm', '--fusion_mode', type=str, default='max', choices=['max','new'], help='')
parser.add_argument('-ofdd', '--original_fs_dict_dir', type=str, 
                    default='tmp/{dataset_name}/sample_prompt_4_greedy', help='')
parser.add_argument('-boo', '--base_on_original', action='store_true', help='true -> use original_fs_dict_dir')
parser.add_argument('-ne', '--no_example', action='store_true', help='true -> no example prompt')
parser.add_argument('-oe', '--one_example', action='store_true', help='true -> only one example prompt')
parser.add_argument('-lbd', '--load_base_dir', type=str, default='', help='load base result/ndcg file dir')
parser.add_argument('-uuc', '--use_user_cluster', action='store_true', help='true -> use user cluster')
parser.add_argument('-uf', '--user_first', action='store_true', help='true -> user first')
parser.add_argument('-dbg', '--DBG', type=int, default=0, help='debug mode')

args = parser.parse_args()
# print all args
pprint(args.__repr__())


# %%
__file__ = os.path.abspath(os.getcwd())
model_path = "/root/autodl-tmp/models/Llama-3-8B-Instruct"
dataset_path = "/root/autodl-tmp/SRMR-Rec/datasets/"

dataset_name = args.dataset_name
bs = args.batch_size
bs2 = args.batch_size2 if args.batch_size2!=0 else max(int(bs/2),1)
mil = args.max_inter_len
gold_examples_dir = args.gold_examples_dir.format(dataset_name=dataset_name)
original_fs_dict_dir = args.original_fs_dict_dir.format(dataset_name=dataset_name)
output_dir = args.output_dir.format(ex=args.ex,dataset_name=dataset_name)
fs_temperature = args.fs_temperature
fs_do_sample = args.fs_do_sample
refl_type = args.refl_type
args.load_base_dir = args.load_base_dir.format(dataset_name=dataset_name) if args.load_base_dir!='' else ''


setproctitle.setproctitle(f'{dataset_name}|{args.ex}|{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# %%
#model = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True)
model = LLM(
    model=model_path, 
    tensor_parallel_size=1, 
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    max_model_len=16384
)
tokenizer = model.get_tokenizer()


tokenizer = model.get_tokenizer()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256
)

if dataset_name == 'Amazon_Arts':
    CF_model_path = '/root/autodl-tmp/SRMR-Rec/saved/DMF-Amazon_Arts-Aug-07-2026_19-21-04.pth'
elif dataset_name == "Amazon_Games":
    CF_model_path = '/root/autodl-tmp/SRMR-Rec/saved/DMF-Amazon_Games-Jul-30-2026_15-53-34.pth'
elif dataset_name == "Amazon_Instruments":
    CF_model_path = '/root/autodl-tmp/SRMR-Rec/saved/DMF-Aug-06-2026_10-34-17.pth'
data_path = f'/root/autodl-tmp/SRMR-Rec/datasets/{dataset_name}'

CF_model = None
dataset = None

config, CF_model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=CF_model_path, data_path=data_path,device='cuda')
# %%
pprint(rec_prompt)


# %%
item_info_df = pd.read_csv(f"{dataset_path}{dataset_name}/{dataset_name}.item", 
                   sep="\t", header=0, 
                   names=["item_id", "title", "categories", "brand", "sales_type","sales_rank"])
item_info_df['item_id'] = item_info_df['item_id'].astype(str)
item_info_dict = item_info_df.set_index('item_id').to_dict('index')
# item_info_dict[item_id] =  {'title': 'Creative Ideas 3&quot; Inch Solid Grosgrain Ribbon, 25 Yards, Pink, 25 yd',
                            #  'brand': "'Sewing', 'Ribbons'",
                            #  'categories': 'Creative Ideas',
                            #  'sales_type': 'Arts, Crafts & Sewing',
                            #  'sales_rank': nan}
if os.path.exists(gold_examples_dir):    
    try:                         
        with open(os.path.join(gold_examples_dir,'gold_improves.json'),'r')as f:
            gold_improves = json.load(f)
        with open(os.path.join(gold_examples_dir,'gold_fs_uids.json'),'r')as f:
            gold_fs_uids = json.load(f)
        with open(os.path.join(gold_examples_dir,'gold_fs_info.json'),'r')as f:
            gold_fs_info = json.load(f)
            for uid,imp in zip(gold_fs_uids,gold_improves):
                gold_fs_info[uid]['improve'] = imp
    except:
        with open(os.path.join(gold_examples_dir,'gold_improves_v0.json'),'r')as f:
            gold_improves = json.load(f)
        with open(os.path.join(gold_examples_dir,'gold_fs_uids_v0.json'),'r')as f:
            gold_fs_uids = json.load(f)
        with open(os.path.join(gold_examples_dir,'gold_fs_info_v0.json'),'r')as f:
            gold_fs_info = json.load(f)
            for uid,imp in zip(gold_fs_uids,gold_improves):
                gold_fs_info[uid]['improve'] = imp
else: # NOTE --gold_examples_dir = '', no loading
    gold_improves = []
    gold_fs_uids = []
    gold_fs_info = {}
    
    

# %%
with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/new_valid_samples.json","r")as f:
    valid_samples = json.load(f)
with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/new_test_samples.json","r")as f:
    test_samples = json.load(f)

for uid in valid_samples:
    valid_samples[uid]['-1']['inters'] = valid_samples[uid]['-1']['inters'][-mil:]
    test_samples[uid]['-1']['inters'] = test_samples[uid]['-1']['inters'][-mil:]

with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/users_1000.json",'r')as f:
    users = json.load(f)


safe_users = []
for uid in users:
    if uid in valid_samples and uid in test_samples:
        safe_users.append(uid)
    elif str(uid) in valid_samples and str(uid) in test_samples:
        safe_users.append(str(uid))


users = safe_users


if args.DBG == 1:
    users = users[:10]


# %%
import numpy as np
from tqdm import tqdm
import re
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

global_false_count = 0

def get_ndcg(rec_list: list, label: str) -> float:
    global global_false_count
    ndcg = 0.0
    is_hit = False
    
    clean_label = clean_text(label)
    
    for ith, rec_item_title in enumerate(rec_list):
        clean_rec = clean_text(rec_item_title)
        
        if clean_label in clean_rec:
            ndcg = 1 / np.log2(ith + 2)
            is_hit = True
            break
            
    return ndcg
def get_mean(li):
    return sum(li)/len(li)

import re

def _lcs_len(a: str, b: str, max_len: int = 25) -> int:
    if not a or not b:
        return 0
    best = 0
    for i in range(len(a)):
        for j in range(min(i + max_len, len(a)), i + best, -1):
            if j - i <= best:
                break
            if a[i:j] in b:
                best = j - i
                break
    return best

def _lcs_ge(a: str, b: str, th: int) -> bool:
    if not a or not b or len(a) < th:
        return False
    for i in range(len(a) - th + 1):
        if a[i:i + th] in b:
            return True
    return False

def rec_str_2_list(response: str, top_k: int = 10) -> list:
    items = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if not re.match(r'^\d+[\.\)\-:\s]', line):
            continue
        cleaned = re.sub(r'^\d+[\.\)\-:\s]+', '', line).strip()
        cleaned = cleaned.strip('\'"')
        if cleaned:
            items.append(cleaned)
        if len(items) >= top_k:
            break
    return items



# get user's cluster
with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/user_cluster.json","r")as f:
    user_cluster_dict = json.load(f)
def get_user_cluster(user_id):
    if user_id in user_cluster_dict:
        return user_cluster_dict[user_id]
    elif str(user_id) in user_cluster_dict:
        return user_cluster_dict[str(user_id)]
    else:
        try:
            if int(user_id) in user_cluster_dict:
                return user_cluster_dict[int(user_id)]
        except ValueError:
            pass
            
    return 0
# %%

from vllm import SamplingParams
from typing import List, Union

def llm_generate(model, tokenizer, dialogs: Union[List[List[dict]], List[dict]], **kwargs) -> list:
    if isinstance(dialogs, list) and len(dialogs) > 0 and isinstance(dialogs[0], dict):
        dialogs = [dialogs]
    elif isinstance(dialogs, dict):
        dialogs = [[dialogs]]   
    prompts = tokenizer.apply_chat_template(
        dialogs,
        add_generation_prompt=True,
        tokenize=False
    )
    
    stop_token_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int):
        stop_token_ids.append(eot_id)
        
    do_sample = kwargs.get("do_sample", False)
    temperature = kwargs.get("temperature", 1.0) if do_sample else 0.0
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=kwargs.get("top_p", 0.9),
        max_tokens=kwargs.get("max_new_tokens", 512),
        stop_token_ids=stop_token_ids,
        n=kwargs.get("num_return_sequences", 1)
    )
    
    vllm_outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    
    results = [output.outputs[0].text.strip() for output in vllm_outputs]
    
    return results

# %%

def get_item_title(item_id:str)->str:
    if item_id in item_info_dict:
        return item_info_dict[item_id]['title']
    return ""
def get_item_category(item_id:str)->dict:
    all_categories = {}
    if item_id in item_info_dict:
        all_categories['title'] = item_info_dict[item_id]['title']
        if pd.isna(item_info_dict[item_id]['categories']):
            all_categories['categories'] = "Unknown"
        else:
            all_categories['categories'] = item_info_dict[item_id]['categories']
        if pd.isna(item_info_dict[item_id]['brand']):
            all_categories['brand'] = "Unknown"
        else:
            all_categories['brand'] = item_info_dict[item_id]['brand']
    return all_categories

def get_user_item_CF_score(user_id,item_id:str):
    # note that user_id has been converted to string, we only need to convert it back to int
    title = get_item_title(item_id)
    item_id = dataset.field2token_id['item_id'][item_id]
    user_id = torch.tensor([int(user_id)])
    item_id = torch.tensor([int(item_id)])
    inter_dict = {'user_id' : user_id, 'item_id' : item_id}
    inter = Interaction(inter_dict)
    return str({'title': title, "CF_score" : float(CF_model.predict(inter))})

def get_items_title(item_ids:list)->list:
    return [get_item_title(item_id) for item_id in item_ids]

GATE_FRAC = 0.3
GATE_LCS_TH = 6

def compute_base_frac(rec_list: list, inters: list, lcs_th: int = GATE_LCS_TH) -> float:
    inters_clean = [clean_text(get_item_title(i)) for i in inters]
    cnt = 0
    for t in rec_list[:10]:
        ct = clean_text(t)
        if any(_lcs_ge(h, ct, lcs_th) for h in inters_clean):
            cnt += 1
    return cnt / 10.0

def get_items_category(item_ids:list)->list:
    return [get_item_category(item_id) for item_id in item_ids]
def get_user_items_CF_score(user_id,item_ids:list)->list:
    return [get_user_item_CF_score(user_id, item_id) for item_id in item_ids]

# %%
def rec_v2(users,samples,bs=10,refl_data=None):
    result_dict = {}
    ndcg_dict = {}
    last_instruction_dict = {}
    dialogs,uids = [],[]
    for uid in tqdm(users):
        # print("GT:",target_item_title_dict[uid])
        if refl_data==None:
            user_query = rec_prompt_v2.format(inters=get_items_title(samples[uid]['-1']['inters']), # '-1', samples can be from valid or test
                                    candidate_set=get_items_title(samples[uid]['-1']['candidate_set'])).replace('[[candidate_set_size]]', str(len(samples[uid]['-1']['candidate_set'])))
        else:
            user_query = rec_w_reflect_prompt_v3.format(
                inters=get_items_title(samples[uid]['-1']['inters']),
                candidate_set=get_items_title(samples[uid]['-1']['candidate_set']),
                reflect=refl_data[uid]).replace('[[candidate_set_size]]', str(len(samples[uid]['-1']['candidate_set'])))
                                   
        last_instruction_dict[uid] = user_query  
        message = [
            {"role": "system", "content": "You are a recommendation assistant."},
            {"role": "user", "content": user_query},
        ]
        
        dialogs.append(message)
        uids.append(uid)
        
    batch_dialogs = [dialogs[i:i+bs] for i in range(0,len(dialogs),bs)]
    batch_uids = [uids[i:i+bs] for i in range(0,len(uids),bs)]
    
    for dialog_batch,uid_batch in tzip(batch_dialogs,batch_uids):
        outputs = llm_generate(model, tokenizer, dialog_batch, 
                                        
                                max_new_tokens=512, 
                                num_return_sequences=1,
                                do_sample=False,
                                temperature=0.1, # temperature higher means the more random output, please change this at your own risk
                                top_p=0.9,
                                )
        for response,uid in zip(outputs,uid_batch):
            rec_list = rec_str_2_list(response) 
            

            if len(rec_list) < 5:
                fallback_query = rec_prompt_v2.format(
                    inters=get_items_title(samples[uid]['-1']['inters']),
                    candidate_set=get_items_title(samples[uid]['-1']['candidate_set'])
                ).replace('[[candidate_set_size]]', str(len(samples[uid]['-1']['candidate_set'])))
                rec_list = rec_str_2_list(
                    llm_generate(model, tokenizer,
                                [{"role": "system", "content": "You are a recommendation assistant."},
                                 {"role": "user", "content": fallback_query}],
                                max_new_tokens=512,
                                num_return_sequences=1,
                                do_sample=False,
                                temperature=0.1,
                                top_p=0.9)[0]
                )

            cand_ids = samples[uid]['-1']['candidate_set']
            cand_title_map = {clean_text(get_item_title(c)): get_item_title(c) for c in cand_ids}
            rec_list = [r for r in rec_list if clean_text(r) in cand_title_map]
            if len(rec_list) < 10:
                used = set(clean_text(r) for r in rec_list)
                rest = [t for ct, t in cand_title_map.items() if ct not in used]
                inters = [clean_text(get_item_title(i)) for i in samples[uid]['-1']['inters']]
                rest.sort(key=lambda t: max((_lcs_len(h, clean_text(t)) for h in inters), default=0),
                          reverse=True)
                rec_list = rec_list + rest[:10 - len(rec_list)]

            if refl_data is None:
                inters_clean = [clean_text(get_item_title(i)) for i in samples[uid]['-1']['inters']]
                rec_list = sorted(rec_list, key=lambda t: max((_lcs_len(h, clean_text(t)) for h in inters_clean), default=0), reverse=True)

            result_dict[uid] = rec_list
            ndcg_dict[uid] = get_ndcg(rec_list, label=get_item_title(samples[uid]['-1']['target_item']))

    return result_dict,ndcg_dict,last_instruction_dict

def sample_examples(candidates, probabilities,size=2, mode='p'):
    if mode=='p':
        probabilities = np.array(probabilities) / np.sum(probabilities)
        indices = np.random.choice(len(candidates), size=size, replace=False, p=probabilities)
        return [candidates[i] for i in indices]
    elif mode=='random':
        indices = np.random.choice(len(candidates), size=size, replace=False)
        return [candidates[i] for i in indices]

def sample_examples_w_cluster(candidates, probabilities, cluster_id,size=2, mode='p'):
    if candidates == []:
        return []
    # elif len(candidates) == 1:
    #     return candidates
    
    # use cluster_id, only select examples from the same cluster
    all_clusters = []
    for candidate in candidates:
        all_clusters.append(get_user_cluster(candidate))
    # examples are too few, return None
    
    cluster_indices = [i for i in range(len(candidates)) if all_clusters[i] == cluster_id]
    if not cluster_indices:
        return []
    if len(cluster_indices) == 1:
        return [candidates[cluster_indices[0]]]
    
    cluster_indices = np.array(cluster_indices)
    probabilities = np.array(probabilities)
    if mode=='p':
        cluster_probabilities = np.array(probabilities[cluster_indices]) / np.sum(probabilities[cluster_indices])
        indices = np.random.choice(len(cluster_indices), size=size, replace=False, p=cluster_probabilities)
        return [candidates[cluster_indices[i]] for i in indices]
    elif mode=='random':
        indices = np.random.choice(len(cluster_indices), size=size, replace=False)
        return [candidates[cluster_indices[i]] for i in indices]
        

def remove_examples_templete(user_query:str)->str:
    idx1 = user_query.find('Here are some examples')
    idx2 = user_query.find('Now, analyze the following case')
    if idx1 == -1 or idx2 == -1 or idx2 < idx1:
        return user_query
    return  user_query[:idx1] + user_query[idx2:]

def remove_1_example_templete(user_query:str)->str:
    marker = '\n## Previous'
    first = user_query.find(marker)
    second = user_query.find(marker, first + 1) if first != -1 else -1
    idx2 = user_query.find('Now, analyze the following case')
    if second == -1 or idx2 == -1 or second > idx2:
        return user_query
    return  user_query[:second + 1] + user_query[idx2:]

# def get_examples()
def get_fs_prompt(refl_type:str, instruction, response:str, last_act_his_str:str, user_id=None)->str:
    if args.use_user_cluster:
        try:
            assert user_id is not None 
        except: 
            ValueError("user_id must be provided when using user cluster")
        cluster_id = get_user_cluster(user_id)
    
    if refl_type == 'EP':
        user_query = fs_prompts[refl_type].format(
                    instruction=instruction,
                    response=response,
                    last_act_his_str=last_act_his_str,
                    # last_act_his_str=random.sample(candidate_set_title_dict[uid],10),
                ) 
        if args.no_example or gold_fs_uids==[]: 
            # if there are no gold examples, remove the example part in the query
            user_query = remove_examples_templete(user_query)
        else:
            if args.one_example: # only one example (demonstration) is provided
                user_query = remove_1_example_templete(user_query) # NOTE remove the second example
                if args.use_user_cluster: # Group level refining
                    example_uids = sample_examples_w_cluster(gold_fs_uids,gold_improves,cluster_id,size=1,mode='p')
                else:
                    example_uids = sample_examples(gold_fs_uids,gold_improves,size=1,mode='p') # Global level refining
            else:
                if args.use_user_cluster: # Group level refining
                    example_uids = sample_examples_w_cluster(gold_fs_uids,gold_improves,cluster_id,size=2,mode='p')
                else:
                    example_uids = sample_examples(gold_fs_uids,gold_improves,size=2,mode='p') # Global level refining
                if len(example_uids) == 1:
                    user_query = remove_1_example_templete(user_query)    
            # note that in some clusters, there may be no examples, or only one example        
            if not example_uids:
                user_query = remove_examples_templete(user_query)
            else: 
                if user_id in gold_fs_uids and args.user_first: # Individual level refining
                    example_uids = [user_id] + example_uids
                    example_uids = example_uids[:-1]
                for ith,example_uid in enumerate(example_uids):
                    fs_info = gold_fs_info[example_uid]
                    previous_instruction = fs_info['previous_instruction']
                    target_item = fs_info['target_item']
                    last_act_his = fs_info['last_act_his']
                    fs = fs_info['fs']
                    user_query = user_query.replace(f'[[previous_instruction {ith+1}]]',previous_instruction)
                    user_query = user_query.replace(f'[[target_item {ith+1}]]',get_item_title(target_item))
                    user_query = user_query.replace(f'[[last_act_his {ith+1}]]',str(last_act_his))
                    user_query = user_query.replace(f'[[fs {ith+1}]]',fs)
        

    elif refl_type == 'IP':
        # build the instruction for IP refl prompt
        inters_IP = get_items_category(instruction[0])
        candidate_IP = get_items_category(instruction[1])
        instruction_IP = instruction_meta_example
        instruction_IP = instruction_IP.replace("[[inters_1]]", str(inters_IP))
        instruction_IP = instruction_IP.replace("[[candidate_set_1]]", str(candidate_IP))
        
        cand_title_map = {clean_text(get_item_title(c)): c for c in instruction[1]}
        resp_item_ids = [cand_title_map[clean_text(t)] for t in (response if isinstance(response, list) else [])
                         if clean_text(t) in cand_title_map]
        response = get_items_category(resp_item_ids)
        user_query = fs_prompts[refl_type].format(
                    instruction=instruction_IP,
                    response=response,
                    last_act_his_str=last_act_his_str,
                    # last_act_his_str=random.sample(candidate_set_title_dict[uid],10),
                ) 
        # TODO
        if args.no_example or gold_fs_uids==[]: 
            # if there are no gold examples, remove the example part in the query
            user_query = remove_examples_templete(user_query)
        else:
            if args.one_example: # only one example (demonstration) is provided
                user_query = remove_1_example_templete(user_query) # NOTE remove the second example
                if args.use_user_cluster:
                    example_uids = sample_examples_w_cluster(gold_fs_uids,gold_improves,cluster_id,size=1,mode='p')
                else:
                    example_uids = sample_examples(gold_fs_uids,gold_improves,size=1,mode='p')
            else:
                if args.use_user_cluster:
                    example_uids = sample_examples_w_cluster(gold_fs_uids,gold_improves,cluster_id,size=2,mode='p')
                else:
                    example_uids = sample_examples(gold_fs_uids,gold_improves,size=2,mode='p')
                if len(example_uids) == 1:
                    user_query = remove_1_example_templete(user_query)
            # Note that in some clusters, there may be no examples, or only one example        
            if not example_uids:
                user_query = remove_examples_templete(user_query)
            else: 
                if user_id in gold_fs_uids and args.user_first: # Individual level refining
                    example_uids = [user_id] + example_uids
                    example_uids = example_uids[:-1]
                for ith,example_uid in enumerate(example_uids):
                    fs_info = gold_fs_info[example_uid]

                    previous_instruction_w_IP = fs_info['previous_instruction']
                    target_item = fs_info['target_item']

                    last_act_his = fs_info['last_act_his']
                    fs = fs_info['fs']
                    user_query = user_query.replace(f'[[item_IP_list {ith+1}]]',previous_instruction_w_IP)
                    user_query = user_query.replace(f'[[target_item {ith+1}]]',get_item_title(target_item))
                    user_query = user_query.replace(f'[[last_act_his {ith+1}]]',str(last_act_his))
                    user_query = user_query.replace(f'[[fs {ith+1}]]',fs)
            
    elif refl_type == 'CF':
        # build the instruction for CF refl prompt
        inters_CF = get_user_items_CF_score(user_id=user_id, item_ids=instruction[0])
        candidate_CF = get_user_items_CF_score(user_id=user_id, item_ids=instruction[1])
        instruction_CF = instruction_meta_example
        instruction_CF = instruction_CF.replace("[[inters_1]]",str(inters_CF))
        instruction_CF = instruction_CF.replace("[[candidate_set_1]]", str(candidate_CF))
        cand_title_map = {clean_text(get_item_title(c)): c for c in instruction[1]}
        resp_item_ids = [cand_title_map[clean_text(t)] for t in (response if isinstance(response, list) else [])
                         if clean_text(t) in cand_title_map]
        response = get_user_items_CF_score(user_id=user_id, item_ids=resp_item_ids)
        user_query = fs_prompts[refl_type].format(
                    instruction=instruction_CF,
                    response=response,
                    last_act_his_str=last_act_his_str,
                    # last_act_his_str=random.sample(candidate_set_title_dict[uid],10),
                ) 

        if args.no_example or gold_fs_uids==[]: 
            user_query = remove_examples_templete(user_query)
        else:
            if args.one_example:
                user_query = remove_1_example_templete(user_query)
                if args.use_user_cluster:
                    example_uids = sample_examples_w_cluster(gold_fs_uids,gold_improves,cluster_id,size=1,mode='p')
                else:
                    example_uids = sample_examples(gold_fs_uids,gold_improves,size=1,mode='p')
            else:
                if args.use_user_cluster:
                    example_uids = sample_examples_w_cluster(gold_fs_uids,gold_improves,cluster_id,size=2,mode='p')
                else:
                    example_uids = sample_examples(gold_fs_uids,gold_improves,size=2,mode='p')
                if len(example_uids) == 1:
                    user_query = remove_1_example_templete(user_query)    
            # note that in some clusters, there may be no examples, or only one example        
            if not example_uids:
                user_query = remove_examples_templete(user_query)
            else:
                if user_id in gold_fs_uids and args.user_first:
                    example_uids = [user_id] + example_uids
                    example_uids = example_uids[:-1]    
                for ith,example_uid in enumerate(example_uids):
                    fs_info = gold_fs_info[example_uid]

                    previous_instruction_w_CF = fs_info['previous_instruction']
                    target_item = fs_info['target_item']

                    last_act_his = fs_info['last_act_his']
                    fs = fs_info['fs']
                    user_query = user_query.replace(f'[[item_CF_scores_list {ith+1}]]',previous_instruction_w_CF)
                    user_query = user_query.replace(f'[[target_item {ith+1}]]',get_item_title(target_item))
                    user_query = user_query.replace(f'[[last_act_his {ith+1}]]',str(last_act_his))
                    user_query = user_query.replace(f'[[fs {ith+1}]]',fs)
    else:   
        raise ValueError("refl_type must be in ['EP','IP','CF']")
    return user_query

def get_critic_prompt(refl_v1: str) -> str:
    return f"""The initial preference conclusion is:
{refl_v1}

Please review and polish the conclusion for better clarity and logic. As a strict consumer behavior critic, point out any logical flaws or hallucinations in the conclusion, and provide a concise revision suggestion."""



def get_generator_v2_prompt(refl_v1: str, critic_feedback: str) -> str:
    return f"""You previously generated this user preference reflection:
{refl_v1}

A strict critic reviewed your conclusion and provided the following feedback:
{critic_feedback}

Task:
Please absorb these criticisms, correct any logical flaws or hallucinations 
in your original reflection, and output a highly accurate and rigorous final 
version of the user's preference. Address each criticism point explicitly 
in your revision. Do not just summarize the feedback; rewrite the conclusion 
so that it is consistent with all available evidence.

CRITICAL CONSTRAINTS:
- Output MUST be under 150 words.
- Use the exact format: PREFERENCE: ... | ANALYSIS: ... | SUGGESTIONS: ...
- DO NOT mention any specific item titles.
- Output only the final reflection text, without any conversational filler."""
    
from tqdm.contrib import tzip

def fs_debate_v2(users, samples, last_instruction_dict, base_result_dict,
                 bs=10, refl_type='IP', model=None, tokenizer=None, 
                 fs_do_sample=True, fs_temperature=1.0):
    fs_final_dict = {}

    
    def _run_batch_generation(current_dialogs, current_uids, desc="LLM Gen"):
        out_dict = {}
        batch_dialogs = [current_dialogs[i:i+bs] for i in range(0, len(current_dialogs), bs)]
        batch_uids = [current_uids[i:i+bs] for i in range(0, len(current_uids), bs)]
        
        for dialog_batch, uid_batch in tzip(batch_dialogs, batch_uids):
            if not fs_do_sample:
                outputs = llm_generate(model, tokenizer, dialog_batch, 
                                       max_new_tokens=512, num_return_sequences=1, do_sample=False)
            else:
                outputs = llm_generate(model, tokenizer, dialog_batch, 
                                       max_new_tokens=512, num_return_sequences=1, do_sample=True,
                                       temperature=fs_temperature, top_p=0.9)
            for response, uid in zip(outputs, uid_batch):
                out_dict[uid] = response
        return out_dict

    
    dialogs_v1 = []
    uids = []
    for uid in users:
        last_act_his_str = str(get_items_title(samples[uid]['-1']['inters']))

        if refl_type == 'EP':
            instruction = last_instruction_dict[uid]
        else:  # IP / CF
            instruction = (samples[uid]['-1']['inters'], samples[uid]['-1']['candidate_set'])
        user_query = get_fs_prompt(
            refl_type=refl_type,                           
            instruction=instruction,        
            response=base_result_dict[uid] if base_result_dict else "",                
            last_act_his_str=last_act_his_str,             
            user_id=uid
        )
        message_v1 = [
            {"role": "system", "content": "You are an expert recommendation system analyst."},
            {"role": "user", "content": user_query}
        ]
        dialogs_v1.append(message_v1)
        uids.append(uid)

    refl_v1_dict = _run_batch_generation(dialogs_v1, uids, desc="stage 1/3 (draft generation)")

    
    dialogs_critic = []
    for uid in uids:
        refl_v1 = refl_v1_dict[uid]
        critic_query = get_critic_prompt(refl_v1)
        message_critic = [
            {"role": "system", "content": "You are a harsh and logical consumer behavior critic."},
            {"role": "user", "content": critic_query}
        ]
        dialogs_critic.append(message_critic)

    feedback_dict = _run_batch_generation(dialogs_critic, uids, desc="stage 2/3 (critic review)")

    
    dialogs_v2 = []
    for i, uid in enumerate(uids):
        refl_v1 = refl_v1_dict[uid]
        feedback = feedback_dict[uid]
        generator_v2_query = get_generator_v2_prompt(refl_v1, feedback)
        message_v2 = dialogs_v1[i] + [
            {"role": "assistant", "content": refl_v1},
            {"role": "user", "content": generator_v2_query}
        ]
        dialogs_v2.append(message_v2)

    fs_final_dict = _run_batch_generation(dialogs_v2, uids, desc="stage 3/3 (final revision)")

    return fs_final_dict

    
# %%
def fs_v2(users, samples, last_instruction_dict, base_result_dict, bs=10,refl_type='IP'):
    assert refl_type in ['EP','IP','CF']
    fs_dict = {}
    dialogs,uids = [],[]
    for uid in tqdm(users):
        # print("GT:",target_item_title_dict[uid])
        if refl_type == 'EP':
            user_query = get_fs_prompt(
                refl_type, 
                instruction=last_instruction_dict[uid],
                response=get_item_title(samples[uid]['-1']['target_item']),
                last_act_his_str=base_result_dict[uid],
                user_id=uid
            )
        else:
            user_query = get_fs_prompt(
                refl_type,
                instruction=[samples[uid]['-1']['inters'], samples[uid]['-1']['candidate_set']],
                response=samples[uid]['-1']['target_item'],
                last_act_his_str=base_result_dict[uid],
                user_id=uid
            )
        message = [
            {"role": "system", "content": "You are a recommendation assistant."},
            {"role": "user", "content": user_query},
        ]
        
        dialogs.append(message)
        uids.append(uid)
        
    batch_dialogs = [dialogs[i:i+bs] for i in range(0,len(dialogs),bs)]
    batch_uids = [uids[i:i+bs] for i in range(0,len(uids),bs)]
    for dialog_batch,uid_batch in tzip(batch_dialogs,batch_uids):
        if not fs_do_sample:
            outputs = llm_generate(model, tokenizer, dialog_batch, 
                                            
                                    max_new_tokens=512, 
                                    num_return_sequences=1,
                                    do_sample=False,
                                    )
        else:
            outputs = llm_generate(model, tokenizer, dialog_batch, 
                                            
                                    max_new_tokens=512, 
                                    num_return_sequences=1,
                                    do_sample=True,
                                    temperature=fs_temperature, 
                                    top_p=0.9,
                                    )
        for response,uid in zip(outputs,uid_batch):
            fs_dict[uid] = response
    return fs_dict


# get new iteration's gold_fs_uids, gold_improves
def get_new_good_gold_uids(ndcg_dict,base_ndcg_dict):

    new_good_improves,new_good_fs_uids = [],[]
    new_gold_improves,new_gold_fs_uids = [],[]
    for uid in users:
        if base_ndcg_dict[uid] < ndcg_dict[uid]:
            improve = ndcg_dict[uid]-base_ndcg_dict[uid]
            new_good_fs_uids.append(uid)
            new_good_improves.append(improve)
            if uid in gold_fs_uids:
                if 0.6 <= ndcg_dict[uid]:
                    new_gold_fs_uids.append(uid)
                    new_gold_improves.append(improve)
            elif 0.5 <= ndcg_dict[uid]:
                new_gold_fs_uids.append(uid)
                new_gold_improves.append(improve)
                
            
    return dict(new_good_improves=new_good_improves,new_good_fs_uids=new_good_fs_uids,
                new_gold_improves=new_gold_improves,new_gold_fs_uids=new_gold_fs_uids)


def fusion_gold_pool(new_dict, valid_fs_dict, valid_base_result_dict,last_instruction_dict,
                     fusion_mode='max', good_as_gold=False):
    new_gold_improves = new_dict['new_gold_improves'] if not good_as_gold else new_dict['new_good_improves']
    new_gold_fs_uids = new_dict['new_gold_fs_uids'] if not good_as_gold else new_dict['new_good_fs_uids']
    
    # update gold_fs_info with new iteration's infomation
    new_gold_fs_info = {}
    for uid,improve in zip(new_gold_fs_uids,new_gold_improves):
        target_item_id = valid_samples[uid]['-1']['target_item']
        raw_his = valid_base_result_dict[uid]
        if isinstance(raw_his, list):
            clean_his = [item.strip('\'"') for item in raw_his 
                        if isinstance(item, str)]
        else:
            clean_his = raw_his
            
        new_gold_fs_info[uid] = dict(
            last_act_his=clean_his,
            fs=valid_fs_dict[uid],
            target_item=get_item_title(target_item_id),
            previous_instruction=last_instruction_dict[uid],
            improve=improve
        )
       

  
    
    if fusion_mode == 'new':
        for uid in gold_fs_info:
            if uid not in new_gold_fs_info:
                new_gold_fs_info[uid] = gold_fs_info[uid] # fusion the new_gold_fs_info with the old one if the user is not in the new_gold_fs_info
    elif fusion_mode == 'max':
        for uid in gold_fs_info:
            if uid not in new_gold_fs_info:
                new_gold_fs_info[uid] = gold_fs_info[uid] # fusion the new_gold_fs_info with the old one if the user is not in the new_gold_fs_info
            else:
                old_imp = gold_fs_info[uid]['improve']
                new_imp = new_gold_fs_info[uid]['improve']
                if new_imp < old_imp:
                    new_gold_fs_info[uid] = gold_fs_info[uid] 
    else:
        raise ValueError("fusion_mode must be in ['max','new']")
    
    return new_gold_fs_info
    
os.makedirs(output_dir,exist_ok=True)
    

# %%




if args.load_base_dir:
    try:
        with open(os.path.join(args.load_base_dir,"valid_base_result_dict.json"),"r")as f:
            valid_base_result_dict = json.load(f)
        with open(os.path.join(args.load_base_dir,"valid_base_ndcg_dict.json"),"r")as f:
            valid_base_ndcg_dict = json.load(f)
        with open(os.path.join(args.load_base_dir,"last_instruction_dict.json"),"r")as f:
            last_instruction_dict = json.load(f)
        print("||valid_base_result_dict,valid_base_ndcg_dict,last_instruction_dict loaded!")
    except:
        print("||WARNING: args.load_base_dir is provided but valid_base_result_dict.json or valid_base_ndcg_dict.json is not found!")
        valid_base_result_dict, \
            valid_base_ndcg_dict, \
            last_instruction_dict = rec_v2(users,valid_samples,bs=bs)
        with open(os.path.join(output_dir,"valid_base_result_dict.json"),"w")as f:
            json.dump(valid_base_result_dict,f)
        with open(os.path.join(output_dir,"valid_base_ndcg_dict.json"),"w")as f:
            json.dump(valid_base_ndcg_dict,f)
        with open(os.path.join(output_dir,"last_instruction_dict.json"),"w")as f:
            json.dump(last_instruction_dict,f)
        
else:
    print("||No base_dir provided! Start from scratch......")
    valid_base_result_dict, \
        valid_base_ndcg_dict, \
        last_instruction_dict = rec_v2(users,valid_samples,bs=bs)
    with open(os.path.join(output_dir,"valid_base_result_dict.json"),"w")as f:
        json.dump(valid_base_result_dict,f)
    with open(os.path.join(output_dir,"valid_base_ndcg_dict.json"),"w")as f:
        json.dump(valid_base_ndcg_dict,f)
    with open(os.path.join(output_dir,"last_instruction_dict.json"),"w")as f:
        json.dump(last_instruction_dict,f)
print()
print('='*10)
print("||>>> base valid_ndcg",get_mean(valid_base_ndcg_dict.values())) # 0.32762017669973675
print('='*10)
print()
    

# %%
if args.load_base_dir:
    try:
        with open(os.path.join(args.load_base_dir,"test_base_result_dict.json"),"r")as f:
            test_base_result_dict = json.load(f)
        with open(os.path.join(args.load_base_dir,"test_base_ndcg_dict.json"),"r")as f:
            test_base_ndcg_dict = json.load(f)
        try:
            with open(os.path.join(args.load_base_dir,"test_last_instruction_dict.json"),"r")as f:
                test_last_instruction_dict = json.load(f)
        except:
            test_last_instruction_dict = None
        print("||test_base_result_dict,test_base_ndcg_dict loaded!")
    except:
        print("||WARNING: args.load_base_dir is provided but test_base_result_dict.json or test_base_ndcg_dict.json is not found!")
        test_base_result_dict, \
            test_base_ndcg_dict, \
            test_last_instruction_dict = rec_v2(users,test_samples,bs=bs)
        with open(os.path.join(output_dir,"test_base_result_dict.json"),"w")as f:
            json.dump(test_base_result_dict,f)
        with open(os.path.join(output_dir,"test_base_ndcg_dict.json"),"w")as f:
            json.dump(test_base_ndcg_dict,f)
else:
    print("||No base_dir provided! Start from scratch......")
    test_base_result_dict, \
        test_base_ndcg_dict, \
        test_last_instruction_dict = rec_v2(users,test_samples,bs=bs)
    with open(os.path.join(output_dir,"test_base_result_dict.json"),"w")as f:
        json.dump(test_base_result_dict,f)
    with open(os.path.join(output_dir,"test_base_ndcg_dict.json"),"w")as f:
        json.dump(test_base_ndcg_dict,f)
print()
print('='*10)
print("||>>> base test_ndcg",get_mean(test_base_ndcg_dict.values())) 
print('='*10)
print()



valid_frac_dict = {}
for uid in users:
    if uid not in valid_base_result_dict or uid not in valid_samples:
        valid_frac_dict[uid] = 1.0
        continue
    valid_frac_dict[uid] = compute_base_frac(
        valid_base_result_dict[uid], valid_samples[uid]['-1']['inters'])
test_frac_dict = {}
for uid in users:
    if uid not in test_base_result_dict or uid not in test_samples:
        test_frac_dict[uid] = 1.0
        continue
    test_frac_dict[uid] = compute_base_frac(
        test_base_result_dict[uid], test_samples[uid]['-1']['inters'])
with open(os.path.join(output_dir,"valid_frac_dict.json"),"w")as f:
    json.dump(valid_frac_dict,f)
with open(os.path.join(output_dir,"test_frac_dict.json"),"w")as f:
    json.dump(test_frac_dict,f)
base_frac_dict = test_frac_dict
with open(os.path.join(output_dir,"base_frac_dict.json"),"w")as f:
    json.dump(base_frac_dict,f)

valid_all_refl_path = os.path.join(output_dir, "valid_all_refl_result_dict.json")
if os.path.exists(valid_all_refl_path):
    with open(valid_all_refl_path, "r") as f:
        valid_all_refl_result_dict = json.load(f)
    with open(os.path.join(output_dir, "valid_all_refl_ndcg_dict.json"), "r") as f:
        valid_all_refl_ndcg_dict = json.load(f)
else:
    valid_all_fs_dict = fs_debate_v2(
        users=users,
        samples=valid_samples,
        last_instruction_dict=last_instruction_dict,
        base_result_dict=valid_base_result_dict,
        bs=args.batch_size2,
        refl_type=refl_type,
        model=model,
        tokenizer=tokenizer,
        fs_do_sample=True,
        fs_temperature=getattr(args, 'fs_temperature', 1.0))
    valid_all_refl_result_dict, \
        valid_all_refl_ndcg_dict, _ = rec_v2(users, valid_samples,
                                              refl_data=valid_all_fs_dict, bs=bs)
    with open(valid_all_refl_path, "w") as f:
        json.dump(valid_all_refl_result_dict, f)
    with open(os.path.join(output_dir, "valid_all_refl_ndcg_dict.json"), "w") as f:
        json.dump(valid_all_refl_ndcg_dict, f)

tau_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
best_tau, best_valid_ndcg = None, -1.0
for tau in tau_candidates:
    mixed_scores = []
    for uid in users:
        if valid_frac_dict.get(uid, 1.0) >= tau:
            mixed_scores.append(valid_base_ndcg_dict[uid])
        else:
            mixed_scores.append(valid_all_refl_ndcg_dict.get(
                uid, valid_base_ndcg_dict[uid]))
    mean_ndcg = get_mean(mixed_scores)
    if mean_ndcg > best_valid_ndcg:
        best_valid_ndcg = mean_ndcg
        best_tau = tau
GATE_FRAC = best_tau

valid_low_conf_users = [u for u in users if valid_frac_dict.get(u, 1.0) < GATE_FRAC]
test_low_conf_users = [u for u in users if test_frac_dict.get(u, 1.0) < GATE_FRAC]
low_conf_users = test_low_conf_users


for iter_i in range(args.max_iter):
    iter_json_path = os.path.join(output_dir, f"iter_{iter_i}.json")
    if os.path.exists(iter_json_path):
        with open(iter_json_path, "r") as f:
            _saved_iter = json.load(f)
        gold_fs_info = _saved_iter["gold_fs_info"]
        gold_fs_uids = list(gold_fs_info.keys())
        gold_improves = [gold_fs_info[uid]["improve"] for uid in gold_fs_uids]
        continue
    if iter_i == 0 and args.base_on_original:
        with open(os.path.join(original_fs_dict_dir,'valid_fs_dict.json'),"r")as f:
            valid_fs_dict = json.load(f)
        with open(os.path.join(original_fs_dict_dir,'valid_refl1_result_dict.json'),"r")as f:
            valid_refl1_result_dict = json.load(f)
        with open(os.path.join(original_fs_dict_dir,'valid_refl1_ndcg_dict.json'),"r")as f:
            valid_refl1_ndcg_dict = json.load(f)
        for uid in users:
            if uid not in valid_low_conf_users:
                valid_refl1_result_dict[uid] = valid_base_result_dict[uid]
                valid_refl1_ndcg_dict[uid] = valid_base_ndcg_dict[uid]
    else:
        print(f"\n--- [Innovation] Running Multi-Agent Debate for Iteration {iter_i} ---")
        base_temp = getattr(args, 'fs_temperature', 1.0)

        current_temp = max(0.1, base_temp - (iter_i * 0.02))


        valid_fs_dict = fs_debate_v2(
            users=valid_low_conf_users,
            samples=valid_samples,
            last_instruction_dict=last_instruction_dict,
            base_result_dict=valid_base_result_dict,
            bs=args.batch_size2,
            refl_type=refl_type,
            model=model,
            tokenizer=tokenizer,
            fs_do_sample=True,
            fs_temperature=current_temp
        )
        test_only_low_conf = [u for u in test_low_conf_users if u not in valid_low_conf_users]
        if test_only_low_conf:
            if refl_type == 'EP' and test_last_instruction_dict is None:
                _, _, test_last_instruction_dict = rec_v2(users, test_samples, bs=bs)
            elif test_last_instruction_dict is None:
                test_last_instruction_dict = {}
            test_fs_dict = fs_debate_v2(
                users=test_only_low_conf,
                samples=test_samples,
                last_instruction_dict=test_last_instruction_dict,
                base_result_dict=test_base_result_dict,
                bs=args.batch_size2,
                refl_type=refl_type,
                model=model,
                tokenizer=tokenizer,
                fs_do_sample=True,
                fs_temperature=current_temp
            )
            valid_fs_dict.update(test_fs_dict)
        valid_refl1_result_dict, \
            valid_refl1_ndcg_dict, _ = rec_v2(valid_low_conf_users,valid_samples,
                                                refl_data=valid_fs_dict,bs=bs)
        for uid in users:
            if uid not in valid_low_conf_users:
                valid_refl1_result_dict[uid] = valid_base_result_dict[uid]
                valid_refl1_ndcg_dict[uid] = valid_base_ndcg_dict[uid]
    print()
    print('='*10)
    print(f'||>>> iter_{iter_i} valid_ndcg',round(get_mean(valid_refl1_ndcg_dict.values()),6)) # 0.33868528072345416
    print('='*10)
    print()

    
    print("||get_new_good_gold_uids... ")
    # get new iteration's gold_fs_uids, gold_improves, gold_fs_info
    new_dict = get_new_good_gold_uids(valid_refl1_ndcg_dict, valid_base_ndcg_dict)
    # fusion gold pool
    print("||fusion_gold_pool... ")
    if refl_type == 'EP':
        gold_instruction_dict = last_instruction_dict
    else:
        gold_instruction_dict = {}
        for uid in valid_low_conf_users:
            inters_ids = valid_samples[uid]['-1']['inters']
            candidate_ids = valid_samples[uid]['-1']['candidate_set']
            inst_text = instruction_meta_example
            if refl_type == 'IP':
                inst_text = inst_text.replace("[[inters_1]]", str(get_items_category(inters_ids)))
                inst_text = inst_text.replace("[[candidate_set_1]]", str(get_items_category(candidate_ids)))
            else:  # CF
                inst_text = inst_text.replace("[[inters_1]]", str(get_user_items_CF_score(user_id=uid, item_ids=inters_ids)))
                inst_text = inst_text.replace("[[candidate_set_1]]", str(get_user_items_CF_score(user_id=uid, item_ids=candidate_ids)))
            gold_instruction_dict[uid] = inst_text
    new_gold_fs_info = fusion_gold_pool(new_dict,
                     valid_fs_dict,
                     valid_base_result_dict,
                     gold_instruction_dict,
                     fusion_mode=args.fusion_mode, good_as_gold=False)
    # replace global gold_fs_info
    print("||replace global gold_fs_info... {}-->{}".format(len(gold_fs_info),len(new_gold_fs_info)))
    gold_fs_info = new_gold_fs_info
    gold_fs_uids = list(new_gold_fs_info.keys())
    gold_improves = [new_gold_fs_info[uid]['improve'] for uid in gold_fs_uids]

    # # %%
    if iter_i == 0 and args.base_on_original:
        with open(os.path.join(original_fs_dict_dir,'test_refl1_result_dict.json'),"r")as f:
            test_refl1_result_dict = json.load(f)
        with open(os.path.join(original_fs_dict_dir,'test_refl1_ndcg_dict.json'),"r")as f:
            test_refl1_ndcg_dict = json.load(f)
        for uid in users:
            if uid not in test_low_conf_users:
                test_refl1_result_dict[uid] = test_base_result_dict[uid]
                test_refl1_ndcg_dict[uid] = test_base_ndcg_dict[uid]
    else:
        test_refl1_result_dict, \
            test_refl1_ndcg_dict, _ = rec_v2(test_low_conf_users,test_samples,refl_data=valid_fs_dict,bs=bs)
        for uid in users:
            if uid not in test_low_conf_users:
                test_refl1_result_dict[uid] = test_base_result_dict[uid]
                test_refl1_ndcg_dict[uid] = test_base_ndcg_dict[uid]
    print()
    print('='*10)
    print(f'||>>> iter_{iter_i} test_ndcg',round(get_mean(test_refl1_ndcg_dict.values()),6)) # 0.33868528072345416
    print('='*10)
    print()
    
    
    with open(os.path.join(output_dir,f"iter_{iter_i}.json"),"w")as f:
        json.dump(
            dict(test_refl1_result_dict=test_refl1_result_dict,
                 test_refl1_ndcg_dict=test_refl1_ndcg_dict,
                 valid_refl1_ndcg_dict=valid_refl1_ndcg_dict,
                 valid_refl1_result_dict=valid_refl1_result_dict,gold_fs_info=gold_fs_info),f)
# %%
