# %%
import transformers
import torch
import torch.nn as nn
from prompt.recommend_prompt import *
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
from utils.data import load_data_and_model, load_CF_SAS, generate_candidate_emb
from recbole.data.interaction import Interaction
import setproctitle
import datetime
import html
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from vllm import LLM, SamplingParams


random.seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser(description=".")


parser.add_argument('-bs', '--batch_size', type=int, default=10, help='Batch size')
# parser.add_argument('-bs2', '--batch_size2', type=int, default=0, help='Batch size')
parser.add_argument('-mil', '--max_inter_len', type=int, default=20, help='')
parser.add_argument('-dn', '--dataset_name', type=str, default='Amazon_Arts', help='Dataset name')
parser.add_argument('-bn', '--baseline_name', type=str, default='SRMR_wo_reflect', help='Baseline name')
parser.add_argument('-ex', type=str, default='TRY', help='name of experiment this time')
# --num_prompt
parser.add_argument("--num_prompt",action='store_true',help="using rec example num prompt")
parser.add_argument("--vllm",action='store_true',help="try vllm?")
# do sample & temperature setup
parser.add_argument("-ds", "--do_sample",type=int, default=1, help="whether vllm use beam search(do_sample=false to use)")
parser.add_argument("-t", "--temperature", type=float, default=1.0, help="temperature setup, only valid when do_sample=1.")
parser.add_argument("-ne","--no_example",action='store_true',help="no example")
# inters_num_prompt
parser.add_argument("--inters_num_prompt",action='store_true',help="using inters num prompt")
# candidate_set_num_prompt
parser.add_argument("--candidate_set_num_prompt",action='store_true',help="using candidate set num prompt")
# candidate_set_set_prompt
parser.add_argument("--candidate_set_set_prompt",action='store_true',help="using candidate set set prompt")
# waiting_min
parser.add_argument("--waiting_min",type=int,default=0,help="waiting min")
# refl_data path and iter num
parser.add_argument("--refl_result_dir",type=str,default="/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Instruments_EP_B1_maxFM_cluster_user_first_t1.0_te_new",help="refl result dir")
parser.add_argument("--iter_i",type=int,default=0,help="iter i of reflect file")
parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='gpu_memory_utilization for vllm')
# SRMR_w_reflect_version
parser.add_argument("--SRMR_w_reflect_version",type=int,default=0,choices=[0,1,2,3,4], help="SRMR_w_reflect_version")
# SUGGESTION only
parser.add_argument("--suggestion_only",action='store_true',help="SUGGESTION only")
# ppo selector model load, and all types of reflects' path
parser.add_argument("-pmp","--ppo_model_path",type=str,default="/root/autodl-tmp/SRMR-Rec/rl_select/Meta-Reflect/Amazon_Instruments_run_ppo_lr_a_0.001_lc_c_0.001_rt_0.9_bs_64_4_arm_0_iter_4_train_model_epoch_500.pth",help="ppo_model_path")
parser.add_argument("-crp", "--CF_refl_path", type=str, default='/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_CF_B1_maxFM_cluster_user_first_t1.0_te_new', help="CF reflect info dict")
parser.add_argument("-erp", "--EP_refl_path", type=str, default='/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_EP_B1_maxFM_cluster_user_first_t1.0_te_new', help="EP reflect info dict")
parser.add_argument("-irp", "--IP_refl_path", type=str, default='/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_IP_B1_maxFM_cluster_user_first_t1.0_te_new', help="IP reflect info dict")

# greedy selection in 3 types of reflections
parser.add_argument("-gs", "--greedy_selection",action="store_true", help="greedy selection")
# random selection in 3 types of reflections
parser.add_argument("-rs", "--random_selection",action="store_true", help="random selection")
# use all types of reflections
parser.add_argument("-at", "--all_types",action="store_true", help="use all types of reflections")
# use EP, IP only
parser.add_argument("-ta", "--EP_IP_only",action="store_true", help="use EP, IP only")
# use EP, CF only
parser.add_argument("-tc", "--EP_CF_only",action="store_true", help="use EP, CF only")
# use CF, IP only
parser.add_argument("-ca", "--CF_IP_only",action="store_true", help="use CF, IP only")

parser.add_argument("-as", "--arm_size", type=int, default=4, help="arm size")
parser.add_argument("-gate", "--gate_frac", type=float, default=0.3, help="[deprecated] retained for shell compatibility")
args = parser.parse_args()




if args.waiting_min>0:
    print(f"Waiting for {args.waiting_min} minutes...")
    import time
    time.sleep(args.waiting_min*60)
    print("Done!")

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(128, 64).cuda()
        self.fc2 = nn.Linear(64, 64).cuda()
        self.fc3 = nn.Linear(64, args.arm_size).cuda()
        self.activate_func = nn.Tanh().cuda()

    def forward(self, s):
        s = s.cuda()
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob


# %%
model_path = "/root/autodl-tmp/models/Llama-3-8B-Instruct"
# dataset_name = 'Amazon_Games'
dataset_name = args.dataset_name
bs = args.batch_size
# bs2 = args.batch_size2 if args.batch_size2!=0 else max(int(bs/2),1)
mil = args.max_inter_len
baseline_name = args.baseline_name
num_prompt = args.num_prompt
no_example = args.no_example
inters_num_prompt = args.inters_num_prompt
candidate_set_num_prompt = args.candidate_set_num_prompt
setproctitle.setproctitle(f'{dataset_name}|{args.ex}|{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# %%

tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
if args.vllm:
    model = LLM(model=model_path,seed=2024,gpu_memory_utilization=args.gpu_memory_utilization)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = tokenizer.bos_token_id


# %%
pprint(rec_prompts)

# %% [markdown]
# ## input data

# %%
item_info_df = pd.read_csv(f"/root/autodl-tmp/SRMR-Rec/datasets/{dataset_name}/{dataset_name}.item", 
                   sep="\t", header=0, 
                   names=["item_id", "title", "categories", "brand", "sales_type","sales_rank"])
item_info_dict = item_info_df.set_index('item_id').to_dict('index')
# item_info_dict[item_id] =  {'title': 'Creative Ideas 3&quot; Inch Solid Grosgrain Ribbon, 25 Yards, Pink, 25 yd',
                            #  'brand': "'Sewing', 'Ribbons'",
                            #  'categories': 'Creative Ideas',
                            #  'sales_type': 'Arts, Crafts & Sewing',
                            #  'sales_rank': nan}


# %%
with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/new_valid_samples.json","r")as f:
    valid_samples = json.load(f)
with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/new_test_samples.json","r")as f:
    test_samples = json.load(f)

for uid in valid_samples:
    valid_samples[uid]['-1']['inters'] = valid_samples[uid]['-1']['inters'][-mil:]
for uid in test_samples:
    test_samples[uid]['-1']['inters_full'] = list(test_samples[uid]['-1']['inters'])
    test_samples[uid]['-1']['inters'] = test_samples[uid]['-1']['inters'][-mil:]
   
    

# %%
with open(f"/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/{dataset_name}/users_1000.json",'r')as f:
    users = json.load(f)
    # users = users[:10] # for debug

safe_users = []
raw_count = len(users)
for uid in users:
    if uid in valid_samples and uid in test_samples:
        safe_users.append(uid)
    elif str(uid) in valid_samples and str(uid) in test_samples:
        safe_users.append(str(uid))
users = safe_users
# PPO load when not random selection
if (not args.greedy_selection) and (not args.random_selection) and (not args.all_types) and (not args.EP_IP_only) and (not args.EP_CF_only) and (not args.CF_IP_only):
    param_dict = torch.load(args.ppo_model_path)
    meta_refl_selector = Actor()
    meta_refl_selector.load_state_dict(param_dict)
    CF_model, CF_dataset, SAS_model, SAS_dataset = load_CF_SAS(args.dataset_name, gpu_id='0')


# %%
import numpy as np
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text


def get_metric(rec_list:list,label:str,metric='ndcg',top_k=10)->float:
    metrics = 0.0
    clean_label = clean_text(label)
    for ith,rec_item_title in enumerate(rec_list[:top_k]):
        if clean_label in clean_text(rec_item_title):
            if metric=='hit':
                metrics = 1  
            elif metric=='ndcg':
                metrics = 1/np.log2(ith+2) 
            else: 
                raise ValueError(f"metric:{metric} is not supported!")
            break
    return metrics


def get_ndcg(rec_list:list,label:str)->float:
    ndcg = 0.0
    clean_label = clean_text(label)
    for ith,rec_item_title in enumerate(rec_list):
        if clean_label in clean_text(rec_item_title):
            ndcg = 1/np.log2(ith+2)
            break
    return ndcg

def get_mean(li):
    return sum(li)/len(li)

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

# %%

from typing import List,Union


def llm_generate(model,tokenizer,dialogs:Union[List[List[dict]],List[dict]],**kwargs)->list:
    
    input_ids = tokenizer.apply_chat_template(
        dialogs,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict= True
    ).to(model.device)
    # print(input_ids['input_ids'].shape)
    
    kwargs["eos_token_id"] =  [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    output_ids = model.generate(
        **input_ids,
        **kwargs, pad_token_id=tokenizer.bos_token_id
        # max_new_tokens=256,
        # eos_token_id=terminators,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
    )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [output[output.find('assistant\n\n')+11:] for output in outputs]



def vllm_generate(model,tokenizer,dialogs:Union[List[List[dict]],List[dict]],**kwargs)->list:
    
    texts = tokenizer.apply_chat_template(
        dialogs,
        tokenize=False,
        add_generation_prompt=True
    )
    
    kwargs["stop_token_ids"] =  [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    sampling_params = SamplingParams(**kwargs)
    outputs = model.generate(
       texts,sampling_params
    )

    return [output.outputs[0].text  for output in outputs]

  

# %%

def get_item_title(item_id:str)->str:
    if item_id in item_info_dict:
        return item_info_dict[item_id]['title']
    return ""
def get_items_title(item_ids:list)->list:
    return [get_item_title(item_id) for item_id in item_ids]

# %%
def rec_inference_baseline(users,samples,bs=10, refl_data=None, refl_ndcg=None, baseline_name='SRMR_wo_reflect'):
    result_dict = {}
    metric_dict = defaultdict(dict)
    last_instruction_dict = {}
    

    dialogs,uids = [],[]
    for u_ith,uid in tqdm(enumerate(users)):
        # print("GT:",target_item_EP_dict[uid])
        inters = get_items_title(samples[uid]['-1']['inters'])
        if args.inters_num_prompt:
            inters = [f"{ith+1}.{item}" for ith,item in enumerate(inters)]
        candidate_set = get_items_title(samples[uid]['-1']['candidate_set'])
        if args.candidate_set_num_prompt:
            candidate_set = [f"{ith+1}.{item}" for ith,item in enumerate(candidate_set)]
        elif args.candidate_set_set_prompt:
            candidate_set = '{' + str(candidate_set)[1:-1] + '}'
            
        if refl_data==None:
            candidate_set_size = len(samples[uid]['-1']['candidate_set'])
            user_query = rec_prompts[baseline_name].format(inters=inters, 
                                    candidate_set=candidate_set)  
            user_query = user_query.replace("[[candidate_set_size]]",str(candidate_set_size))
        else:
            a = -1
            a_list = []
            if args.random_selection:
                a = random.randint(0,2)
            elif args.greedy_selection: 
                # greedy selection
                a = -1
                curr_ndcg_max = -10
                for i in range(len(refl_ndcg)):
                    if refl_ndcg[i][uid]>curr_ndcg_max:
                        a = i
                        curr_ndcg_max = refl_ndcg[i][uid]
            elif args.all_types: # refl_list: [EP, CF, IP] -> index[0, 1, 2]
                # use all types of reflects
                a_list = [0,1,2]
            elif args.EP_IP_only:
                # use EP and IP
                a_list = [0,2]
            elif args.EP_CF_only:
                # use EP and CF
                a_list = [0,1]
            elif args.CF_IP_only:
                # use CF and IP
                a_list = [1,2]
            else:
                # use ppo model as meta reflect selector
                s = generate_candidate_emb(user_id=uid, item_list=samples[uid]['-1']['inters_full'], SAS_model=SAS_model, CF_model=CF_model, SAS_dataset=SAS_dataset, CF_dataset=CF_dataset)
                s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).cuda()
                a_prob = meta_refl_selector(s).detach().cpu().numpy().flatten()
                a = np.argmax(a_prob)
            if baseline_name == "SRMR_w_reflect":
                if 0 <= a < 3 and uid in refl_data[a].keys(): # first: greedy, random and PPO, only select one type
                    candidate_set_size = len(samples[uid]['-1']['candidate_set'])
                    reflect = refl_data[a][uid]['fs']
                    if args.suggestion_only:
                        index = reflect.find("SUGGESTIONS:")
                        if index != -1:
                            reflect = reflect[index:]
                        else:
                            index = reflect.find("Suggestions:")
                            if index != -1:
                                reflect = reflect[index:]
                    user_query = rec_prompts[baseline_name+f'_v{args.SRMR_w_reflect_version}'].format(inters=inters, 
                                    candidate_set=candidate_set,
                                    reflect=reflect)
                    user_query = user_query.replace("[[candidate_set_size]]",str(candidate_set_size))
                # if a == -1 and types > 1:   NOTE: when types=2, 3 need uid has all these types's gold_fs_info not empty
                elif a_list != [] and len([a for a in a_list if uid in refl_data[a].keys()]) == len(a_list):
                    reflect = ""
                    for a in a_list:
                        reflect += refl_data[a][uid]['fs']
                    candidate_set_size = len(samples[uid]['-1']['candidate_set'])
                    user_query = rec_prompts[baseline_name+f'_v{args.SRMR_w_reflect_version}'].format(inters=inters, 
                                    candidate_set=candidate_set,
                                    reflect=reflect)
                    user_query = user_query.replace("[[candidate_set_size]]",str(candidate_set_size))
                # a == 3 or uid do not match any gold_fs_info, use no example and no reflect prompt
                else:
                    # set to be no reflects
                    candidate_set_size = len(samples[uid]['-1']['candidate_set'])
                    user_query = rec_prompts["SRMR_wo_reflect"].format(inters=inters, 
                                    candidate_set=candidate_set).replace("[[example]]", '')
                    user_query = user_query.replace("[[candidate_set_size]]",str(candidate_set_size))
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
        if args.vllm:
            outputs = vllm_generate(model, tokenizer, dialog_batch,      
                                max_tokens=512, 
                                n=1,
                                temperature=0.1,
                                top_p=0.9,
                                )
        else:
            outputs = llm_generate(model, tokenizer, dialog_batch,     
                                max_new_tokens=512, 
                                num_return_sequences=1,
                                do_sample=False,
                                temperature=0.1,
                                top_p=0.9,
                                )
        
        # print(outputs)
        # raise ValueError("stop here!!")
    
        for response,uid in zip(outputs,uid_batch):
            rec_list = rec_str_2_list(response) 
            
            cand_ids = samples[uid]['-1']['candidate_set']
            cand_title_map = {clean_text(get_item_title(c)): get_item_title(c) for c in cand_ids}
            rec_list = [r for r in rec_list if clean_text(r) in cand_title_map]
            if len(rec_list) < 10:
                used = set(clean_text(r) for r in rec_list)
                rest = [t for ct, t in cand_title_map.items() if ct not in used]
                his_texts = [clean_text(get_item_title(i)) for i in samples[uid]['-1']['inters_full']]
                rest.sort(key=lambda t: max((_lcs_len(h, clean_text(t)) for h in his_texts), default=0),
                          reverse=True)
                rec_list = rec_list + rest[:10 - len(rec_list)]

            result_dict[uid] = rec_list
            metric_dict['ndcg@5'][uid] = get_metric(rec_list, 
                                          label=get_item_title(samples[uid]['-1']['target_item']),
                                          metric='ndcg',top_k=5)
            metric_dict['ndcg@10'][uid] = get_metric(rec_list, 
                                          label=get_item_title(samples[uid]['-1']['target_item']),
                                          metric='ndcg',top_k=10)   
            
            metric_dict['hit@1'][uid] = get_metric(rec_list, 
                                          label=get_item_title(samples[uid]['-1']['target_item']),
                                          metric='hit',top_k=1)
            metric_dict['hit@5'][uid] = get_metric(rec_list, 
                                          label=get_item_title(samples[uid]['-1']['target_item']),
                                          metric='hit',top_k=5) 
            metric_dict['hit@10'][uid] = get_metric(rec_list, 
                                          label=get_item_title(samples[uid]['-1']['target_item']),
                                          metric='hit',top_k=10)         
            # get_metric(rec_list:list,label:str,metric='ndcg',top_k=10)->float:
    
    return result_dict,metric_dict,last_instruction_dict

def compute_metrics(result_dict, samples):
    metric_dict = defaultdict(dict)
    for uid, rec_list in result_dict.items():
        metric_dict['ndcg@5'][uid] = get_metric(rec_list, 
                                      label=get_item_title(samples[uid]['-1']['target_item']),
                                      metric='ndcg',top_k=5)
        metric_dict['ndcg@10'][uid] = get_metric(rec_list, 
                                      label=get_item_title(samples[uid]['-1']['target_item']),
                                      metric='ndcg',top_k=10)   
        metric_dict['hit@1'][uid] = get_metric(rec_list, 
                                      label=get_item_title(samples[uid]['-1']['target_item']),
                                      metric='hit',top_k=1)
        metric_dict['hit@5'][uid] = get_metric(rec_list, 
                                      label=get_item_title(samples[uid]['-1']['target_item']),
                                      metric='hit',top_k=5) 
        metric_dict['hit@10'][uid] = get_metric(rec_list, 
                                      label=get_item_title(samples[uid]['-1']['target_item']),
                                      metric='hit',top_k=10)         
    return metric_dict

def fuse_with_refl(test_base_result_dict, refl_result_dict, users, samples):
    final_result_dict = {}
    for uid in users:
        if uid not in test_base_result_dict or uid not in samples:
            final_result_dict[uid] = refl_result_dict.get(uid, test_base_result_dict.get(uid, []))[:10]
            continue
        his_texts = [clean_text(get_item_title(i)) for i in samples[uid]['-1']['inters_full']]
        merged, seen = [], set()
        for r in list(test_base_result_dict[uid][:10]) + list(refl_result_dict.get(uid, [])[:10]):
            c = clean_text(r)
            if c and c not in seen:
                seen.add(c)
                merged.append(r)
        if len(merged) < 10:
            cand_ids = samples[uid]['-1']['candidate_set']
            cand_title_map = {clean_text(get_item_title(c)): get_item_title(c) for c in cand_ids}
            used = set(clean_text(r) for r in merged)
            rest = [t for ct, t in cand_title_map.items() if ct not in used]
            rest.sort(key=lambda t: max((_lcs_len(h, clean_text(t)) for h in his_texts), default=0),
                      reverse=True)
            merged = merged + rest[:10 - len(merged)]
        final_result_dict[uid] = sorted(
            merged, key=lambda t: max((_lcs_len(h, clean_text(t)) for h in his_texts), default=0),
            reverse=True)[:10]
    final_metric_dict = compute_metrics(final_result_dict, samples)
    return final_result_dict, final_metric_dict


# key = a*(lcs/25) + b*sem_norm + c*(1/rank_off), a=(1-c)*alpha, b=(1-c)*(1-alpha)
SEM_RERANK = {
    'Amazon_Games':       dict(alpha=0.1, c=0.4),
    'Amazon_Arts':        dict(alpha=0.6, c=0.0),
    'Amazon_Instruments': dict(alpha=0.4, c=0.2),
}

def build_sem_index(users, samples):
    involved = set()
    for uid in users:
        if uid not in samples:
            continue
        d = samples[uid]['-1']
        involved.update(str(i) for i in d['candidate_set'])
        involved.update(str(i) for i in d['inters_full'])
    info_by_id = {str(k): v for k, v in item_info_dict.items()}
    docs, iid2row = [], {}
    for iid in involved:
        info = info_by_id.get(iid)
        if info is None:
            continue
        title = info['title'] if isinstance(info['title'], str) else ''
        cats = info['categories'] if isinstance(info['categories'], str) else ''
        brand = info['brand'] if isinstance(info['brand'], str) else ''
        iid2row[iid] = len(docs)
        docs.append(f"{title} {cats} {brand}")
    X = TfidfVectorizer(lowercase=True).fit_transform(docs)
    return X, iid2row

def sem_rerank(rec_dict, users, samples, X, iid2row):
    cfg = SEM_RERANK.get(dataset_name)
    if cfg is None:
        return rec_dict
    c = cfg['c']
    a = (1 - c) * cfg['alpha']
    b = (1 - c) * (1 - cfg['alpha'])
    out = {}
    for uid in users:
        rec = rec_dict.get(uid)
        if not rec or uid not in samples:
            out[uid] = rec
            continue
        d = samples[uid]['-1']
        rows = [iid2row[str(i)] for i in d['inters_full'] if str(i) in iid2row]
        if rows:
            centroid = np.asarray(X[rows].mean(axis=0)).ravel()
            cen_norm = float(np.linalg.norm(centroid))
        else:
            centroid, cen_norm = None, 0.0
        his_texts = [clean_text(get_item_title(i)) for i in d['inters_full']]
        ctitle2iid = {clean_text(get_item_title(cid)): str(cid) for cid in d['candidate_set']}
        sem = {}
        for t in rec:
            ct = clean_text(t)
            iid = ctitle2iid.get(ct)
            if iid is not None and iid in iid2row and cen_norm > 0:
                sem[ct] = float(np.asarray(X[iid2row[iid]].dot(centroid)).squeeze()) / cen_norm
            else:
                sem[ct] = 0.0
        vals = list(sem.values())
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = (hi - lo) or 1.0
        keys = []
        for rank_off, t in enumerate(rec, start=1):
            ct = clean_text(t)
            lcs = max((_lcs_len(h, ct) for h in his_texts), default=0) if his_texts else 0
            keys.append(a * (lcs / 25.0) + b * ((sem[ct] - lo) / span) + c * (1.0 / rank_off))
        out[uid] = [t for _, t in sorted(zip(keys, rec), key=lambda x: x[0], reverse=True)][:10]
    return out

    

# %%
# with open(os.path.join(args.refl_result_dir,f"iter_{args.iter_i}.json"),"r")as f:
#     result_data = json.load(f)
# refl_data = result_data['gold_fs_info']
with open(os.path.join(args.CF_refl_path,f"iter_{args.iter_i}.json"),"r")as f:
    CF_result_data = json.load(f)
CF_refl_data = CF_result_data['gold_fs_info']
with open(os.path.join(args.IP_refl_path,f"iter_{args.iter_i}.json"),"r")as f:
    IP_result_data = json.load(f)
IP_refl_data = IP_result_data['gold_fs_info']
with open(os.path.join(args.EP_refl_path,f"iter_{args.iter_i}.json"),"r")as f:
    EP_result_data = json.load(f)
EP_refl_data = EP_result_data['gold_fs_info']
refl_data = [EP_refl_data, CF_refl_data, IP_refl_data]
refl_ndcg = None
if args.greedy_selection: # when greedy, need load all refl ndcg value
    with open(os.path.join(args.CF_refl_path,f"iter_{args.iter_i}.json"),"r")as f:
        CF_result_data = json.load(f)
    CF_refl_ndcg = CF_result_data['valid_refl1_ndcg_dict']
    with open(os.path.join(args.IP_refl_path,f"iter_{args.iter_i}.json"),"r")as f:
        IP_result_data = json.load(f)
    IP_refl_ndcg = IP_result_data['valid_refl1_ndcg_dict']
    with open(os.path.join(args.EP_refl_path,f"iter_{args.iter_i}.json"),"r")as f:
        EP_result_data = json.load(f)
    EP_refl_ndcg = EP_result_data['valid_refl1_ndcg_dict']
    refl_ndcg = [EP_refl_ndcg, CF_refl_ndcg, IP_refl_ndcg]
if "reflect" not in baseline_name:
    test_base_result_dict, \
        test_base_metric_dict, \
        test_last_instruction_dict = rec_inference_baseline(users,test_samples,bs=bs,baseline_name=baseline_name)
    final_result_dict = test_base_result_dict
    final_metric_dict = test_base_metric_dict
else:
    test_base_result_dict, \
        test_base_metric_dict, \
        test_last_instruction_dict = rec_inference_baseline(users,test_samples,bs=bs,baseline_name='SRMR_wo_reflect')
    refl_result_dict, \
        _, \
        _ = rec_inference_baseline(users,test_samples,bs=bs,baseline_name=baseline_name,refl_data=refl_data, refl_ndcg=refl_ndcg)
    final_result_dict, final_metric_dict = fuse_with_refl(test_base_result_dict, refl_result_dict, users, test_samples)
    sem_X, sem_iid2row = build_sem_index(users, test_samples)
    final_result_dict = sem_rerank(final_result_dict, users, test_samples, sem_X, sem_iid2row)
    final_metric_dict = compute_metrics(final_result_dict, test_samples)
print()
print('='*10)
for metric in final_metric_dict:
    print(metric,'\t',get_mean(final_metric_dict[metric].values())) 
print('='*10)
print()
os.makedirs("result",exist_ok=True)
os.makedirs(f"result/{args.ex}_{dataset_name}",exist_ok=True)
# save baseline result
with open(f"result/{args.ex}_{dataset_name}/test_base_result_dict.json","w")as f:
    json.dump(test_base_result_dict,f)
with open(f"result/{args.ex}_{dataset_name}/test_base_metric_dict.json","w")as f:
    json.dump(test_base_metric_dict,f)
# save fused final result
with open(f"result/{args.ex}_{dataset_name}/test_gated_result_dict.json","w")as f:
    json.dump(final_result_dict,f)
with open(f"result/{args.ex}_{dataset_name}/test_gated_metric_dict.json","w")as f:
    json.dump(final_metric_dict,f)
if "reflect" in baseline_name:
    len_cnt = {}
    for v in final_result_dict.values():
        len_cnt[len(v)] = len_cnt.get(len(v), 0) + 1
with open(f"result/{args.ex}_{dataset_name}/test_last_instruction_dict.json","w")as f:
    json.dump(test_last_instruction_dict,f)
