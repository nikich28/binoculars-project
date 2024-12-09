import torch
import pandas as pd
from datasets import Dataset, logging as datasets_logging
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import gc
from model import Binoculars

torch.cuda.empty_cache(); gc.collect()


# bino = Binoculars("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", max_token_length=128)
bino = Binoculars("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", max_token_length=128)

# Load dataset
ds = Dataset.from_json("./cnn-llama2_13.jsonl")

for d in tqdm(ds):
    d["meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"] = d["meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"]
    d['article'] = d['article']

# Set keys
machine_sample_key = "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"
machine_text_source = "LLaMA-2-13B"


print(f"Scoring human text")

human_scores = []
batch = []
for i in tqdm(range(1000)):
    batch.append(ds[i]['article'])
    if len(batch) == 4:
        human_scores.extend(bino.forward(batch))
        batch = []
        torch.cuda.empty_cache(); gc.collect()


print(f"Scoring machine text")

machine_scores = []
batch = []
for i in tqdm(range(1000)):
    batch.append(ds[i][machine_sample_key])
    if len(batch) == 4:
        machine_scores.extend(bino.forward(batch))
        batch = []
        torch.cuda.empty_cache(); gc.collect()

score_df = pd.DataFrame(
    {"score": human_scores + machine_scores, "class": [0] * len(human_scores) + [1] * len(machine_scores)}
)
#default threshold from paper
score_df["pred"] = np.where(score_df["score"] < 0.9015310749276843, 1, 0)

score_df.to_csv("score_df.csv")

# Compute metrics
f1_score = metrics.f1_score(score_df["class"], score_df["pred"])

print(f1_score)
