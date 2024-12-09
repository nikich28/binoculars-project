import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(tokenized, performer_pred):
    shifted_labels = tokenized.input_ids[..., 1:]
    shifted_attention_mask = tokenized.attention_mask[..., 1:]
    shifted_logits = performer_pred[..., :-1, :]

    ce = F.cross_entropy(shifted_logits.transpose(1, 2), shifted_labels, reduction="none")

    perp = (ce * shifted_attention_mask).sum(axis=1) / shifted_attention_mask.sum(axis=1)
    perp = perp.detach().cpu().float().numpy()

    return perp


def compute_entropy(tokenized, observer_pred, performer_pred, pad_token):
    vocab_size = observer_pred.shape[-1]
    total_tokens_available = performer_pred.shape[-2]

    perf_scores = performer_pred.view(-1, vocab_size)
    obs_scores = F.softmax(observer_pred, dim=-1).view(-1, vocab_size)

    entropy = F.cross_entropy(input=perf_scores, target=obs_scores, reduction="none").view(-1, total_tokens_available)
    padding_mask = (tokenized.input_ids != pad_token).type(torch.uint8)


    ce = (entropy * padding_mask).sum(axis=1) / padding_mask.sum(axis=1)          
    ce = ce.detach().cpu().float().numpy()

    return ce



class Binoculars(nn.Module):
    def __init__(self, observer_path, performer_path, fp16=True, metric="acc", max_token_length=256):
        super().__init__()
        """
        observer_path, performer_path - path for huggingface backbone models
        fp16 - use float16, default True
        metric - 'acc' for accuracy (optimized for f1), 'frp' for low-fpr
        max_token_length - maximum tokens length for tokenizer
        """
        self.threshold = 0.9015310749276843 # from the original repo
        if metric == "fpr":
            self.threshold = 0.8536432310785527

        self.max_length = max_token_length

        if fp16:
            self.precision = torch.bfloat16
        else:
            self.precision = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(observer_path)

        self.observer = AutoModelForCausalLM.from_pretrained(observer_path, device_map="cuda", torch_dtype=self.precision,
                                                              trust_remote_code=True)

        self.performer = AutoModelForCausalLM.from_pretrained(performer_path, device_map="cuda", torch_dtype=self.precision,
                                                              trust_remote_code=True)

        self.observer.eval()
        self.performer.eval()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def forward(self, inputs):
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length,
            return_token_type_ids=False).to("cuda")

        observer_pred = self.observer(**tokenized).logits
        performer_pred = self.performer(**tokenized).logits

        perpl = compute_perplexity(tokenized, performer_pred)
        entropy = compute_entropy(tokenized, observer_pred, performer_pred, self.tokenizer.pad_token_id)

        return perpl / entropy

    def predict(self, inputs):
        metric = self.forward(inputs)
        preds = np.where(metric >= self.threshold, "Human", "Machine")
        return preds.tolist()
