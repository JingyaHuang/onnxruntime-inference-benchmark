# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from optimum.onnxruntime import ORTModelForCausalLM


SEED = 42

logging.basicConfig(level=logging.INFO)

# Instanciate PyTorch model
def get_transformer_model(model_checkpoint):
    set_seed(SEED)
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (model, tokenizer)


# Instanciate ONNX model w/. and w/o. graph optimization
def get_onnx_model(model_checkpoint):
    model = ORTModelForCausalLM.from_pretrained(model_checkpoint, from_transformers=True, use_io_binding=False, provider="CUDAExecutionProvider")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (model, tokenizer)


def benchmark(seq_len, model, tokenizer, device, frame, iterations=200, num_beam=1, model_id="bigscience/bloom-560m"):
    # prepare data
    payload = {
        "input_ids": torch.randint(low=0, high=tokenizer.vocab_size - 1, size=(seq_len,), dtype=torch.int64).unsqueeze(0).to(device),
        "attention_mask": torch.ones((seq_len,), dtype=torch.int64).unsqueeze(0).to(device),
    }
    payload = {key: val.to(device) for key, val in payload.items()}
    latencies_per_seq = []
    num_gen_tokens = []
    latencies_per_token = []
    max_length = int(seq_len * 1.5)
    min_length = seq_len // 2
    # Warm up
    for _ in range(10):
        _ = model.generate(
            **payload,
            pad_token_id=tokenizer.eos_token_id,
            min_length=min_length,
            max_length=max_length,
        )

    # Timed run
    for _ in tqdm(range(iterations)):
        start_time = perf_counter()
        generated_tokens = model.generate(
            **payload,
            num_beams=num_beam,
            min_length=min_length,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        latency = 1000 * (perf_counter() - start_time)  # Unit: ms
        num_tokens = generated_tokens.size(1)
        latency_per_token = latency / num_tokens
        latencies_per_seq.append(latency)
        num_gen_tokens.append(num_tokens)
        latencies_per_token.append(latency_per_token)

    # Compute run statistics
    time_avg_ms_per_seq = np.mean(latencies_per_seq)
    time_avg_ms_per_token = np.mean(latencies_per_token)
    time_p95_ms_per_seq = np.percentile(latencies_per_seq, 95)
    time_p95_ms_per_token = np.percentile(latencies_per_token, 95)

    # Record statistics for each iteration
    stat_dict = {
        "time_ms_per_seq": latencies_per_seq,
        "num_gen_tokens": num_gen_tokens,
        "time_ms_per_token": latencies_per_token,
    }
    df = pd.DataFrame.from_dict(stat_dict)
    df["num_iter"] = iterations
    df["seq_len"] = payload["input_ids"].shape[1]
    df["model_id"] = model_id
    df["framework"] = frame
    df["num_beam"] = num_beam
    df["time_avg_ms_per_seq"] = time_avg_ms_per_seq
    df["time_avg_ms_per_token"] = time_avg_ms_per_token
    df["time_p95_ms_per_seq"] = time_p95_ms_per_seq
    df["time_p95_ms_per_token"] = time_p95_ms_per_token

    return df

# Run benchmark one by one
model_id = "bigscience/bloom-560m"
# model_id = "gpt2"

device = torch.device("cuda:0")
pt_model, tokenizer = get_transformer_model(model_id)
# pt_model.to(device)
onnx_model, _ = get_onnx_model(model_id)
# onnx_model.to(device)

# Benchmark
seq_lengths = [8] #, 16, 32, 64, 128, 256, 512
num_beams = [5]
for seq_len in seq_lengths:
    print("seq_len: ", seq_len)
    res = []
    for num_beam in num_beams:
        print("num_beam: ", num_beam)
        df_onnx = benchmark(seq_len, onnx_model, tokenizer, device, frame="ONNX", num_beam=num_beam, iterations=500, model_id=model_id)
        res.append(df_onnx)
        df_pt = benchmark(seq_len, pt_model, tokenizer, device, frame="PyTorch", num_beam=num_beam, iterations=500, model_id=model_id)
        res.append(df_pt)



    res = pd.concat(res, ignore_index=True)
    res.to_pickle(f"t4_res_ort_bloom_beam5_{seq_len}.pkl")


# # Save result
# res = pd.concat(res, ignore_index=True)
# res.to_pickle("t4_res_ort_gpt2_greedy.pkl")

# pd.read_pickle('t4_res_ort_gpt2_greedy.pkl').head(10)
