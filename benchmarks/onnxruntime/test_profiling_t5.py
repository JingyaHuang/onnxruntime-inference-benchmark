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
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

from optimum.onnxruntime import ORTModelForSeq2SeqLM


SEED = 42

logging.basicConfig(level=logging.INFO)

# Instanciate PyTorch model
def get_transformer_model(model_checkpoint):
    set_seed(SEED)
    device = torch.device("cuda:0")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (model, tokenizer)


# Instanciate ONNX model w/. and w/o. graph optimization
def get_onnx_model(model_checkpoint, onnx_path):
    config = AutoConfig.from_pretrained(model_checkpoint, use_cache=True)
    device = torch.device("cuda:0")
    # model = ORTModelForSeq2SeqLM.from_pretrained(
    #     model_id=onnx_path,
    #     encoder_file_name="encoder_model.onnx",
    #     decoder_file_name="decoder_model.onnx",
    #     decoder_with_past_file_name="decoder_with_past_model.onnx",
    #     config=config,
    # ).to(device)
    model = ORTModelForSeq2SeqLM.from_pretrained("optimum/t5-small").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (model, tokenizer)


def benchmark(seq_len, model, tokenizer, device, frame, iterations=200, num_beam=1, model_id="t5-small"):
    # prepare data
    payload = {
        "input_ids": torch.cat((torch.randint(low=0, high=tokenizer.vocab_size, size=(seq_len - 1,), dtype=torch.int32), torch.tensor([tokenizer.eos_token_id]))).unsqueeze(0).to(device),
        "attention_mask": torch.ones((seq_len,), dtype=torch.int32).unsqueeze(0).to(device),
    }
    latencies_per_seq = []
    num_gen_tokens = []
    latencies_per_token = []
    # Warm up
    for _ in range(10):
        _ = model.generate(
            **payload,
            max_length=1200,
        )

    # Timed run
    max_length = int(seq_len * 1.5)
    min_length = seq_len // 2
    for _ in tqdm(range(iterations)):
        start_time = perf_counter()
        generated_tokens = model.generate(
            **payload,
            num_beams=num_beam,
            min_length=min_length,
            max_length=max_length,
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
model_id = "t5-small"  # max_seq_len=512

device = torch.device("cuda:0")
pt_model, tokenizer = get_transformer_model(model_id)
pt_model.to(device)
onnx_model, _ = get_onnx_model(model_id, "results/t5-small")
onnx_model.to(device)

# Benchmark
res = []
seq_lengths = [8, 16, 32, 64, 128, 256, 512]
num_beams = [1]
for seq_len in seq_lengths:
    print("seq_len: ", seq_len)
    for num_beam in num_beams:
        print("num_beam: ", num_beam)
        df_pt = benchmark(seq_len, pt_model, tokenizer, device, frame="pt", num_beam=num_beam, iterations=500)
        res.append(df_pt)

        df_onnx = benchmark(seq_len, onnx_model, tokenizer, device, frame="onnx", num_beam=num_beam, iterations=500)
        res.append(df_onnx)

# Save result
res = pd.concat(res, ignore_index=True)
res.to_pickle("t4_res_ort_t5_s_greedy.pkl")

pd.read_pickle('t4_res_ort_t5_s_greedy.pkl').head(10)



