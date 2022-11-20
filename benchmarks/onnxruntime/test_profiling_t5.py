# -*- coding: utf-8 -*-
import os
import logging
from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor
)

from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForSpeechSeq2Seq, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


SEED = 42

logging.basicConfig(level=logging.INFO)

# All candidate checkpoints
model_checkpoints = ["facebook/m2m100_418M", "t5-small"] #"facebook/m2m100_1.2B", "t5-large"
onnx_paths = [Path(os.path.join("results/", model_checkpoint.split("/")[-1])) for model_checkpoint in model_checkpoints]
seq_lengths = [8, 16, 32, 64, 128, 256, 512]

# Export vanilla & optimized onnx model
def export_vanilla_optimized_onnx(model_checkpoint):
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # # Vanilla
    model = ORTModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_transformers=True)
    onnx_path = Path(os.path.join("results/", model_checkpoint.split("/")[-1]))
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)

    # Optimized
    # Graph optimization
    optimizer = ORTOptimizer.from_pretrained(model)
    onnx_path = Path(os.path.join("results/", model_checkpoint.split("/")[-1] + "_optimized"))
    optimization_config = OptimizationConfig(
        optimization_level=1,
        optimize_for_gpu=True,
        fp16=True,
    )
    optimizer.optimize(save_dir=onnx_path, optimization_config=optimization_config)

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
    is_optimized = "optimized" in str(onnx_path)
    device = torch.device("cuda:0")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_id=onnx_path,
        encoder_file_name="encoder_model_optimized.onnx" if is_optimized else "encoder_model.onnx",
        decoder_file_name="decoder_model_optimized.onnx" if is_optimized else "decoder_model.onnx",
        decoder_with_past_file_name="decoder_with_past_model_optimized.onnx" if is_optimized else "decoder_with_past_model.onnx",
        config=config,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (model, tokenizer)


def benchmark(seq_len, model, tokenizer, device, iterations=200, num_beam=1):
    # prepare data
    seq_len = torch.randint(
        low=0, high=tokenizer.vocab_size, size=(seq_len-1, ), dtype=torch.int32, device="cuda"
    )
    payload = tokenizer.decode(seq_len)
    payload = tokenizer(payload, return_tensors="pt")
    payload = {key: val.to(device) for key, val in payload.items()}
    latencies = []
    latencies_per_token = []
    # Warm up
    for _ in range(10):
        _ = model.generate(
            **payload,
        )

    # Timed run
    for _ in range(iterations):
        start_time = perf_counter()
        generated_tokens = model.generate(**payload, num_beams=num_beam, max_length=1200)
        latency = perf_counter() - start_time
        num_tokens = generated_tokens.size(1)
        latency_per_token = latency / num_tokens
        latencies.append(latency)
        latencies_per_token.append(latency_per_token)

    # Compute run statistics

    time_avg_ms = 1000 * np.mean(latencies)
    time_avg_ms_per_token = 1000 * np.mean(latencies_per_token)
    time_p95_ms = 1000 * np.percentile(latencies, 95)
    time_p95_ms_per_token = 1000 * np.percentile(latencies_per_token, 95)
    return {
        "seq_len": payload["input_ids"].shape[1],
        "time_avg_ms": time_avg_ms,
        "time_avg_ms_per_token": time_avg_ms_per_token,
        "time_p95_ms": time_p95_ms,
        "time_p95_ms_per_token": time_p95_ms_per_token,
    }




# # Export all candidate models
# for model_checkpoint in model_checkpoints:
#     export_vanilla_optimized_onnx(model_checkpoint)

# Run benchmark one by one
model_id = "t5-small"  # max_seq_len=512
# model_id = "facebook/m2m100_418M"

device = torch.device("cuda:0")
pt_model, tokenizer = get_transformer_model(model_id)
pt_model.to(device)
onnx_model, _ = get_onnx_model(model_id, "results/t5-small")
onnx_model.to(device)
# optim_model, _ = get_onnx_model(model_id, "results/t5-small_optimized")
# optim_model.to(device)

# Benchmark
res = []
for seq_len in seq_lengths:
    print("seq_len: ", seq_len)
    pt = benchmark(seq_len, pt_model, tokenizer, device, num_beam=5, iterations=500)
    res.append({**pt, "model": "pt"})

    v_onnx = benchmark(seq_len, onnx_model, tokenizer, device, num_beam=5, iterations=500)
    res.append({**v_onnx, "model": "v_onnx"})

    # optim_onnx = benchmark(seq_len, optim_model, tokenizer, device, iterations=500)
    # res.append({**optim_onnx, "model": "optim_onnx"})

df = pd.DataFrame(res)
print(df)

chart_df = pd.merge(
    df[df.model == "pt"][["seq_len", "time_p95_ms", "time_p95_ms_per_token"]],
    df[df.model == "v_onnx"][["seq_len", "time_p95_ms", "time_p95_ms_per_token"]],
    on="seq_len",
)
chart_df = chart_df.rename(
    columns={
        "time_p95_ms_x": "pt_seq_p95",
        "time_p95_ms_y": "v_onnx_seq_p95",
        "time_p95_ms_per_token_x": "pt_token_p95",
        "time_p95_ms_per_token_y": "v_onnx_token_p95",
    }
)
# chart_df = pd.merge(
#     chart_df,
#     df[df.model == "optim_onnx"][["seq_len", "time_p95_ms", "time_p95_ms_per_token"]],
#     on="seq_len",
# )
# chart_df = chart_df.rename(
#     columns={
#         "time_p95_ms": "optim_onnx_seq_p95",
#         "time_p95_ms_per_token": "optim_onnx_token_p95",
#     }
# )


chart_df["io_improvement/pt(token)"] = f"{round((chart_df['pt_token_p95'] - chart_df['v_onnx_token_p95']) / chart_df['pt_token_p95'] * 100,2)}%"
# chart_df["io+optim/pt"] = f"{round((chart_df['pt_token_p95'] - chart_df['optim_onnx_token_p95']) / chart_df['pt_token_p95'] * 100,2)}%"

plt = chart_df.plot(x="seq_len", y=["pt_token_p95", "v_onnx_token_p95"], kind="line")
plt.figure.savefig("t4_res_iobinding_t5_s_beam5.png", dpi=900)

print(chart_df.head(10))
chart_df.to_csv("t4_res_iobinding_t5_s_beam5.csv")






