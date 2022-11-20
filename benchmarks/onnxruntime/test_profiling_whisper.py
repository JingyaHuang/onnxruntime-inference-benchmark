# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

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

# Export vanilla & optimized onnx model
def export_vanilla_optimized_onnx(model_checkpoint):
    set_seed(SEED)
    processor = AutoProcessor.from_pretrained(model_checkpoint)

    # Vanilla
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_checkpoint, from_transformers=True, use_cache=True)
    onnx_path = Path(os.path.join("results/", "whisper-tiny"))
    model.save_pretrained(onnx_path)
    processor.save_pretrained(onnx_path)

# # Export Whisper ONNX models
# model_checkpoint = "openai/whisper-tiny.en"
# export_vanilla_optimized_onnx(model_checkpoint)

# Instanciate PyTorch model
def get_transformer_model(model_checkpoint):
    set_seed(SEED)
    device = torch.device("cuda:0")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_checkpoint).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    return (model, processor)

# Instanciate ONNX model w/. and w/o. graph optimization
def get_onnx_model(model_checkpoint, onnx_path):
    config = AutoConfig.from_pretrained(model_checkpoint, use_cache=True)
    device = torch.device("cuda:0")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_id=onnx_path,
        encoder_file_name="encoder_model.onnx",
        decoder_file_name="decoder_model.onnx",
        decoder_with_past_file_name="decoder_with_past_model.onnx",
        config=config,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    return (model, processor)

# Prepare data
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

def benchmark(model, processor, device, iterations=200, num_beam=1):
    # prepare data
    payload = ds[0]["audio"]["array"]
    payload = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    payload = {key: val.to(device) for key, val in payload.items()}
    latencies = []
    latencies_per_token = []
    # Warm up
    for _ in range(10):
        _ = model.generate(**payload, num_beams=num_beam)

    # Timed run
    for _ in range(iterations):
        start_time = perf_counter()
        generated_tokens = model.generate(**payload, num_beams=num_beam)
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
        "seq_len": payload["input_features"].shape[2],
        "time_avg_ms": time_avg_ms,
        "time_avg_ms_per_token": time_avg_ms_per_token,
        "time_p95_ms": time_p95_ms,
        "time_p95_ms_per_token": time_p95_ms_per_token,
    }

model_id = "openai/whisper-tiny.en"

device = torch.device("cuda:0")
pt_model, tokenizer = get_transformer_model(model_id)
pt_model.to(device)
onnx_model, _ = get_onnx_model(model_id, "results/whisper-tiny")
onnx_model.to(device)

# Benchmark
res = []
pt = benchmark(pt_model, tokenizer, device, num_beam=5, iterations=500)
res.append({**pt, "model": "pt"})

v_onnx = benchmark(onnx_model, tokenizer, device, num_beam=5, iterations=500)
res.append({**v_onnx, "model": "v_onnx"})

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


chart_df["io_improvement/pt(token)"] = f"{round((chart_df['pt_token_p95'] - chart_df['v_onnx_token_p95']) / chart_df['pt_token_p95'] * 100,2)}%"


plt = chart_df.plot(x="seq_len", y=["pt_token_p95", "v_onnx_token_p95"], kind="line")
plt.figure.savefig("t4_res_iobinding_whisper_tiny_beam5.png", dpi=900)

print(chart_df.head(10))
chart_df.to_csv("t4_res_iobinding_whisper_tiny_beam5.csv")





