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
seq_lengths = [8, 16, 32, 64, 128, 256, 512, 1024]

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
    payload = "机器学习是人工智能的一个分支。人工智能的研究历史有着一条从以“推理”为重点，到以“知识”为重点，再到以“学习”为重点的自然、清晰的脉络。显然，机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。机器学习在近30多年已发展为一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析（英语：Convex analysis）、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。很多推论问题属于无程序可循难度，所以部分的机器学习研究是开发容易处理的近似算法。机器学习已广泛应用于数据挖掘、计算机视觉、自然语言处理、生物特征识别、搜索引擎、医学诊断、检测信用卡欺诈、证券市场分析、DNA序列测序、语音和手写识别、战略游戏和机器人等领域。"
    len_pay = tokenizer(payload, return_tensors="pt")["input_ids"][0].size(0) - 1
    payload = tokenizer(payload, return_tensors="pt")["input_ids"][0][1:-2]
    if (seq_len - 1) // len_pay > 0:
        payload = torch.cat((payload.tile((((seq_len - 2) // len_pay), )), payload[:((seq_len - 2) % len_pay)]))
    else:
        payload = payload[:((seq_len - 2) % len_pay)]
    payload = tokenizer.decode(payload)
    payload = tokenizer(payload, return_tensors="pt")
    payload = {key: val.to(device) for key, val in payload.items()}
    latencies = []
    latencies_per_token = []
    # Warm up
    for _ in range(10):
        _ = model.generate(
            **payload,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=num_beam,
            max_length=1200,
        )

    # Timed run
    for _ in range(iterations):
        start_time = perf_counter()
        generated_tokens = model.generate(
            **payload,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=num_beam,
            max_length=1200,
        )
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
# model_id = "facebook/m2m100_1.2B"
model_id = "facebook/m2m100_418M"

device = torch.device("cuda:0")
pt_model, tokenizer = get_transformer_model(model_id)
pt_model.to(device)
onnx_model, _ = get_onnx_model(model_id, "results/m2m100_418M")
onnx_model.to(device)

# Benchmark
res = []
for seq_len in seq_lengths:
    print("seq_len: ", seq_len)
    pt = benchmark(seq_len, pt_model, tokenizer, device, num_beam=1, iterations=500)
    res.append({**pt, "model": "pt"})

    v_onnx = benchmark(seq_len, onnx_model, tokenizer, device, num_beam=1, iterations=500)
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
plt.figure.savefig("t4_res_iobinding_m2m_418m_greedy.png", dpi=900)

print(chart_df.head(10))
chart_df.to_csv("t4_res_iobinding_m2m_418m_greedy.csv")