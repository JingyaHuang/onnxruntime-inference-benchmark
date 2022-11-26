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
def get_onnx_model(model_checkpoint):
    device = torch.device("cuda:0")
    model = ORTModelForSeq2SeqLM.from_pretrained("optimum/m2m100_418M").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (model, tokenizer)


def benchmark(seq_len, model, tokenizer, device, frame, iterations=200, num_beam=1, model_id="m2m100_418m"):
    # prepare data
    payload = "机器学习是人工智能的一个分支。人工智能的研究历史有着一条从以“推理”为重点，到以“知识”为重点，再到以“学习”为重点的自然、清晰的脉络。显然，机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。机器学习在近30多年已发展为一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析（英语：Convex analysis）、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。很多推论问题属于无程序可循难度，所以部分的机器学习研究是开发容易处理的近似算法。机器学习已广泛应用于数据挖掘、计算机视觉、自然语言处理、生物特征识别、搜索引擎、医学诊断、检测信用卡欺诈、证券市场分析、DNA序列测序、语音和手写识别、战略游戏和机器人等领域。"
    payload = tokenizer(payload, return_tensors="pt")
    bos_token_ids = payload["input_ids"][0][:2]
    eos_token_id = payload["input_ids"][0][-1:]
    pure_payload = payload["input_ids"][0][2:-1]
    if (seq_len - 3) <= len(pure_payload):
        payload = {
            "input_ids": torch.cat((bos_token_ids, pure_payload[:(seq_len - 3)], eos_token_id)).unsqueeze(0).to(device),
            "attention_mask": torch.ones((seq_len,), dtype=torch.int32).unsqueeze(0).to(device),
        }
    else:
        repeat = (seq_len - 3) // len(pure_payload)
        rest = (seq_len - 3) % len(pure_payload)
        payload = {
            "input_ids": torch.cat(
                (
                    bos_token_ids,
                    torch.cat((pure_payload.tile((repeat,)), pure_payload[:rest])),
                    eos_token_id,
                )
            ).unsqueeze(0).to(device),
            "attention_mask": torch.ones((seq_len,), dtype=torch.int32).unsqueeze(0).to(device),
        }
    latencies_per_seq = []
    num_gen_tokens = []
    latencies_per_token = []
    # Warm up
    max_length = int(seq_len * 1.5)
    min_length = seq_len // 2
    for _ in range(10):
        _ = model.generate(
            **payload,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=num_beam,
            min_length=min_length,
            max_length=max_length,
        )

    # Timed run
    for _ in tqdm(range(iterations)):
        start_time = perf_counter()
        generated_tokens = model.generate(
            **payload,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
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
model_id = "facebook/m2m100_418M"

device = torch.device("cuda:0")
pt_model, tokenizer = get_transformer_model(model_id)
pt_model.to(device)
onnx_model, _ = get_onnx_model(model_id)
onnx_model.to(device)

# Benchmark
res = []
seq_lengths = [8, 16, 32, 64, 128, 256, 512]
num_beams = [1]
for seq_len in seq_lengths:
    print("seq_len: ", seq_len)
    for num_beam in num_beams:
        print("num_beam: ", num_beam)
        df_pt = benchmark(seq_len, pt_model, tokenizer, device, frame="PyTorch", num_beam=num_beam, iterations=500)
        res.append(df_pt)

        df_onnx = benchmark(seq_len, onnx_model, tokenizer, device, frame="ONNX", num_beam=num_beam, iterations=500)
        res.append(df_onnx)

# Save result
res = pd.concat(res, ignore_index=True)
res.to_pickle("t4_res_ort_m2m100_418m_greedy.pkl")

pd.read_pickle('t4_res_ort_m2m100_418m_greedy.pkl').head(10)
