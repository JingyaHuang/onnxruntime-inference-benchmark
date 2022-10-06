import os
import time
from pathlib import Path
from time import perf_counter
import numpy as np

import datasets
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig

from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

# Compare ditilbert on text-classification: PyTorch V.S. Optimized ORT(CPU) V.S. ORTInferenceModule(OpenVINO)

model_id="optimum/distilbert-base-uncased-finetuned-banking77"
dataset_id="banking77"
onnx_path = Path("results/")

# # load vanilla transformers and convert to onnx
# model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# # save onnx checkpoint and tokenizer
# model.save_pretrained(onnx_path)
# tokenizer.save_pretrained(onnx_path)

# # create ORTOptimizer and define optimization configuration
# optimizer = ORTOptimizer.from_pretrained(model)
# optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations

# # apply the optimization configuration to the model
# optimizer.optimize(save_dir=onnx_path, optimization_config=optimization_config)

# # create ORTQuantizer and define quantization configuration
# model = ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model_optimized.onnx")
# dynamic_quantizer = ORTQuantizer.from_pretrained(model)
# dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# # apply the quantization configuration to the model
# model_quantized_path = dynamic_quantizer.quantize(save_dir=onnx_path, quantization_config=dqconfig)


def measure_latency(model, tokens):
    latencies = []
    # warm up
    for _ in range(10):
        _ = model(**tokens)
    # Timed run
    for _ in range(300):
        start_time = perf_counter()
        _ =  model(**tokens)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms

# Benchmark
payload="Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend "*2
print(f'Payload sequence length: {len(tokenizer(payload)["input_ids"])}')
tokens = tokenizer([payload], return_tensors="pt")

# Case 0: Vanilla PyTorch model & speed measurement
model = AutoModelForSequenceClassification.from_pretrained(model_id)
m_vanilla_pytorch_model=measure_latency(model, tokens)

# Case 1: Vanilla ONNX model & speed measurement
vanilla_onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model.onnx")
m_vanilla_onnx_model=measure_latency(vanilla_onnx_model, tokens)

# Case 2: Optimized pipe & speed measurement
optimized_model = ORTModelForSequenceClassification.from_pretrained(onnx_path,file_name="model_optimized_quantized.onnx")
m_optimized_model=measure_latency(optimized_model, tokens)

# Case 3: ORTInferenceModule
# 3.1 - CPU + FP32
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model = ORTInferenceModule(model)
m_ort_ov_model_fp32 = measure_latency(model, tokens)
# # 3.2 - GPU + FP16(need to test with Intel's GPU)
# model = AutoModelForSequenceClassification.from_pretrained(model_id)
# provider_options = OpenVINOProviderOptions(backend = "GPU", precision = "FP16")
# model = ORTInferenceModule(model, provider_options = provider_options)
# m_ort_ov_model_fp16 = measure_latency(model, tokens)

print(f"Vanilla PyTorch model: {m_vanilla_pytorch_model[0]}")
print(f"Vanilla ONNX model: {m_vanilla_onnx_model[0]}")
print(f"Optimized model: {m_optimized_model[0]}")
print(f"ORTInference wrapped model(fp32): {m_ort_ov_model_fp32[0]}")
print(f"Improvement through quantization(compared to pt): {round(m_vanilla_pytorch_model[1]/m_optimized_model[1],2)}x")
print(f"Improvement through quantization(compared to onnx): {round(m_vanilla_onnx_model[1]/m_optimized_model[1],2)}x")
print(f"Improvement through ORTInferenceModule(fp32)(compared to pt): {round(m_vanilla_pytorch_model[1]/m_ort_ov_model_fp32[1],2)}x")
print(f"Improvement through ORTInferenceModule(fp32)(compared to onnx): {round(m_vanilla_onnx_model[1]/m_ort_ov_model_fp32[1],2)}x")

# Payload sequence length: 128
# Vanilla PyTorch model: P95 latency (ms) - 23.65705099969091; Average latency (ms) - 23.48 +\- 0.79;
# Vanilla ONNX model: P95 latency (ms) - 39.39561919983135; Average latency (ms) - 25.02 +\- 3.55;
# Optimized model: P95 latency (ms) - 12.789480549781729; Average latency (ms) - 12.37 +\- 0.18;
# ORTInference wrapped model(fp32): P95 latency (ms) - 16.678588199465594; Average latency (ms) - 16.58 +\- 0.07;
# Improvement through quantization(compared to pt): 1.85x
# Improvement through quantization(compared to onnx): 3.08x
# Improvement through ORTInferenceModule(fp32)(compared to pt): 1.42x
# Improvement through ORTInferenceModule(fp32)(compared to onnx): 2.36x