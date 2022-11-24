# ONNX Runtime Inference Benchmark

### Benchmark

__Torch ORT__
1. Inference on Intel CPU - Optimum ORTModel(CPUProvider) V.S. ORTInferenceModule(OpenVINOProvider, `backend="CPU"`)
2. Inference on Intel integrated GPU -  Optimum ORTModel(CudaProvider) V.S ORTInferenceModule(OpenVINOProvider, `backend="GPU"`, `precisions = FP32/FP16`)

__ONNX Runtime__
1. Inference speed PyTorch vanilla V.S. Optimum ORTModels w/. IOBinding(BART, T5, gpt2 / T4)

--------------------------
__References__

- [ORTInferenceModule intro](https://github.com/pytorch/ort#accelerate-inference-for-pytorch-models-with-onnx-runtime-preview)
- [ORTInferenceModule src](https://github.com/pytorch/ort/blob/main/torch_ort_inference/torch_ort/ortinferencemodule/ortinferencemodule.py#L26)
- [ORTInferenceModule example(sequence classification with BERT)](https://github.com/pytorch/ort/blob/main/torch_ort_inference/tests/bert_for_sequence_classification.py)
