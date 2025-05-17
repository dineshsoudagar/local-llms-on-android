import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="meta-llama_Llama-3.2-1B_onnx/model.onnx",             # your model
    model_output="meta-llama_Llama-3.2-1B_onnx/quantized_model.onnx",  # output file
    weight_type=QuantType.QInt8           # or QuantType.QUInt8
)
