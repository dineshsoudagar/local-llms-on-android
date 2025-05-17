# Load the model without loading external data automatically
import onnx
from onnx.external_data_helper import load_external_data_for_model, convert_model_from_external_data

model_path = "qwen_2.5_05B_onnx_converted_w_optim/model.onnx"
external_data_dir = "qwen_2.5_05B_onnx_converted_w_optim"  # Folder containing model.onnx_data

# Step 1: Load model without external data
model = onnx.load(model_path, load_external_data=True)

# Merge external data into the main ONNX file
onnx.save_model(
    model,
    "merged_model.onnx",
    save_as_external_data=False  # This ensures everything goes into one file
)