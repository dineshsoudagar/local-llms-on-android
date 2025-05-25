import onnx
import os
import argparse
from onnx.external_data_helper import load_external_data_for_model

def merge_external_data(model_path: str, output_path: str = None):
    """
    Load an ONNX model with external data and merge it into a single .onnx file.

    Args:
        model_path (str): Path to the model.onnx file (may reference external data).
        output_path (str): Path to save the merged ONNX file. Defaults to model_path with '_merged.onnx' suffix.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model (with external data)
    print(f"[INFO] Loading ONNX model from: {model_path}")
    model = onnx.load(model_path, load_external_data=True)

    # Define output path
    if output_path is None:
        base, ext = os.path.splitext(model_path)
        output_path = f"{base}_merged{ext}"

    # Save model with all data packed inside
    print(f"[INFO] Saving merged model to: {output_path}")
    onnx.save_model(model, output_path, save_as_external_data=False)

    print("[SUCCESS] External data merged successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ONNX external tensor data into a single model file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model file (e.g. model.onnx)")
    parser.add_argument("--output_path", type=str, help="Path to save the merged ONNX model")

    args = parser.parse_args()

    try:
        merge_external_data(args.model_path, args.output_path)
    except Exception as e:
        print(f"[ERROR] {e}")
