import onnx
from onnx import shape_inference

model = onnx.load("meta-llama_Llama-3.2-1B_onnx/model.onnx")
print("loadede")
inferred_model = shape_inference.infer_shapes(model)
print("loadede")
graph = inferred_model.graph

def shape2tuple(tensor_type):
    return tuple(dim.dim_value if (dim.dim_value > 0) else None for dim in tensor_type.shape.dim)

# Iterate over nodes and find all Trilu nodes
for node in graph.node:
    print(node)
    if node.op_type.lower() == 'trilu':
        print(f"Node: {node.name} (type: {node.op_type})")
        # Print input shapes
        for input_name in node.input:
            value_info = next((vi for vi in graph.value_info if vi.name == input_name), None)
            if value_info:
                shape = shape2tuple(value_info.type.tensor_type)
                print(f"  Input: {input_name}, shape: {shape}")
            else:
                print(f"  Input: {input_name}, shape: Unknown")
        # Print output shapes
        for output_name in node.output:
            value_info = next((vi for vi in graph.value_info if vi.name == output_name), None)
            if value_info:
                shape = shape2tuple(value_info.type.tensor_type)
                print(f"  Output: {output_name}, shape: {shape}")
            else:
                print(f"  Output: {output_name}, shape: Unknown")
        print()


all_ops = set(node.op_type for node in model.graph.node)
print("Operators used in model:", all_ops)
# Check for 'Trilu' nodes
trilu_nodes = [node for node in model.graph.node if node.op_type == "Trilu"]
print(f"Found {len(trilu_nodes)} Trilu nodes")

# Known supported ops (partial, update as needed)
supported_ops_mobile = {
    'Add', 'Relu', 'MatMul', 'Conv', 'Gemm', 'Reshape',
    'Transpose', 'Softmax', 'Sigmoid', 'Tanh', 'BatchNormalization',
    'MaxPool', 'AveragePool', 'Flatten', 'Concat', 'Mul', 'Sub',
    'Div', 'Clip', 'Cast', 'Shape', 'Unsqueeze', 'Squeeze',
    'ReduceMean', 'ReduceSum', 'Gather', 'Expand', 'Slice',
}

used_ops = set(node.op_type for node in model.graph.node)
unsupported = used_ops - supported_ops_mobile
print("Unsupported ops on mobile:", unsupported)


# Load your ONNX model
# Run shape inference to populate shape info

