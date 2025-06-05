from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import Qwen2OnnxConfig
from packaging import version

# ✅ Inherit from Qwen2 config
class Gemma2OnnxConfig(Qwen2OnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.37.0")

    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "position_ids": {0: "batch", 1: "sequence"},
        }

    @property
    def outputs(self):
        return {
            "logits": {0: "batch", 1: "sequence"},
        }

# ✅ Your Hugging Face model ID
model_id = "google/gemma-2-2b-it"

# ✅ Preload config for use in OnnxConfig
config = AutoConfig.from_pretrained(model_id)

# ✅ Custom ONNX config map
custom_config = {
    "model": Gemma2OnnxConfig(config)
}

# ✅ Export the model
main_export(
    model_id,
    output="gemma2_onnx_no_past_kv/",
    custom_onnx_configs=custom_config,
    task="text-generation-with-past",  # crucial for decoder models
    device="cuda",                      # safer for memory
    optimize="O1",                     # light optimization, low RAM
    no_post_process=True              # prevents decoder merge (RAM saver)
)
