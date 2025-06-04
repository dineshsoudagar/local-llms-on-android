from transformers import AutoModelForCausalLM, AutoConfig
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.base import OnnxConfigWithPast
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers.cache_utils import HybridCache
from packaging import version
from typing import Dict, Optional, Any, Callable, Union
from collections import OrderedDict
import torch
import torch.nn as nn


def safe_patch_gemma(model):
    def patched_forward(self, *args, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)

        if "cache_position" not in kwargs or kwargs["cache_position"] is None:
            # ONNX tracing flattens HybridCache into dict; handle both
            try:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None and hasattr(past_key_values, "get_seq_length")
                    else 0
                )
            except Exception:
                # Try best-effort fallback for ONNX dict-style past_kv
                try:
                    key = next(iter(past_key_values))
                    past_seen_tokens = past_key_values[key].shape[2]
                except Exception:
                    past_seen_tokens = 0

            seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else 1
            device = kwargs["input_ids"].device if "input_ids" in kwargs else torch.device("cpu")
            kwargs["cache_position"] = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=device)

        return self._original_forward(*args, **kwargs)

    if not hasattr(model, "_original_forward"):
        model._original_forward = model.forward
        model.forward = patched_forward.__get__(model, model.__class__)
    return model



# ✅ Dummy config used by ONNXConfig
class Gemma2DummyNormalizedConfig:
    def __init__(self, config):
        self._config = config

    @property
    def vocab_size(self): return self._config.vocab_size
    @property
    def num_attention_heads(self): return self._config.num_attention_heads
    @property
    def num_hidden_layers(self): return self._config.num_hidden_layers
    @property
    def hidden_size(self): return self._config.hidden_size
    @property
    def use_cache(self): return self._config.use_cache
    @property
    def num_layers(self): return self._config.num_hidden_layers
    @property
    def num_key_value_heads(self): return self._config.num_key_value_heads

class DummyCacheEntry:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def update(self, key_states, value_states, layer_idx=None, cache_kwargs=None):
        return key_states, value_states  # simulate identity update

    def get_seq_length(self):
        return self.key.shape[2]  # assume key: (B, H, S, D)


from optimum.utils.input_generators import DummyPastKeyValuesGenerator

class Gemma2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: Gemma2DummyNormalizedConfig,
        batch_size: int = 2,
        sequence_length: int = 1,
        random_batch_size_range=None,
        random_sequence_length_range=None,
        **kwargs,
    ):
        self.num_layers = normalized_config.num_layers
        self.num_attention_heads = normalized_config.num_attention_heads
        self.num_key_value_heads = normalized_config.num_key_value_heads  # 🟢 Add this
        self.hidden_size = normalized_config.hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def supports_input(self, input_name: str) -> bool:
        return input_name.startswith("past_key_values")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,  # ✅ correct: match GQA head count
            1,
            256,  # ✅ per-head dim is still / attention_heads
        )

        print(f"[DUMMY GENERATOR] PKV shape = {shape} | dtype = {float_dtype} | framework = {framework}")

        return [
            (
                self.random_float_tensor(shape, framework=framework),#, dtype="fp32"),
                self.random_float_tensor(shape, framework=framework),#, dtype="fp32"),
            )
            for _ in range(self.num_layers)
        ]


# ✅ ONNX config for decoder with past
class Gemma2OnnxConfig(OnnxConfigWithPast):
    MIN_TRANSFORMERS_VERSION = version.parse("4.37.0")
    NORMALIZED_CONFIG_CLASS = Gemma2DummyNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Gemma2DummyPastKeyValuesGenerator)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "past_sequence_length + sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        outputs = OrderedDict({
            "logits": {0: "batch_size", 1: "sequence_length"},
        })
        if self.use_past:
            self.add_past_key_values(outputs, direction="outputs")
        return outputs


# === Load model and apply patch ===
model_id = "google/gemma-2-2b-it"
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)#, torch_dtype=torch.float32)

# ✅ Patch actual decoder for ONNX compatibility
model.model = safe_patch_gemma(model.model)

# === Wrap model for export ===
def get_submodels(model) -> Dict[str, nn.Module]:
    return {"decoder_with_past_model": model}

custom_onnx_config = {
    "decoder_with_past_model": Gemma2OnnxConfig(config, use_past=True, use_past_in_inputs=True)
}
print("Patched model.forward:", model.forward.__name__)
print("Patched model.model.forward:", model.model.forward.__name__)
# === Export ===
main_export(
    model_name_or_path=model_id,
    output="gemma2_with_past_onnx_2/",
    custom_onnx_configs=custom_onnx_config,
    task="text-generation-with-past",
    device="cuda",
    no_post_process=True,
    fn_get_submodels=lambda _: {"decoder_with_past_model": model},  # inject your patched model
    opset=19
)
