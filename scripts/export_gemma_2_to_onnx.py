# export_gemma2_with_past_onnx.py
#
# This script exports the Gemma 2B IT model to ONNX with full support for past key/value caching
# and dynamic shapes. It uses the Optimum ONNX exporter and a custom OnnxConfig to wire all inputs
# (input_ids, attention_mask, position_ids, past_key_values) correctly into the graph.
#
# Usage:
#   python export_gemma2_with_past_onnx.py
#
# Requires:
#   - transformers>=4.37.0
#   - optimum>=1.x
#   - torch
#   - packaging
#   - A CUDA-enabled device (or change device="cpu" below)
#
from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoConfig
from packaging import version
from collections import OrderedDict

# Optimum ONNX exporter utilities:
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.base import OnnxConfigWithPast
from optimum.utils.input_generators import DummyTextInputGenerator, DummyPastKeyValuesGenerator

# We will import HybridCache from HF to inspect cache shapes (not strictly used in ONNX).
from transformers.cache_utils import HybridCache


def hybrid_to_legacy_cache(self):
    return tuple((self.key_cache[i], self.value_cache[i]) for i in range(len(self.key_cache)))


HybridCache.to_legacy_cache = hybrid_to_legacy_cache


def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert HybridCache into legacy list-of-tuples format"""
    return tuple((self.key_cache[i], self.value_cache[i]) for i in range(len(self.key_cache)))


# --------------------------------------------------------------------------------------------------
# 1) A helper function to “patch” the model’s forward method so that `cache_position` is always set.
#    During ONNX tracing, the Python-side generation of cache_position might be skipped, so we ensure
#    it’s created from either past_key_values or inputs_embeds shape.
# --------------------------------------------------------------------------------------------------
def safe_patch_gemma(model: nn.Module) -> nn.Module:
    """
    Replaces model.forward to ensure:
      - cache_position is always generated
      - HybridCache is unwrapped into tuples
      - ONNX exporter sees the correct input signature
    """
    original_forward = model.forward  # scoped correctly here

    def patched_forward(input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, **kwargs):
        # === Ensure cache_position is set ===
        if "cache_position" not in kwargs or kwargs["cache_position"] is None:
            try:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None and hasattr(past_key_values, "get_seq_length")
                    else 0
                )
            except Exception:
                try:
                    key0 = next(iter(past_key_values))
                    past_seen_tokens = past_key_values[key0].shape[2]
                except Exception:
                    past_seen_tokens = 0

            if inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
                device = inputs_embeds.device
            else:
                seq_len = input_ids.shape[1] if input_ids is not None else 1
                device = input_ids.device if input_ids is not None else torch.device("cpu")

            kwargs["cache_position"] = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=device, dtype=torch.long
            )

        # === Run the original forward ===
        output = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        print(f"Output type: {type(output)}")
        # === Unwrap HybridCache if present ===
        from transformers.cache_utils import HybridCache
        if isinstance(output, dict) and "past_key_values" in output:
            pkv = output["past_key_values"]
            if isinstance(pkv, HybridCache):
                output["past_key_values"] = tuple(
                    (pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache))
                )
        elif isinstance(output, HybridCache):
            output = tuple((output.key_cache[i], output.value_cache[i]) for i in range(len(output.key_cache)))

        return output

    # Apply patch
    model.forward = patched_forward
    return model


# --------------------------------------------------------------------------------------------------
# 2) A "normalized" dummy config class wrapping Gemma2Config so that OnnxConfigWithPast can read it.
#    The ONNX exporter uses this to infer shapes (# layers, # heads, etc.).
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# 3) A dummy “cache entry” class that simulates how HybridCache works.
#    This is only used to tell the ONNX exporter how to shape past_key_values.
# --------------------------------------------------------------------------------------------------
class DummyCacheEntry:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def update(self, key_states, value_states, layer_idx=None, cache_kwargs=None):
        # ONNX export will “trace” calls to .update(); we simply return the same tensors
        return key_states, value_states

    def get_seq_length(self):
        # If key has shape [B, H_kv, S_cached, D], return S_cached
        return self.key.shape[2]


# --------------------------------------------------------------------------------------------------
# 4) A DummyPastKeyValuesGenerator: used by the ONNXConfig to create fake past_key_values for tracing.
#    This generator returns a list of (key, value) pairs per layer, where each key/value has shape:
#      [batch_size, num_key_value_heads, 1, head_dim]
#    This allows the ONNX exporter to “see” the correct shape of past_kvs.
# --------------------------------------------------------------------------------------------------
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
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        # Number of layers and heads
        self.num_layers = normalized_config.num_layers
        self.num_attention_heads = normalized_config.num_attention_heads
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.hidden_size = normalized_config.hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def supports_input(self, input_name: str) -> bool:
        # ONNX exporter will ask for “past_key_values_0”, “past_key_values_1”, ... etc.
        return input_name.startswith("past_key_values")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Returns a list of (key, value) tuples for each layer:
          - Each key/value: [batch_size, num_key_value_heads, 1, head_dim]
          - head_dim = hidden_size // num_attention_heads
        """
        head_dim = self.hidden_size // self.num_attention_heads
        shape = (self.batch_size, self.num_key_value_heads, 1, 256)

        # Print shape so we can verify during export
        print(f"[DUMMY PKV] layer-shape = {shape}, dtype={float_dtype}, framework={framework}")

        # Generate `num_layers` pairs of random tensors
        return [
            (
                self.random_float_tensor(shape, framework=framework),
                self.random_float_tensor(shape, framework=framework),
            )
            for _ in range(self.num_layers)
        ]


# --------------------------------------------------------------------------------------------------
# 5) Custom ONNXConfig for Gemma2: tells Optimum exactly which inputs/outputs to expect when exporting
#    * inputs(): includes input_ids, attention_mask, position_ids, and past_key_values (all dynamic)
#    * outputs(): includes logits + present_key_values (if use_cache=True)
# --------------------------------------------------------------------------------------------------
class Gemma2OnnxConfig(OnnxConfigWithPast):
    MIN_TRANSFORMERS_VERSION = version.parse("4.37.0")
    NORMALIZED_CONFIG_CLASS = Gemma2DummyNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,  # for input_ids / attention_mask
        Gemma2DummyPastKeyValuesGenerator  # for past_key_values
    )

    @property
    def inputs(self) -> dict:
        """
        Returns a dict of input names → dynamic axes maps.
        0 → batch_size, 1 → seq_len for input_ids/attention_mask/position_ids.
        If `use_past_in_inputs=True`, automatically add past_key_values_N inputs.
        """
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            # attention_mask will have shape [B, past_seq_len + sequence_length]
            "attention_mask": {0: "batch_size", 1: "past_sequence_length + sequence_length"},
            # position_ids: [B, sequence_length]
            "position_ids": {0: "batch_size", 1: "sequence_length"},
        }

        if self.use_past_in_inputs:
            # This will add past_key_values_0, past_key_values_1, ... each with shape [B, num_kv_heads, past_seq_len, head_dim]
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    @property
    def outputs(self) -> dict:
        """
        Returns a dict of output names → dynamic axes maps.
        Always outputs:
          - logits: [batch_size, sequence_length, vocab_size]
        If `use_past=True`, also outputs present_key_values (one pair per layer)
        """
        outputs = OrderedDict({
            "logits": {0: "batch_size", 1: "sequence_length"},
        })
        if self.use_past:
            self.add_past_key_values(outputs, direction="outputs")
        return outputs


# --------------------------------------------------------------------------------------------------
# 6) Main execution: load config, load model, patch, wrap with custom ONNXConfig, and call main_export()
# --------------------------------------------------------------------------------------------------
def main():
    # 6.1) Identify the Hugging Face model ID
    model_id = "google/gemma-2-2b-it"

    # 6.2) Load configuration from Hugging Face
    config = AutoConfig.from_pretrained(model_id)
    # Ensure use_cache=True so that past_key_values logic is active
    config.use_cache = True

    # 6.3) Load the pre-trained Gemma2ForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # 6.4) Patch the underlying .model.forward to guarantee cache_position is always provided
    model = safe_patch_gemma(model)

    # 6.5) Create our custom ONNXConfig: “decoder_with_past_model” is the submodel name
    custom_onnx_config = {
        "decoder_with_past_model": Gemma2OnnxConfig(
            config,
            use_past=True,  # we want present_key_values in outputs
            use_past_in_inputs=False  # we want past_key_values in inputs
        )
    }

    # 6.6) Define a function that maps submodel name to the actual nn.Module (patched Gemma2ForCausalLM)
    def get_submodels(_):
        return {"decoder_with_past_model": model}

    # 6.7) Call Optimum’s main_export to produce ONNX files
    #      - task="text-generation-with-past" instructs Optimum to export the CausalLM with past-key-values logic
    #      - opset=19 to support dynamic shapes and newer operators
    #      - device="cuda": export on GPU; change to "cpu" if CUDA is unavailable

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Check that past_key_values is a tuple of tuples
    print(type(outputs))
    if isinstance(outputs, dict):
        print("past_key_values type:", type(outputs["past_key_values"]))
    elif isinstance(outputs, tuple) and hasattr(outputs, "past_key_values"):
        print("past_key_values type:", type(outputs.past_key_values))

    main_export(
        model_name_or_path=model_id,
        output="gemma2_with_past_KV_onnx_latest_fixing/",
        custom_onnx_configs=custom_onnx_config,
        task="text-generation-with-past",
        device="cuda",
        fn_get_submodels=get_submodels,
        opset=19,
    )

    print("✅ ONNX export complete")


if __name__ == "__main__":
    main()
