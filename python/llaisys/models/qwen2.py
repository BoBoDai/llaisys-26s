import json
from typing import Sequence
from ..libllaisys import DeviceType, DataType, LIB_LLAISYS
from ..libllaisys.qwen2_model import LlaisysQwen2Meta
from ctypes import c_size_t, c_int, c_int64, c_void_p, c_byte

from pathlib import Path
import safetensors

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        config = json.load(open(model_path / "config.json", "r"))

        self.meta = LlaisysQwen2Meta(
            dtype=DataType.BF16,
            nlayer=config["num_hidden_layers"],
            hs=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=config["eos_token_id"],
        )
        device_ids_list = [0]
        ndevice = len(device_ids_list)
        device_ids = (c_int * ndevice)(*device_ids_list)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(self.meta, device, device_ids, ndevice);

        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                tensor = data.get_tensor(name)
                LIB_LLAISYS.llaisysQwen2ModelLoadWeights(
                    self.model, 
                    name.encode("utf-8"), 
                    c_void_p(tensor.data_ptr()), 
                    c_size_t(tensor.numel() * tensor.element_size()))


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)

        # prefill: feed all input tokens
        n_input = len(tokens)
        token_ids = (c_int64 * n_input)(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, token_ids, c_size_t(n_input))
        tokens.append(next_token)
        if next_token == self.meta.end_token:
            return tokens

        # decode: feed one token at a time
        for _ in range(max_new_tokens - 1):
            token_ids = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, token_ids, c_size_t(1))
            tokens.append(next_token)
            if next_token == self.meta.end_token:
                break

        return tokens
