# ruff: noqa: SIM117
import dataclasses
import glob
import os
from tqdm.auto import tqdm
from typing import Generator, Iterable, List, Optional, Tuple, cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import LoadConfig, LoadFormat, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.loader import BaseModelLoader, _initialize_model, device_loading_context
from vllm.model_executor.model_loader.weight_utils import filter_duplicate_safetensors_files, _BAR_FORMAT

from vllm_mindspore.model_executor.model_loader.utils import set_default_torch_dtype

import mindspore as ms
from mindspore.communication import get_rank

logger = init_logger(__name__)


def safetensors_weights_iterator(hf_weights_files: List[str]) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = get_rank() == 0
    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        weights = ms.load_checkpoint(st_file, format="safetensors")
        for name, loaded_weight in weights.items():
            yield name, loaded_weight


class MsModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for " f"load format {load_config.load_format}"
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: Optional[str],
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded.
        """

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if not is_local:
            raise RuntimeError("Not support download weights now!")
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                raise RuntimeError("Not support download safetensors index file now!")
            hf_weights_files = filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file)
        else:
            raise RuntimeError("Not support other format weights now!")

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(self, source: "Source") -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(source.model_or_path, source.revision)
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            raise RuntimeError("Not support other format weights now!")

        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        primary_weights = MsModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[MsModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        raise RuntimeError("MindSpore doesnot support download model now!")

    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(vllm_config=vllm_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(self._get_all_weights(model_config, model))
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from " f"checkpoint: {weights_not_loaded}"
                    )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model.eval()


def get_ms_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        raise RuntimeError("Donot support for mindspore model now!")

    if load_config.load_format == LoadFormat.TENSORIZER:
        raise RuntimeError("Donot support for mindspore model now!")

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        raise RuntimeError("Donot support for mindspore model now!")

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        raise RuntimeError("Donot support for mindspore model now!")

    if load_config.load_format == LoadFormat.GGUF:
        raise RuntimeError("Donot support for mindspore model now!")

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        raise RuntimeError("Donot support for mindspore model now!")

    return MsModelLoader(load_config)


def get_ms_model(*, vllm_config: VllmConfig) -> nn.Module:
    loader = get_ms_model_loader(vllm_config.load_config)
    return loader.load_model(vllm_config=vllm_config)
