from typing import (TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional,
                    Protocol, Type, Union, overload, runtime_checkable)

@runtime_checkable
class SupportsLoRA(Protocol):
    """The interface required for all models that support LoRA."""

    supports_lora: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports LoRA.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    packed_modules_mapping: ClassVar[Dict[str, List[str]]]
    supported_lora_modules: ClassVar[List[str]]
    embedding_modules: ClassVar[Dict[str, str]]
    embedding_padding_modules: ClassVar[List[str]]

@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: Dict[str, List[str]]
    supported_lora_modules: List[str]
    embedding_modules: Dict[str, str]
    embedding_padding_modules: List[str]