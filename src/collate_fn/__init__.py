from .collate import collate_fn
from .collate_2 import collate_fn as collate_fn_fastspeech2

__all__ = {
    "collate_fn",
    "collate_fn_fastspeech2"
}