from collections.abc import Sequence
from typing import TypeVar


Item = TypeVar("Item")


def group_slides(slides: Sequence[Item], slides_per_screen: int = 4) -> list[list[Item]]:
    if slides_per_screen < 1:
        raise ValueError("slides_per_screen must be at least 1")
    return [
        list(slides[start : start + slides_per_screen])
        for start in range(0, len(slides), slides_per_screen)
    ]
