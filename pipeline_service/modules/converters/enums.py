from enum import Enum


class AlphaMode(str, Enum):
    OPAQUE: str = 'OPAQUE'
    MASK: str = 'MASK'
    BLEND: str = 'BLEND'
    DITHER: str = 'DITHER'

    @property
    def cutoff(self) -> float | None:
        if self == AlphaMode.MASK or self == AlphaMode.DITHER:
            return 0.5
        elif self == AlphaMode.BLEND:
            return 0.0
        return None
