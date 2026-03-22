from enum import Enum


class TrellisMode(str, Enum):
    STOCHASTIC: str = 'stochastic'
    MULTIDIFFUSION: str = 'multidiffusion'


class TrellisPipeType(str, Enum):
    MODE_512: str = '512'
    MODE_1024: str = '1024'
    MODE_1024_CASCADE: str = '1024_cascade'
    MODE_1536_CASCADE: str = '1536_cascade'
