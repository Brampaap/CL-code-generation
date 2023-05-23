from enum import Enum


class SpecialStr(Enum):
    PAD = "<PAD>"
    UNK = "<UNK>"
    EOS = "<EOS>"
    BOS = "<BOS>"


class SpecialTokens(Enum):
    PAD = 0
    UNK = 1
    EOS = 2
    BOS = 3
