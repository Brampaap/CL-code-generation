from CL.common import types


class Tokenizer:
    """Word tokenizer"""

    def __init__(self, dataset: list[list[str]]):
        super(Tokenizer, self).__init__()

        self.vocab: dict[str, int] = {}
        self.reversed_vocab: dict[int, str] = {}
        self.__build__(dataset)

    def __build__(self, dataset: list[list[str]]):
        tokens = set()
        special_tokens = [el.value for el in types.SpecialStr]
        for row in dataset:
            tokens.update(row)

        # if you're changing this line, update types.py
        tokens = special_tokens + list(tokens)

        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.reversed_vocab = {i: token for token, i in self.vocab.items()}

    def encode_all(self, dataset: list[list[str]]):
        # TODO: add multiprocessing
        encoded = []
        for sent in dataset:
            encoded.append(self.encode(sent))

        return encoded

    def decode_all(self, dataset: list[list[int]]):
        # TODO: add multiprocessing
        decoded = []
        for sent in dataset:
            decoded.append(self.decode(sent))

        return decoded

    def encode(self, tokens: list[str]) -> list[int]:
        return [
            self.vocab.get(x, types.SpecialTokens.UNK.value) for x in tokens
        ]

    def decode(self, tokens: list[int]) -> list[str]:
        return [
            self.reversed_vocab.get(x, types.SpecialStr.UNK.value)
            for x in tokens
        ]

    def __len__(self):
        return len(self.vocab)
