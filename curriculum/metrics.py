import torch


class InputMetricABC:
    def __init__(self):
        super(InputMetricABC, self).__init__()

    def __call__(self, input_data: list, *args, **kwargs):
        raise NotImplementedError

    def get_score(self, input_data: torch.IntTensor):
        return self(input_data)


class SentLenMetric(InputMetricABC):
    def __call__(self, input_data: list, *args, **kwargs):
        return len(input_data)
