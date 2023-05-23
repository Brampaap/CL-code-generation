from torch.nn.utils.rnn import pad_sequence

from CL.common import types


def collate_fn(batch):
    input_data, target_data = zip(*batch)
    input_data = pad_sequence(
        input_data,
        batch_first=True,
        padding_value=types.SpecialTokens.PAD.value,
    )
    target_data = pad_sequence(
        target_data,
        batch_first=True,
        padding_value=types.SpecialTokens.PAD.value,
    )

    return input_data, target_data
