from CL.common.dataset.pauq.pauq_dataset import Text2SQLDataset
from CL.curriculum.metrics import InputMetricABC


def sort_dataset(
    metric: InputMetricABC, dataset: Text2SQLDataset
) -> Text2SQLDataset:
    sql_tokens = dataset.trg_data
    indices = sorted(
        range(len(sql_tokens)),
        key=lambda idx: metric.get_score(sql_tokens[idx]),
    )
    dataset.src_data = list(map(dataset.src_data.__getitem__, indices))
    dataset.trg_data = list(map(dataset.trg_data.__getitem__, indices))

    return dataset
