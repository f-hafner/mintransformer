"""Mini-implementation of huggingface datasets."""

from __future__ import annotations
import logging
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_sample_data() -> tuple[list[list[int]], list[list[int]]]:
    """Create sample integer sequence data for testing."""
    # Example: simple sequences where y is x shifted by 1
    x_sequences = [
        [3, 5, 7, 8, 5, 10],
        [1, 2, 3, 4, 5, 6],
        [10, 9, 8, 7, 6, 5],
        [2, 4, 6, 8, 10, 12],
        [15, 14, 13, 12, 11, 10],
        [20, 22, 24, 26, 28, 30],
    ]

    y_sequences = [
        [5, 7, 8, 5, 10, 15],
        [2, 3, 4, 5, 6, 7],
        [9, 8, 7, 6, 5, 4],
        [4, 6, 8, 10, 12, 14],
        [14, 13, 12, 11, 10, 9],
        [22, 24, 26, 28, 30, 32],
    ]

    return x_sequences, y_sequences


def write_to_arrow_stream(data: dict[str, list[list[int]]], filename: str) -> None:
    """Create memory-mappable arrow file.

    A list of list of ints is stored as one column, where each inner
    list corresponds to one record.
    """
    table = pa.table({k: pa.array(v) for k, v in data.items()})

    with pa.OSFile(filename, "wb") as sink, pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)


class MemoryMappedTable:
    """Minimal implementation of HF memmapped table."""

    def __init__(self, table: pa.Table, path: str, colnames: list[str]):
        self.table = table
        self.path = path
        self.colnames = colnames

    @property
    def num_rows(self) -> int:
        """Returns number of rows in the table."""
        return self.table.num_rows

    @classmethod
    def read_file(cls, filename: str) -> MemoryMappedTable:
        """Read file into class."""
        memory_mapped_stream = pa.memory_map(filename, "r")
        opened_stream = pa.ipc.open_stream(memory_mapped_stream)

        pa_table = opened_stream.read_all()
        return cls(pa_table, filename, pa_table.column_names)

    def slice(self, offset: int, length: int) -> MemoryMappedTable:
        """Slice a table. Calls slice method of pyarrow table."""
        return MemoryMappedTable(self.table.slice(offset=offset, length=length), self.path, self.colnames)

    def fast_slice(self, offset: int, length: int) -> pa.Table:
        """Fast way to search. See original implementation."""
        raise NotImplementedError


class ArrowReader:
    """Class for reading arrow files."""

    @staticmethod
    def read(filename: str, in_memory: bool = True) -> dict:
        """Read file into kwargs to build a single Dataset instance."""
        if in_memory:
            pa_table = MemoryMappedTable.read_file(filename)
        else:
            raise NotImplementedError

        return {"arrow_table": pa_table}  # in original, this dict has more items


class ArrowDataset:  # datsets.arrow_dataset.Dataset
    """Mini HF dataset backed by arrow table."""

    def __init__(self, arrow_table: MemoryMappedTable):
        self.data = arrow_table
        # split? indices_table?

    def __iter__(self):
        """Iterate through the samples."""
        table = self.data.table
        colnames = self.data.colnames
        for i in range(table.num_rows):
            sample = table.slice(i, 1)  # this fetches exactly 1 record
            data = {col: sample.column(col).chunk(0).flatten().to_numpy() for col in colnames}
            data = {k: torch.from_numpy(v) for k, v in data.items()}
            yield data


class CustomDataset(Dataset):
    """Custom torch dataset."""

    def __init__(self, dataset: ArrowDataset):
        self.dataset = dataset
        self.samples = []

        n_samples = dataset.data.table.num_rows

        for sample in tqdm(self.dataset, "Building dataset", total=n_samples):
            # FIXME: implicitly assuming the columns are in the right order! (ie, (x,y))
            sample_tuple = tuple(sample.values())
            self.samples.append(sample_tuple)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create data, save as arrow IPC file
    x_seqs, y_seqs = create_sample_data()
    data_dict = {"x": x_seqs, "y": y_seqs}

    filename = "data/mydata.arrow"
    write_to_arrow_stream(data=data_dict, filename=filename)

    # Arrow dataset
    dataset_kwargs = ArrowReader().read(filename=filename, in_memory=True)
    arrow_dataset = ArrowDataset(**dataset_kwargs)
    dataset = CustomDataset(arrow_dataset)

    # Dataloader
    # Note: each process will load (and memory-map) the data separately!
    dataloader = DataLoader(dataset, batch_size=3)

    myiter = iter(dataloader)
    batch = next(myiter)
    logger.info("First batch: %s", batch)
