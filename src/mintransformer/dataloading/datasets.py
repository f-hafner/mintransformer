"""Mini-implementation of huggingface datasets."""

from __future__ import annotations
import logging
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
