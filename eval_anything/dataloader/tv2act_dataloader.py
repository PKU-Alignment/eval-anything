from eval_anything.dataloader.base_dataloader import BaseDataLoader

import glob
import gzip
import multiprocessing as mp
import os
import pickle
import shutil
import warnings
from collections import defaultdict
from typing import Optional, Dict, Sequence, List, Any, Union
import json
import copy
from typing import Literal, Union
from tqdm import tqdm
import compress_json


class Dataset:
    def __init__(
        self, data: List[Any], dataset: str, split: Literal["train", "val", "test"]
    ) -> None:
        """Initialize a dataset split.

        Args:
            data: The data of the dataset split.
            dataset: The name of the dataset.
            split: The name of the dataset split.
        """
        self.data = data
        self.dataset = dataset
        self.split = split

    def __iter__(self):
        """Return an iterator over the dataset."""
        for item in self.data:
            yield item

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        return self.data[index]

    def __repr__(self):
        """Return a string representation of the dataset."""
        return (
            "Dataset(\n"
            f"    dataset={self.dataset},\n"
            f"    size={len(self.data)},\n"
            f"    split={self.split}\n"
            ")"
        )

    def __str__(self):
        """Return a string representation of the dataset."""
        return self.__repr__()

    def select(self, indices: Sequence[int]) -> "Dataset":
        """Return a new dataset containing only the given indices."""
        # ignoring type checker due to mypy bug with attrs
        return Dataset(
            data=[self.data[i] for i in indices],
            dataset=self.dataset,
            split=self.split,
        )  # type: ignore
        
class LazyJsonDataset(Dataset):
    """Lazily load the json house data."""

    def __init__(
        self, data: List[Any], dataset: str, split: Literal["train", "val", "test"]
    ) -> None:
        super().__init__(data, dataset, split)
        self.cached_data: Dict[int, Union[list, dict]] = {}

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        if index not in self.cached_data:
            self.cached_data[index] = json.loads(self.data[index])
        return self.cached_data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return super().__len__()

    def __repr__(self):
        """Return a string representation of the dataset."""
        return super().__repr__()

    def __str__(self):
        """Return a string representation of the dataset."""
        return super().__str__()

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            if i not in self.cached_data:
                self.cached_data[i] = json.loads(x)
            yield self.cached_data[i]

    def select(self, indices: Sequence[int]) -> "Dataset":
        """Return a new dataset containing only the given indices."""
        # ignoring type checker due to mypy bug with attrs
        return LazyJsonDataset(
            data=[self.data[i] for i in indices],
            dataset=self.dataset,
            split=self.split,
        )  # type: ignore
        
class DatasetDict:
    def __init__(
        self,
        train: Optional[Dataset] = None,
        val: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test

    def __getitem__(self, key: str) -> Any:
        if key == "train":
            if self.train is None:
                raise KeyError(key)
            return self.train
        elif key == "val":
            if self.val is None:
                raise KeyError(key)
            return self.val
        elif key == "test":
            if self.test is None:
                raise KeyError(key)
            return self.test
        else:
            raise KeyError(key)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return (
            "DatasetDict(\n"
            f"    train={self.train},\n"
            f"    val={self.val},\n"
            f"    test={self.test}\n"
            ")"
        )

    def __str__(self):
        """Return a string representation of the dataset."""
        return self.__repr__()




def load_jsongz_as_str(subpath: Optional[str]) -> str:
    """Load the subpath file."""

    if subpath is None:
        # If the subpath is None, return a json representation of `None`
        return "null"

    with gzip.open(subpath, "r") as f:
        return f.read().strip()


def read_jsonlgz(path: str, max_lines: Optional[int] = None):
    with gzip.open(path, "r") as f:
        lines = []
        for line in tqdm(f, desc=f"Loading {path}"):
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


def read_jsongz_files(
    paths: Sequence[str], num_workers: int, max_files: Optional[int] = None
) -> List[str]:
    if len(paths) == 0:
        return []

    ind_to_path = {int(os.path.basename(p).split(".")[0]): p for p in paths}
    max_ind = max(ind_to_path.keys())

    missing_count = 0
    for i in range(max_ind + 1):
        if i not in ind_to_path:
            ind_to_path[i] = None
            missing_count += 1

    if missing_count > 0:
        warnings.warn(f"Missing {missing_count} files in {os.path.dirname(paths[0])}.")

    paths = [ind_to_path[i] for i in range(max_ind + 1)]

    if max_files is not None:
        paths = paths[:max_files]

    if len(paths) == 0:
        return []

    if num_workers > 0:
        with mp.Pool(num_workers) as p:  # Create a multiprocessing Pool
            file_data = p.map(load_jsongz_as_str, paths)
            return file_data
    else:
        return [load_jsongz_as_str(p) for p in tqdm(paths, desc=f"Loading {paths[0]}")]


def get_cache_file_path(path: str) -> str:
    """Get the cache file name for a split."""
    return os.path.join(path, f"dataset_cache.pkl.gz")


def compress_file_then_delete_old(
    input_file_path: str, output_file_path: str, compresslevel: int
):
    with open(input_file_path, "rb") as f_in:
        with gzip.open(output_file_path, "wb", compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(input_file_path)


def save_pickle_gzip(
    data: Any, save_path: str, compresslevel: int = 2, protocol: int = 4
) -> None:
    assert save_path.endswith(".pkl.gz")
    tmp_path = save_path.replace(".gz", "")
    assert not os.path.exists(tmp_path)
    print(f"Caching dataset to {save_path}, this may take a few minutes...")
    with open(tmp_path, "wb") as f:
        pickle.dump(
            obj=data,
            file=f,
            protocol=protocol,
        )
    compress_file_then_delete_old(tmp_path, save_path, compresslevel=compresslevel)


def load_pickle_gzip(path: str):
    assert path.endswith(".pkl.gz")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)



def process_and_load_data(split, task_type, path):
    if split == "val":
        filename = f"closedtype_minival_Oct22_fixedlemma_filterasset_{task_type.lower()}_val.jsonl.gz"
    else:
        raise ValueError(f"Unknown split {split}")
    full_path = os.path.join(path, filename)
    print(path, filename, full_path)
    
    with gzip.open(full_path, "rt") as f:
        tasks = [line for line in tqdm(f, desc=f"Loading {split}")]

    return tasks


class TV2ACTDataLoader(BaseDataLoader):
    
    def __init__(self):
        
        pass
    
    def load_task_dataset(self, task_types: str, path: str = None) -> DatasetDict:
        """Load the houses dataset."""
        data = {}

        for split in ["val"]:
            split_task_list = []
            tasks = process_and_load_data(split, task_types, path)
            split_task_list.extend(tasks)

            data[split] = LazyJsonDataset(
                data=split_task_list, dataset="vida-benchmark", split=split
            )
        return DatasetDict(**data)
    
    
    def load_house_assets(
        self, 
        path_to_splits: Optional[str],
        split_to_path: Optional[Dict[str, str]] = None,
        num_workers: int = 0,
        use_cache: bool = False,  # TODO: Caching -> ~2x speed up when loading, should be faster to justify complexity
        max_houses_per_split: Optional[Union[int, Dict[str, int]]] = None,
    ) -> DatasetDict:
        """Load the dataset from a path or a mapping from split to path.

        Arguments :
            path_to_splits (Optional[str]): Path to a directory containing train, val, and test splits.
            split_to_path (Optional[Dict[str, str]]): Mapping from split to path to a directory containing the split.
            num_workers (int): The number of worker processes to use. If 0, no parallelization is done. If <0, all
                available cores are used.
            use_cache (bool): Whether to use a cache file to speed up loading. If True, the cache file will be saved
                in the directory specified by `path_to_splits`.

        Returns:
            prior.DatasetDict: A dictionary of LazyJsonDataset objects for each split found in the input.
        """
        assert (path_to_splits is None) != (
            split_to_path is None
        ), "Exactly one of path or split_to_path must be provided."

        assert (not use_cache) or path_to_splits is not None, (
            "Must provide `path_to_splits` to splits to use cache as we"
            " will save the cache file into this directory."
        )
        assert (not use_cache) or (
            max_houses_per_split is None
        ), "Cannot use cache when `max_houses_per_split` is not None."

        if not isinstance(max_houses_per_split, Dict):
            max_houses_per_split = (lambda x: defaultdict(lambda: x))(max_houses_per_split)

        if use_cache:
            cache_file_path = get_cache_file_path(path_to_splits)
            if os.path.exists(cache_file_path):
                return load_pickle_gzip(cache_file_path)

        if num_workers < 0:
            num_workers = mp.cpu_count()

        if path_to_splits is not None:
            assert os.path.exists(path_to_splits), f"Path {path_to_splits} does not exist."
            split_to_path = {
                # "train": os.path.join(path_to_splits, "train.jsonl.gz"),
                "val": os.path.join(path_to_splits, "val.jsonl.gz"),
                # "test": os.path.join(path_to_splits, "test.jsonl.gz"),
            }

        for split, path in list(split_to_path.items()):
            if not os.path.exists(path):
                del split_to_path[split]
                warnings.warn(
                    f"Split {split} does not exist at path {path}, won't be included."
                )

        if len(split_to_path) == 0:
            raise ValueError("No splits found.")

        split_to_house_strs = defaultdict(lambda: [])
        for split, path in split_to_path.items():
            if path.endswith(".jsonl.gz"):
                split_to_house_strs[split] = read_jsonlgz(
                    path=path,
                    max_lines=max_houses_per_split[split],
                )
            elif os.path.isdir(path):
                subpaths = glob.glob(os.path.join(path, "*.json.gz"))
                split_to_house_strs[split] = read_jsongz_files(
                    paths=subpaths,
                    num_workers=num_workers,
                    max_files=max_houses_per_split[split],
                )
            else:
                raise NotImplementedError(f"Unknown path type: {path}")

        dd = DatasetDict(
            **{
                split: LazyJsonDataset(data=houses, dataset="procthor-100k", split=split)
                for split, houses in split_to_house_strs.items()
            }
        )

        if use_cache:
            cache_file_path = get_cache_file_path(path_to_splits)
            if not os.path.exists(cache_file_path):
                save_pickle_gzip(data=dd, save_path=cache_file_path)

        return dd
