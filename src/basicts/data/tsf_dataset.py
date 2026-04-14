import json
import os
from typing import Union

import numpy as np

from basicts.utils.constants import BasicTSMode

from .base_dataset import BasicTSDataset


class BasicTSForecastingDataset(BasicTSDataset):
    """
    A dataset class for time series forecasting problems.

    Attributes:
        dataset_name (str): The name of the dataset.
        input_len (int): The length of the input sequence (number of historical points).
        output_len (int): The length of the output sequence (number of future points to predict).
        mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
        use_timestamps (bool): Flag to determine if timestamps should be used.
        local (bool): Flag to determine if the dataset is local.
        data_file_path (str | None): Path to the file containing the time series data. Default to "datasets/{dataset_name}".
        memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
    """

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            output_len: int,
            mode: Union[BasicTSMode, str],
            use_timestamps: bool = False,
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False) -> None:
        """
        Initializes the BasicTSForecastingDataset by setting up paths, loading data, and
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
            use_timestamps (bool): Flag to determine if timestamps should be used.
            local (bool): Flag to determine if the dataset is local.
            data_file_path (str | None): Path to the file containing the time series data. Default to "datasets/{name}".
            memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
        """
        super().__init__(dataset_name, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len

        if not local:
            pass  # TODO: support download remotely

        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}"

        try:
            self._data = np.load(
                os.path.join(data_file_path, f"{mode}_data.npy"),
                mmap_mode="r" if memmap else None
            )
            if use_timestamps:
                self.timestamps = np.load(
                    os.path.join(data_file_path, f"{mode}_timestamps.npy"),
                    mmap_mode="r" if memmap else None
                )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot load dataset from {data_file_path}, Please set a correct local path."
                "If you want to download the dataset, please set the argument `local` to False."
            ) from e

        self.memmap = memmap
        self.use_timestamps = use_timestamps

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.
        """
        item = {}

        history_data = self._data[index: index + self.input_len]
        future_data = self._data[index + self.input_len: index + self.input_len + self.output_len]

        item["inputs"] = history_data.copy() if self.memmap else history_data
        item["targets"] = future_data.copy() if self.memmap else future_data

        if self.use_timestamps:
            history_timestamps = self.timestamps[index: index + self.input_len]
            future_timestamps = self.timestamps[index + self.input_len: index + self.input_len + self.output_len]
            item["inputs_timestamps"] = history_timestamps.copy() if self.memmap else history_timestamps
            item["targets_timestamps"] = future_timestamps.copy() if self.memmap else future_timestamps

        return item

    def __len__(self) -> int:
        return len(self._data) - self.input_len - self.output_len + 1

    @property
    def data(self) -> np.ndarray:
        return self._data


class BasicTSTabularDataset(BasicTSDataset):
    """
    Dataset adapter for supervised tabular samples exported by
    scripts/data_preparation/normalized_data/generate_training_data.py

    Expected directory structure under data_file_path:
        train_data.npy
        train_label.npy
        train_timestamps.npy
        val_data.npy
        val_label.npy
        val_timestamps.npy
        test_data.npy
        test_label.npy
        test_timestamps.npy
        meta.json

    Task semantics:
        current inputs -> current output

    Returned item format:
        {
            "inputs": np.ndarray,              # [1, num_features]
            "targets": np.ndarray,             # [1, target_dim]
            "inputs_timestamps": np.ndarray,   # [1, num_timestamps]
            "targets_timestamps": np.ndarray,  # [1, num_timestamps]
        }

    Notes:
        1. This is NOT a sliding-window forecasting dataset.
        2. Each row is already one supervised sample.
        3. Historical information, if any, should already be encoded in feature columns
           (e.g. lag / rolling / statistical features).
    """

    def __init__(
            self,
            dataset_name: str,
            mode: Union[BasicTSMode, str],
            use_timestamps: bool = True,
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False,
            with_meta_check: bool = True) -> None:
        super().__init__(dataset_name, mode, memmap)

        if not local:
            pass  # TODO: support download remotely

        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}"

        mode_str = str(mode)

        data_path = os.path.join(data_file_path, f"{mode_str}_data.npy")
        label_path = os.path.join(data_file_path, f"{mode_str}_label.npy")
        ts_path = os.path.join(data_file_path, f"{mode_str}_timestamps.npy")
        meta_path = os.path.join(data_file_path, "meta.json")

        mmap_mode = "r" if memmap else None

        try:
            self._data = np.load(data_path, mmap_mode=mmap_mode)
            self._targets = np.load(label_path, mmap_mode=mmap_mode)
            self.timestamps = np.load(ts_path, mmap_mode=mmap_mode) if use_timestamps else None
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot load dataset from {data_file_path}. "
                f"Expected files: {mode_str}_data.npy / {mode_str}_label.npy / {mode_str}_timestamps.npy"
            ) from e

        self.use_timestamps = use_timestamps
        self.meta = None

        if with_meta_check and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

            expected_samples = self.meta.get("split", {}).get(f"{mode_str}_len", None)
            if expected_samples is not None and int(expected_samples) != int(len(self._data)):
                raise ValueError(
                    f"{dataset_name} {mode_str} split size mismatch: "
                    f"meta={expected_samples}, data={len(self._data)}"
                )

        if len(self._data) != len(self._targets):
            raise ValueError(
                f"{dataset_name} {mode_str} data/label length mismatch: "
                f"{len(self._data)} vs {len(self._targets)}"
            )

        if self.use_timestamps and self.timestamps is not None and len(self._data) != len(self.timestamps):
            raise ValueError(
                f"{dataset_name} {mode_str} data/timestamps length mismatch: "
                f"{len(self._data)} vs {len(self.timestamps)}"
            )

    def __getitem__(self, index: int) -> dict:
        item = {}

        x = self._data[index]
        y = self._targets[index]

        if x.ndim == 1:
            x = x[None, :]
        if y.ndim == 1:
            y = y[None, :]
        elif y.ndim == 0:
            y = np.array([[y]], dtype=np.float32)

        item["inputs"] = x.copy() if self.memmap else x
        item["targets"] = y.copy() if self.memmap else y

        if self.use_timestamps and self.timestamps is not None:
            ts = self.timestamps[index]
            if ts.ndim == 1:
                ts = ts[None, :]

            item["inputs_timestamps"] = ts.copy() if self.memmap else ts
            item["targets_timestamps"] = ts.copy() if self.memmap else ts

        return item

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> np.ndarray:
        return self._data

