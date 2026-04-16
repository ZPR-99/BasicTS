from typing import Union

import numpy as np
import torch

from .base_scaler import BasicTSScaler


class NaNSafeZScoreScaler(BasicTSScaler):
    """
    NaN-safe Z-score scaler.

    Rules:
        1. Only finite values participate in fitting.
        2. NaN / +Inf / -Inf are excluded from statistics.
        3. All-invalid channels fallback to mean=0, std=1.
        4. Zero-variance channels fallback to std=1.
    """

    def __init__(self, norm_each_channel: bool, rescale: bool, stats: dict = None):
        super().__init__(norm_each_channel, rescale, stats or {})

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        if self.stats:
            return

        if isinstance(data, np.ndarray):
            valid_mask = np.isfinite(data)

            if self.norm_each_channel:
                valid_count = valid_mask.sum(axis=-2, keepdims=True)

                safe_data = np.where(valid_mask, data, 0.0)
                mean = safe_data.sum(axis=-2, keepdims=True) / np.maximum(valid_count, 1)

                centered = np.where(valid_mask, data - mean, 0.0)
                std = np.sqrt((centered ** 2).sum(axis=-2, keepdims=True) / np.maximum(valid_count, 1))

                mean = np.where(valid_count > 0, mean, 0.0)
                std = np.where((valid_count > 0) & np.isfinite(std) & (std > 0), std, 1.0)

            else:
                valid_count = valid_mask.sum()

                if valid_count == 0:
                    mean = 0.0
                    std = 1.0
                else:
                    safe_data = np.where(valid_mask, data, 0.0)
                    mean = safe_data.sum() / valid_count

                    centered = np.where(valid_mask, data - mean, 0.0)
                    std = np.sqrt((centered ** 2).sum() / valid_count)

                    if not np.isfinite(mean):
                        mean = 0.0
                    if (not np.isfinite(std)) or std == 0:
                        std = 1.0

            self.stats["mean"] = torch.tensor(mean, dtype=torch.float32)
            self.stats["std"] = torch.tensor(std, dtype=torch.float32)

        else:
            valid_mask = torch.isfinite(data)

            if self.norm_each_channel:
                valid_count = valid_mask.sum(dim=-2, keepdim=True)

                safe_data = torch.where(valid_mask, data, torch.zeros_like(data))
                mean = safe_data.sum(dim=-2, keepdim=True) / valid_count.clamp_min(1)

                centered = torch.where(valid_mask, data - mean, torch.zeros_like(data))
                std = torch.sqrt((centered ** 2).sum(dim=-2, keepdim=True) / valid_count.clamp_min(1))

                mean = torch.where(valid_count > 0, mean, torch.zeros_like(mean))
                std = torch.where(
                    (valid_count > 0) & torch.isfinite(std) & (std > 0),
                    std,
                    torch.ones_like(std)
                )

                self.stats["mean"] = mean
                self.stats["std"] = std

            else:
                valid_count = valid_mask.sum()

                if valid_count.item() == 0:
                    self.stats["mean"] = torch.tensor(0.0, device=data.device, dtype=data.dtype)
                    self.stats["std"] = torch.tensor(1.0, device=data.device, dtype=data.dtype)
                else:
                    safe_data = torch.where(valid_mask, data, torch.zeros_like(data))
                    mean = safe_data.sum() / valid_count.clamp_min(1)

                    centered = torch.where(valid_mask, data - mean, torch.zeros_like(data))
                    std = torch.sqrt((centered ** 2).sum() / valid_count.clamp_min(1))

                    if not torch.isfinite(mean):
                        mean = torch.tensor(0.0, device=data.device, dtype=data.dtype)
                    if (not torch.isfinite(std)) or std.item() == 0:
                        std = torch.tensor(1.0, device=data.device, dtype=data.dtype)

                    self.stats["mean"] = mean
                    self.stats["std"] = std

    def transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mean = self.stats["mean"].to(input_data.device)
        std = self.stats["std"].to(input_data.device)

        normed_data = (input_data - mean) / std
        normed_data = torch.nan_to_num(normed_data, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            normed_data = torch.where(mask, normed_data, input_data)

        return normed_data

    def inverse_transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mean = self.stats["mean"].to(input_data.device)
        std = self.stats["std"].to(input_data.device)

        denormed_data = input_data * std + mean
        denormed_data = torch.nan_to_num(denormed_data, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            denormed_data = torch.where(mask, denormed_data, input_data)

        return denormed_data


class NaNSafeMinMaxScaler(BasicTSScaler):
    """
    NaN-safe Min-Max scaler.

    Rules:
        1. Only finite values participate in fitting.
        2. All-invalid channels fallback to min=0, range=1.
        3. Zero-range channels fallback to range=1.
    """

    def __init__(self, norm_each_channel: bool, rescale: bool, stats: dict = None):
        super().__init__(norm_each_channel, rescale, stats or {})

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        if self.stats:
            return

        if isinstance(data, np.ndarray):
            valid_mask = np.isfinite(data)

            if self.norm_each_channel:
                safe_min_data = np.where(valid_mask, data, np.inf)
                safe_max_data = np.where(valid_mask, data, -np.inf)

                data_min = np.min(safe_min_data, axis=-2, keepdims=True)
                data_max = np.max(safe_max_data, axis=-2, keepdims=True)
                data_range = data_max - data_min

                has_valid = valid_mask.any(axis=-2, keepdims=True)
                data_min = np.where(has_valid & np.isfinite(data_min), data_min, 0.0)
                data_range = np.where(has_valid & np.isfinite(data_range) & (data_range > 0), data_range, 1.0)

            else:
                valid_values = data[valid_mask]

                if valid_values.size == 0:
                    data_min = 0.0
                    data_range = 1.0
                else:
                    data_min = np.min(valid_values)
                    data_max = np.max(valid_values)
                    data_range = data_max - data_min

                    if not np.isfinite(data_min):
                        data_min = 0.0
                    if (not np.isfinite(data_range)) or data_range == 0:
                        data_range = 1.0

            self.stats["min"] = torch.tensor(data_min, dtype=torch.float32)
            self.stats["range"] = torch.tensor(data_range, dtype=torch.float32)

        else:
            valid_mask = torch.isfinite(data)

            if self.norm_each_channel:
                inf_tensor = torch.full_like(data, float("inf"))
                ninf_tensor = torch.full_like(data, float("-inf"))

                safe_min_data = torch.where(valid_mask, data, inf_tensor)
                safe_max_data = torch.where(valid_mask, data, ninf_tensor)

                data_min = torch.min(safe_min_data, dim=-2, keepdim=True).values
                data_max = torch.max(safe_max_data, dim=-2, keepdim=True).values
                data_range = data_max - data_min

                has_valid = valid_mask.any(dim=-2, keepdim=True)
                data_min = torch.where(has_valid & torch.isfinite(data_min), data_min, torch.zeros_like(data_min))
                data_range = torch.where(
                    has_valid & torch.isfinite(data_range) & (data_range > 0),
                    data_range,
                    torch.ones_like(data_range)
                )

            else:
                valid_values = data[valid_mask]

                if valid_values.numel() == 0:
                    data_min = torch.tensor(0.0, device=data.device, dtype=data.dtype)
                    data_range = torch.tensor(1.0, device=data.device, dtype=data.dtype)
                else:
                    data_min = torch.min(valid_values)
                    data_max = torch.max(valid_values)
                    data_range = data_max - data_min

                    if not torch.isfinite(data_min):
                        data_min = torch.tensor(0.0, device=data.device, dtype=data.dtype)
                    if (not torch.isfinite(data_range)) or data_range.item() == 0:
                        data_range = torch.tensor(1.0, device=data.device, dtype=data.dtype)

            self.stats["min"] = data_min
            self.stats["range"] = data_range

    def transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        data_min = self.stats["min"].to(input_data.device)
        data_range = self.stats["range"].to(input_data.device)

        normed_data = (input_data - data_min) / data_range
        normed_data = torch.nan_to_num(normed_data, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            normed_data = torch.where(mask, normed_data, input_data)

        return normed_data

    def inverse_transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        data_min = self.stats["min"].to(input_data.device)
        data_range = self.stats["range"].to(input_data.device)

        denormed_data = input_data * data_range + data_min
        denormed_data = torch.nan_to_num(denormed_data, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            denormed_data = torch.where(mask, denormed_data, input_data)

        return denormed_data

