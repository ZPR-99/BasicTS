"""
LightTS 单目标预测训练脚本（最终替换版）
- 完全独立，不依赖 BasicTS runner
- 保留当前单目标监督训练范式：历史全特征 -> 未来目标列
- 使用 BasicTS 源码目录中的 LightTS 作为 backbone
- 自动尝试多种 LightTS 导入路径 / 类名 / 输入布局 / 输出布局
- checkpoint 自洽：保存/加载 scaler、模型元信息、LightTS 配置、输出通道信息
- 按验证集最优指标选模，离线推理生成趋势图与 R2
- 兼容 PyTorch 2.6+：torch.load(..., weights_only=False)
"""

import csv
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 全局配置
# =============================================================================

CONFIG = {
    # ── BasicTS 源码根目录（用于导入 LightTS） ──
    "basicts_src_root": r"D:\云天化\软仪表\BasicTS\src",

    # ── 数据路径（与单目标数据导出脚本保持一致） ──
    "single_target_output_root": r"D:\云天化\软仪表\BasicTS\datasets\二期磷酸",
    "single_target_dataset_root_name": "dap_all_targets",

    # ── 训练目标列表（逐行取消注释即可启用） ──
    "target_cols": [
        "SO3",
        # "LF302_OCMJ",
        # "LF302_F4JL",
        # "LF302_DA43",
        # "LF302_S2YO",
    ],

    # ── 超参数搜索空间 ──
    "input_lens":  [24, 48, 96, 168],
    "output_lens": [12, 24, 48, 96],
    "stride": 1,

    # ── 模型类型 ──
    "model_name": "LightTSSingleTarget",

    # ── LightTS 导入候选（按顺序尝试） ──
    "lightts_import_candidates": [
        ("basicts.models.LightTS", "LightTS"),
        ("basicts.models.LightTS", "Model"),
        ("basicts.models.LightTS.arch", "LightTS"),
        ("basicts.models.LightTS.arch", "Model"),
        ("basicts.models.LightTS.model", "LightTS"),
        ("basicts.models.LightTS.model", "Model"),
        ("basicts.models.LightTS.lightts", "LightTS"),
        ("basicts.models.LightTS.lightts", "Model"),
    ],

    # ── LightTS 初始化参数（用于构造 config） ──
    # 会与自动补齐字段合并后传给 config 对象
    "lightts_init_kwargs": {
        "hidden_size": 64,
        "d_model": 64,
        "dropout": 0.1,
        "e_layers": 2,
        "factor": 3,
        "chunk_size": 24,
        "window_size": 24,
    },

    # ── 训练超参数 ──
    "seed": 233,
    "num_epochs": 100,
    "batch_size": 64,
    "eval_batch_size": 256,

    "lr": 5e-4,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,

    "milestones": [25, 50],
    "gamma": 0.5,
    "early_stopping_patience": 10,

    # ── 设备 ──
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ── 输出路径 ──
    "checkpoint_root": r"D:\云天化\软仪表\BasicTS\examples\forecasting\checkpoints_single_target_lightts",
    "best_run_root": r"D:\云天化\软仪表\BasicTS\best_runs_lightts_single_target",

    # ── 模型选择策略（用验证集，禁止用测试集选模） ──
    "select_metric": "val_MAE",
    "select_metric_mode": "min",

    # ── 特征缺失值填充策略 ──
    "feature_nan_strategy": "ffill_then_train_median",

    # ── 实验管理 ──
    "remove_non_best_runs": True,
    "copy_best_run_to_best_root": True,

    # ── 离线分析（加载最优 checkpoint，对 val/test 生成趋势图与指标） ──
    "enable_best_run_offline_analysis": True,
    "offline_analysis_splits": ["val", "test"],
    "offline_plot_dpi": 160,
    "offline_plot_max_points": None,
    "mape_eps": 1e-6,
}


# =============================================================================
# 工具函数
# =============================================================================

def log_info(msg):
    print(msg)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def remove_dir_if_exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def sanitize_name(name):
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(name).strip())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def format_metric(v, digits=6):
    if v is None:
        return "nan"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "nan"


def add_basicts_src_to_path():
    src_root = CONFIG["basicts_src_root"]
    if src_root and src_root not in sys.path:
        sys.path.insert(0, src_root)


# =============================================================================
# 数据集路径管理
# =============================================================================

def get_dataset_root_dir():
    return os.path.join(
        CONFIG["single_target_output_root"],
        CONFIG["single_target_dataset_root_name"],
    )


def build_dataset_name(current_target):
    return f"{CONFIG['single_target_dataset_root_name']}_{sanitize_name(current_target)}"


def get_single_target_dataset_dir(current_target):
    dataset_root_dir = get_dataset_root_dir()
    dataset_name = build_dataset_name(current_target)
    dataset_dir = os.path.join(dataset_root_dir, sanitize_name(current_target))
    return dataset_name, dataset_dir


# =============================================================================
# 数据集校验与加载
# =============================================================================

def validate_single_target_dataset(current_target):
    dataset_name, dataset_dir = get_single_target_dataset_dir(current_target)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"未找到单目标数据集目录: {dataset_dir}\n"
            f"请先运行单目标数据导出脚本。"
        )

    required_files = [
        "train_data.npy",
        "val_data.npy",
        "test_data.npy",
        "meta.json",
        "columns.json",
        "split_info.json",
    ]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(dataset_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"单目标数据集目录缺少文件: dataset_dir={dataset_dir}, missing={missing}"
        )

    meta = load_json(os.path.join(dataset_dir, "meta.json"))
    columns_info = load_json(os.path.join(dataset_dir, "columns.json"))
    split_info = load_json(os.path.join(dataset_dir, "split_info.json"))

    dataset_target = columns_info.get("target_col")
    if dataset_target is None:
        raise ValueError(f"{dataset_dir}/columns.json 缺少字段 'target_col'")
    if dataset_target != current_target:
        raise ValueError(
            f"目标不匹配: 请求目标={current_target}, "
            f"数据集内目标={dataset_target}, dataset_dir={dataset_dir}"
        )

    target_index = columns_info.get("target_index")
    if target_index is None:
        raise ValueError(f"{dataset_dir}/columns.json 缺少字段 'target_index'")

    data_columns = columns_info.get("data_columns", [])
    if not data_columns:
        raise ValueError(f"{dataset_dir}/columns.json 缺少字段 'data_columns'")

    num_features = int(meta.get("num_vars", len(data_columns)))

    return {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "target_col": dataset_target,
        "target_index": int(target_index),
        "num_features": num_features,
        "meta": meta,
        "columns_info": columns_info,
        "split_info": split_info,
        "train_samples": int(split_info.get("train_samples", 0)),
        "val_samples": int(split_info.get("val_samples", 0)),
        "test_samples": int(split_info.get("test_samples", 0)),
        "data_columns": data_columns,
    }


def load_split_arrays(dataset_dir):
    return {
        split: np.load(os.path.join(dataset_dir, f"{split}_data.npy")).astype(np.float32)
        for split in ("train", "val", "test")
    }


# =============================================================================
# 缺失值预处理
# =============================================================================

def forward_fill_2d(arr_2d):
    if arr_2d.size == 0:
        return arr_2d.astype(np.float32)
    return pd.DataFrame(arr_2d).ffill().to_numpy(dtype=np.float32)


def fill_nan_with_values(arr_2d, fill_values):
    out = arr_2d.copy()
    nan_mask = np.isnan(out)
    if nan_mask.any():
        out[nan_mask] = np.take(fill_values, np.where(nan_mask)[1])
    return out.astype(np.float32)


def preprocess_split_arrays(split_arrays, target_index):
    num_vars = split_arrays["train"].shape[1]
    feature_indices = [i for i in range(num_vars) if i != int(target_index)]

    for split_name, arr in split_arrays.items():
        if np.isnan(arr[:, int(target_index)]).any():
            raise ValueError(
                f"{split_name}_data.npy 的目标列（index={target_index}）存在 NaN，"
                f"请检查数据导出逻辑。"
            )

    train_feature_block = forward_fill_2d(split_arrays["train"][:, feature_indices])
    train_medians = np.nanmedian(train_feature_block, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians).astype(np.float32)

    processed = {}
    nan_report = {}
    strategy = CONFIG["feature_nan_strategy"]

    for split_name, arr in split_arrays.items():
        out = arr.copy()
        feature_block = out[:, feature_indices]
        before_nan = int(np.isnan(feature_block).sum())

        if strategy == "ffill_then_train_median":
            feature_block = forward_fill_2d(feature_block)
            feature_block = fill_nan_with_values(feature_block, train_medians)
        else:
            raise ValueError(f"不支持的 feature_nan_strategy: {strategy}")

        after_nan = int(np.isnan(feature_block).sum())
        out[:, feature_indices] = feature_block
        processed[split_name] = out.astype(np.float32)

        nan_report[split_name] = {
            "feature_nan_before_fill": before_nan,
            "feature_nan_after_fill": after_nan,
        }

    for split_name, arr in processed.items():
        total_nan = int(np.isnan(arr).sum())
        if total_nan > 0:
            raise ValueError(
                f"{split_name} 预处理后仍存在 NaN，总数={total_nan}，请检查填充逻辑。"
            )

    return processed, train_medians, nan_report


# =============================================================================
# Scaler（ZScore）
# =============================================================================

def fit_input_scaler(train_arr):
    mean = np.mean(train_arr, axis=0).astype(np.float32)
    std = np.std(train_arr, axis=0).astype(np.float32)
    std = np.where(np.abs(std) < 1e-12, 1.0, std).astype(np.float32)
    return mean, std


def fit_target_scaler(train_arr, target_index):
    target = train_arr[:, int(target_index)].astype(np.float64)
    mean = float(np.mean(target))
    std = float(np.std(target))
    if abs(std) < 1e-12:
        std = 1.0
    return mean, std


def normalize_data_arr(data_arr, x_mean, x_std):
    return ((data_arr - x_mean) / x_std).astype(np.float32)


def inverse_transform_target(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


# =============================================================================
# 滑窗构建
# =============================================================================

def calc_num_windows(split_len, input_len, output_len, stride):
    if split_len < input_len + output_len:
        return 0
    return int((split_len - input_len - output_len) // stride + 1)


def evaluate_window_combo(dataset_info, input_len, output_len):
    stride = int(CONFIG["stride"])
    window_info = {
        split: calc_num_windows(dataset_info[f"{split}_samples"], input_len, output_len, stride)
        for split in ("train", "val", "test")
    }
    is_valid = all(v > 0 for v in window_info.values())
    return is_valid, {f"{k}_windows": v for k, v in window_info.items()}


def build_windows(data_arr, target_index, input_len, output_len, stride):
    total = len(data_arr)
    max_start = total - input_len - output_len
    if max_start < 0:
        raise ValueError(
            f"无法构造滑窗样本: total={total}, "
            f"input_len={input_len}, output_len={output_len}"
        )

    x_list, y_list = [], []
    for start in range(0, max_start + 1, stride):
        x_list.append(data_arr[start: start + input_len, :])
        y_list.append(
            data_arr[
                start + input_len: start + input_len + output_len,
                int(target_index)
            ]
        )

    return (
        np.stack(x_list, axis=0).astype(np.float32),
        np.stack(y_list, axis=0).astype(np.float32),
    )


# =============================================================================
# 指标计算
# =============================================================================

def calc_basic_metrics_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]

    if len(y_true) == 0:
        return {
            "MSE": None,
            "MAE": None,
            "RMSE": None,
            "MAPE": None,
            "R2": None,
            "count": 0
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) /
                                np.maximum(np.abs(y_true), CONFIG["mape_eps"]))))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": float(r2),
        "count": int(len(y_true)),
    }


# =============================================================================
# LightTS 动态导入与包装
# =============================================================================

def try_import_lightts_class(import_module=None, import_class=None):
    add_basicts_src_to_path()

    if import_module is not None and import_class is not None:
        module = importlib.import_module(import_module)
        if not hasattr(module, import_class):
            raise ImportError(f"指定导入失败：{import_module}.{import_class} 不存在")
        return getattr(module, import_class), import_module, import_class

    errors = []
    for module_name, class_name in CONFIG["lightts_import_candidates"]:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                model_cls = getattr(module, class_name)
                return model_cls, module_name, class_name
            errors.append(f"{module_name}.{class_name}: 类不存在")
        except Exception as e:
            errors.append(f"{module_name}.{class_name}: {repr(e)}")

    raise ImportError(
        "未能导入 LightTS，请检查本地 BasicTS 源码结构。\n"
        + "\n".join(errors)
    )


class LightTSConfigLike:
    """
    提供 config.xxx 访问；未知字段使用缺省值兜底。
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        fallback_map = {
            "task_name": "long_term_forecast",

            # 长度相关：这次必须补 input_len/output_len 别名
            "seq_len": 24,
            "input_len": 24,
            "label_len": 0,
            "pred_len": 12,
            "output_len": 12,
            "forecasting_horizon": 12,
            "horizon": 12,

            # 通道相关
            "enc_in": 1,
            "dec_in": 1,
            "c_in": 1,
            "c_out": 1,
            "num_features": 1,
            "channels": 1,
            "feature_dim": 1,

            # 模型维度
            "hidden_size": 64,
            "d_model": 64,
            "d_ff": 64,

            # 网络结构
            "dropout": 0.1,
            "e_layers": 2,
            "num_layers": 2,
            "factor": 3,

            # LightTS 常见块参数
            "chunk_size": 24,
            "window_size": 24,

            # 其他常见字段
            "moving_avg": 25,
            "kernel_size": 25,
            "decomp_kernel": 25,
            "embed": "timeF",
            "freq": "h",
            "activation": "gelu",
            "individual": False,
            "top_k": 5,
            "num_class": 1,
            "features": "M",
            "target": "OT",
        }
        if name in fallback_map:
            return fallback_map[name]
        raise AttributeError(f"'LightTSConfigLike' object has no attribute '{name}'")


def build_lightts_config(num_features, input_len, output_len, resolved_config=None):
    if resolved_config is not None:
        config_dict = dict(resolved_config)
    else:
        user_kwargs = CONFIG.get("lightts_init_kwargs", {})
        config_dict = {
            "task_name": "long_term_forecast",

            # 长度字段：同时补齐多组别名
            "seq_len": int(input_len),
            "input_len": int(input_len),
            "label_len": 0,
            "pred_len": int(output_len),
            "output_len": int(output_len),
            "forecasting_horizon": int(output_len),
            "horizon": int(output_len),

            # 通道字段：同时补齐多组别名
            "enc_in": int(num_features),
            "dec_in": int(num_features),
            "c_in": int(num_features),
            "c_out": int(num_features),
            "num_features": int(num_features),
            "channels": int(num_features),
            "feature_dim": int(num_features),

            # 模型维度
            "hidden_size": int(user_kwargs.get("hidden_size", 64)),
            "d_model": int(user_kwargs.get("d_model", user_kwargs.get("hidden_size", 64))),
            "d_ff": int(user_kwargs.get("d_ff", user_kwargs.get("d_model", user_kwargs.get("hidden_size", 64)))),

            # 网络结构
            "dropout": float(user_kwargs.get("dropout", 0.1)),
            "e_layers": int(user_kwargs.get("e_layers", 2)),
            "num_layers": int(user_kwargs.get("num_layers", user_kwargs.get("e_layers", 2))),
            "factor": int(user_kwargs.get("factor", 3)),

            # LightTS 常见块参数
            "chunk_size": int(user_kwargs.get("chunk_size", min(int(input_len), 24))),
            "window_size": int(user_kwargs.get("window_size", min(int(input_len), 24))),

            # 其他常见字段
            "moving_avg": int(user_kwargs.get("moving_avg", 25)),
            "kernel_size": int(user_kwargs.get("kernel_size", 25)),
            "decomp_kernel": int(user_kwargs.get("decomp_kernel", 25)),
            "embed": user_kwargs.get("embed", "timeF"),
            "freq": user_kwargs.get("freq", "h"),
            "activation": user_kwargs.get("activation", "gelu"),
            "individual": bool(user_kwargs.get("individual", False)),
            "top_k": int(user_kwargs.get("top_k", 5)),
            "num_class": int(user_kwargs.get("num_class", 1)),
            "features": user_kwargs.get("features", "M"),
            "target": user_kwargs.get("target", "OT"),
        }
        config_dict.update(user_kwargs)

    return LightTSConfigLike(**config_dict), config_dict


def try_build_lightts_backbone(model_cls, num_features, input_len, output_len, resolved_config=None):
    config_obj, config_dict = build_lightts_config(
        num_features=num_features,
        input_len=input_len,
        output_len=output_len,
        resolved_config=resolved_config
    )

    try:
        backbone = model_cls(config_obj)
        return backbone, {
            "init_mode": "config",
            "resolved_config": config_dict,
        }
    except Exception as e:
        raise RuntimeError(
            "LightTS 实例化失败。\n"
            f"config={config_dict}\n"
            f"error={repr(e)}"
        )


class LightTSSingleTarget(nn.Module):
    """
    包装 LightTS backbone：
    - 自动尝试输入布局 BLC / BCL
    - 自动识别输出布局 [B,L] / [B,L,C] / [B,C,L] / [B,L*C]
    - 多通道输出时使用 channel_projection 聚合为单目标
    - channel_projection 的输入维度在首次成功前向后确定，并保存到 checkpoint
    """

    def __init__(
        self,
        num_features,
        input_len,
        output_len,
        import_module=None,
        import_class=None,
        resolved_config=None,
        resolved_input_layout=None,
        resolved_output_layout=None,
        projection_in_features=None
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.input_len = int(input_len)
        self.output_len = int(output_len)

        model_cls, resolved_module, resolved_class = try_import_lightts_class(
            import_module=import_module,
            import_class=import_class
        )
        self.backbone, build_meta = try_build_lightts_backbone(
            model_cls=model_cls,
            num_features=self.num_features,
            input_len=self.input_len,
            output_len=self.output_len,
            resolved_config=resolved_config
        )

        self.import_module = resolved_module
        self.import_class = resolved_class
        self.resolved_config = build_meta["resolved_config"]

        self._resolved_input_layout = resolved_input_layout
        self._resolved_output_layout = resolved_output_layout

        self.channel_projection = None
        if projection_in_features is not None:
            self.channel_projection = nn.Linear(int(projection_in_features), 1)

    def _extract_tensor_output(self, out):
        if torch.is_tensor(out):
            return out

        if isinstance(out, (list, tuple)) and len(out) > 0:
            for item in out:
                if torch.is_tensor(item):
                    return item

        if isinstance(out, dict):
            for k in ["prediction", "pred", "output", "out", "forecast"]:
                if k in out and torch.is_tensor(out[k]):
                    return out[k]
            for _, v in out.items():
                if torch.is_tensor(v):
                    return v

        raise ValueError(f"LightTS 输出无法解析为 tensor，type={type(out)}")

    def _call_backbone(self, x):
        errors = []

        if self._resolved_input_layout == "BLC":
            try:
                return self._extract_tensor_output(self.backbone(x))
            except Exception as e:
                errors.append(f"BLC缓存失败: {repr(e)}")

        if self._resolved_input_layout == "BCL":
            try:
                return self._extract_tensor_output(self.backbone(x.permute(0, 2, 1)))
            except Exception as e:
                errors.append(f"BCL缓存失败: {repr(e)}")

        try:
            out = self._extract_tensor_output(self.backbone(x))
            self._resolved_input_layout = "BLC"
            return out
        except Exception as e:
            errors.append(f"BLC尝试失败: {repr(e)}")

        try:
            out = self._extract_tensor_output(self.backbone(x.permute(0, 2, 1)))
            self._resolved_input_layout = "BCL"
            return out
        except Exception as e:
            errors.append(f"BCL尝试失败: {repr(e)}")

        raise RuntimeError(
            "LightTS backbone 前向失败，无法确定输入布局。\n" + "\n".join(errors)
        )

    def _ensure_projection(self, in_features, device, dtype):
        in_features = int(in_features)

        if self.channel_projection is None:
            self.channel_projection = nn.Linear(in_features, 1).to(device=device, dtype=dtype)
            return

        if self.channel_projection.in_features != in_features:
            raise ValueError(
                f"channel_projection 输入维度与当前 backbone 输出不一致："
                f"projection.in_features={self.channel_projection.in_features}, "
                f"actual={in_features}"
            )

    def _normalize_output(self, out):
        # [B, L]
        if out.dim() == 2 and out.shape[1] == self.output_len:
            self._resolved_output_layout = "BL"
            return out

        # [B, L*C]，尝试 reshape
        if out.dim() == 2 and out.shape[1] == self.output_len * self.num_features:
            self._resolved_output_layout = "BLC_FLAT"
            return out.reshape(out.shape[0], self.output_len, self.num_features)

        if out.dim() != 3:
            raise ValueError(f"LightTS 输出维度异常: shape={tuple(out.shape)}")

        # [B, L, C]
        if out.shape[1] == self.output_len:
            self._resolved_output_layout = "BLC"
            return out

        # [B, C, L]
        if out.shape[2] == self.output_len:
            self._resolved_output_layout = "BCL"
            return out.permute(0, 2, 1)

        raise ValueError(
            f"LightTS 输出 shape 无法识别：got={tuple(out.shape)}, "
            f"output_len={self.output_len}, num_features={self.num_features}"
        )

    def forward(self, x):
        out = self._call_backbone(x)
        out = self._normalize_output(out)

        if out.dim() == 2:
            return out

        if out.shape[2] == 1:
            return out.squeeze(-1)

        self._ensure_projection(out.shape[2], out.device, out.dtype)
        return self.channel_projection(out).squeeze(-1)

    def get_model_meta(self):
        return {
            "import_module": self.import_module,
            "import_class": self.import_class,
            "resolved_config": self.resolved_config,
            "resolved_input_layout": self._resolved_input_layout,
            "resolved_output_layout": self._resolved_output_layout,
            "projection_in_features": None if self.channel_projection is None else int(self.channel_projection.in_features),
        }


# =============================================================================
# 模型工厂
# =============================================================================

def build_model(model_name, num_features, input_len, output_len, model_meta=None):
    if model_name == "LightTSSingleTarget":
        model_meta = model_meta or {}
        return LightTSSingleTarget(
            num_features=num_features,
            input_len=input_len,
            output_len=output_len,
            import_module=model_meta.get("import_module"),
            import_class=model_meta.get("import_class"),
            resolved_config=model_meta.get("resolved_config"),
            resolved_input_layout=model_meta.get("resolved_input_layout"),
            resolved_output_layout=model_meta.get("resolved_output_layout"),
            projection_in_features=model_meta.get("projection_in_features"),
        )

    raise ValueError(f"不支持的 model_name: {model_name}")


# =============================================================================
# DataLoader 构建
# =============================================================================

def create_data_loaders(processed_arrays, target_index, input_len, output_len):
    # x 使用归一化后的全特征窗口
    # y 使用原始目标列窗口，再单独按目标 scaler 做归一化
    x_mean, x_std = fit_input_scaler(processed_arrays["train"])
    y_mean, y_std = fit_target_scaler(processed_arrays["train"], target_index)

    stride = int(CONFIG["stride"])
    loaders = {}

    for split_name, arr in processed_arrays.items():
        x_norm_arr = normalize_data_arr(arr, x_mean, x_std)
        x_arr, _ = build_windows(x_norm_arr, target_index, input_len, output_len, stride)

        _, y_raw_arr = build_windows(arr, target_index, input_len, output_len, stride)
        y_arr_norm = ((y_raw_arr - y_mean) / y_std).astype(np.float32)

        ds = TensorDataset(
            torch.from_numpy(x_arr).float(),
            torch.from_numpy(y_arr_norm).float()
        )

        batch_size = CONFIG["batch_size"] if split_name == "train" else CONFIG["eval_batch_size"]
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            drop_last=False,
        )

    scaler_stats = {
        # 转成 list，减少序列化兼容问题
        "x_mean": x_mean.astype(np.float32).tolist(),
        "x_std": x_std.astype(np.float32).tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }

    return loaders, scaler_stats


# =============================================================================
# 训练 / 评估
# =============================================================================

def run_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_count = 0.0, 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = torch.mean((pred - y_batch) ** 2)
        loss.backward()

        if CONFIG.get("grad_clip_norm") is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(CONFIG["grad_clip_norm"])
            )

        optimizer.step()

        n = len(x_batch)
        total_loss += float(loss.detach().cpu().item()) * n
        total_count += n

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device, y_mean, y_std):
    model.eval()
    norm_loss_sum, norm_loss_count = 0.0, 0
    preds, trues = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)
        norm_loss = torch.mean((pred - y_batch) ** 2)

        n = len(x_batch)
        norm_loss_sum += float(norm_loss.detach().cpu().item()) * n
        norm_loss_count += n

        pred_real = inverse_transform_target(pred.detach().cpu().numpy(), y_mean, y_std)
        true_real = inverse_transform_target(y_batch.detach().cpu().numpy(), y_mean, y_std)

        preds.append(pred_real)
        trues.append(true_real)

    pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 0), dtype=np.float32)
    true_arr = np.concatenate(trues, axis=0) if trues else np.zeros((0, 0), dtype=np.float32)

    metrics = calc_basic_metrics_np(true_arr.reshape(-1), pred_arr.reshape(-1))
    metrics["norm_MSE"] = norm_loss_sum / max(norm_loss_count, 1)
    return metrics


# =============================================================================
# 文件 I/O 工具
# =============================================================================

def build_run_dir(dataset_name, input_len, output_len):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_token = f"{timestamp}_{str(time.time_ns())[-6:]}"
    run_dir = os.path.join(
        CONFIG["checkpoint_root"],
        CONFIG["model_name"],
        dataset_name,
        f"L{input_len}_H{output_len}",
        run_token,
    )
    ensure_dir(run_dir)
    return run_dir


def save_training_history(history_rows, save_path):
    if not history_rows:
        return
    fieldnames = list(history_rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def write_summary_csv(rows, csv_path):
    ensure_dir(os.path.dirname(csv_path))
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 离线推理：趋势聚合与可视化
# =============================================================================

def aggregate_prediction_series(raw_split_arr, pred_windows, target_index, input_len, output_len, stride):
    total_len = len(raw_split_arr)
    actual_series = raw_split_arr[:, int(target_index)].astype(np.float64)

    pred_sum = np.zeros(total_len, dtype=np.float64)
    pred_count = np.zeros(total_len, dtype=np.int32)

    for w_idx, start in enumerate(range(0, total_len - input_len - output_len + 1, stride)):
        for h in range(output_len):
            pos = start + input_len + h
            pred_sum[pos] += float(pred_windows[w_idx, h])
            pred_count[pos] += 1

    valid_mask = pred_count > 0
    pred_series = np.full(total_len, np.nan, dtype=np.float64)
    pred_series[valid_mask] = pred_sum[valid_mask] / pred_count[valid_mask]

    return actual_series, pred_series, pred_count, valid_mask


def save_prediction_csv(time_axis, actual_series, pred_series, pred_count, valid_mask, save_path):
    pd.DataFrame({
        "time": time_axis,
        "actual": actual_series,
        "pred": pred_series,
        "pred_count": pred_count,
        "is_predicted_point": valid_mask.astype(int),
    }).to_csv(save_path, index=False, encoding="utf-8-sig")


def plot_prediction_trend(time_axis, actual_series, pred_series, valid_mask, target_col, title, save_path):
    plot_mask = valid_mask.copy()

    max_points = CONFIG.get("offline_plot_max_points")
    if max_points is not None:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > int(max_points):
            step = max(1, len(valid_indices) // int(max_points))
            sampled_mask = np.zeros_like(valid_mask, dtype=bool)
            sampled_mask[valid_indices[::step]] = True
            plot_mask = sampled_mask

    t = np.asarray(time_axis)
    plt.figure(figsize=(16, 5))
    plt.plot(t[plot_mask], actual_series[plot_mask], label="actual", linewidth=1.0)
    plt.plot(t[plot_mask], pred_series[plot_mask], label="pred", linewidth=1.0, alpha=0.85)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG["offline_plot_dpi"])
    plt.close()


def run_offline_analysis(best_row, dataset_info, processed_arrays):
    if not CONFIG["enable_best_run_offline_analysis"]:
        return best_row

    checkpoint_path = best_row.get("best_checkpoint")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        log_info(f"[警告] 未找到 best_checkpoint，跳过离线分析: {checkpoint_path}")
        return best_row

    ckpt = torch.load(
        checkpoint_path,
        map_location=CONFIG["device"],
        weights_only=False
    )

    input_len = int(ckpt["input_len"])
    output_len = int(ckpt["output_len"])
    target_index = int(ckpt["target_index"])
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])

    model = build_model(
        model_name=ckpt["model_name"],
        num_features=int(ckpt["num_features"]),
        input_len=input_len,
        output_len=output_len,
        model_meta=ckpt.get("model_meta"),
    ).to(CONFIG["device"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    analysis_dir = os.path.join(best_row["run_dir"], "offline_target_analysis")
    ensure_dir(analysis_dir)

    stride = int(CONFIG["stride"])
    summary = {
        "target": dataset_info["target_col"],
        "dataset_name": dataset_info["dataset_name"],
        "model_name": ckpt["model_name"],
        "input_len": input_len,
        "output_len": output_len,
        "model_meta": ckpt.get("model_meta"),
        "splits": {},
    }

    for split_name in CONFIG["offline_analysis_splits"]:
        raw_arr = processed_arrays[split_name].astype(np.float32)
        norm_arr = normalize_data_arr(raw_arr, x_mean, x_std)

        x_arr, _ = build_windows(
            data_arr=norm_arr,
            target_index=target_index,
            input_len=input_len,
            output_len=output_len,
            stride=stride,
        )

        ds = TensorDataset(torch.from_numpy(x_arr).float())
        loader = DataLoader(
            ds,
            batch_size=CONFIG["eval_batch_size"],
            shuffle=False,
            drop_last=False
        )

        pred_batches = []
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(CONFIG["device"])
                pred_norm = model(x_batch)
                pred_real = inverse_transform_target(
                    pred_norm.detach().cpu().numpy(), y_mean, y_std
                )
                pred_batches.append(pred_real.astype(np.float32))

        if not pred_batches:
            raise ValueError(
                f"离线分析时未生成预测窗口: split={split_name}, "
                f"input_len={input_len}, output_len={output_len}"
            )

        pred_windows = np.concatenate(pred_batches, axis=0)

        actual_series, pred_series, pred_count, valid_mask = aggregate_prediction_series(
            raw_split_arr=raw_arr,
            pred_windows=pred_windows,
            target_index=target_index,
            input_len=input_len,
            output_len=output_len,
            stride=stride,
        )

        metrics = calc_basic_metrics_np(actual_series[valid_mask], pred_series[valid_mask])

        metrics_path = os.path.join(analysis_dir, f"{split_name}_target_metrics.json")
        trend_csv_path = os.path.join(analysis_dir, f"{split_name}_target_trend.csv")
        trend_png_path = os.path.join(analysis_dir, f"{split_name}_target_trend.png")

        save_json(metrics_path, {
            "split_name": split_name,
            "input_len": input_len,
            "output_len": output_len,
            "metrics": metrics,
        })

        time_axis = np.arange(len(raw_arr))
        save_prediction_csv(time_axis, actual_series, pred_series, pred_count, valid_mask, trend_csv_path)

        title = (
            f"{dataset_info['target_col']} | {split_name} trend | "
            f"input_len={input_len}, output_len={output_len} | "
            f"R2={format_metric(metrics['R2'])}"
        )
        plot_prediction_trend(
            time_axis=time_axis,
            actual_series=actual_series,
            pred_series=pred_series,
            valid_mask=valid_mask,
            target_col=dataset_info["target_col"],
            title=title,
            save_path=trend_png_path,
        )

        summary["splits"][split_name] = {
            "metrics_path": metrics_path,
            "trend_csv_path": trend_csv_path,
            "trend_png_path": trend_png_path,
            "metrics": metrics,
        }

        for k, v in metrics.items():
            if k == "count":
                continue
            best_row[f"{split_name}_{k}"] = v
        best_row[f"{split_name}_metrics_path"] = metrics_path
        best_row[f"{split_name}_trend_csv"] = trend_csv_path
        best_row[f"{split_name}_trend_png"] = trend_png_path

        log_info(
            f"离线分析: target={dataset_info['target_col']}, split={split_name}, "
            f"R2={format_metric(metrics['R2'])}, MAE={format_metric(metrics['MAE'])}"
        )

    summary_path = os.path.join(analysis_dir, "offline_target_analysis_summary.json")
    save_json(summary_path, summary)
    best_row["offline_target_analysis_summary"] = summary_path
    return best_row


# =============================================================================
# 单次训练 + 结果收集
# =============================================================================

def train_and_collect_one_run(current_target, dataset_info, processed_arrays, input_len, output_len, window_info):
    set_seed(CONFIG["seed"])

    loaders, scaler_stats = create_data_loaders(
        processed_arrays=processed_arrays,
        target_index=dataset_info["target_index"],
        input_len=input_len,
        output_len=output_len,
    )

    run_dir = build_run_dir(dataset_info["dataset_name"], input_len, output_len)

    model = build_model(
        model_name=CONFIG["model_name"],
        num_features=dataset_info["num_features"],
        input_len=input_len,
        output_len=output_len,
        model_meta=None,
    ).to(CONFIG["device"])

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=CONFIG["milestones"],
        gamma=CONFIG["gamma"]
    )

    run_config = {
        "target": current_target,
        "dataset_name": dataset_info["dataset_name"],
        "dataset_dir": dataset_info["dataset_dir"],
        "model_name": CONFIG["model_name"],
        "input_len": input_len,
        "output_len": output_len,
        "stride": CONFIG["stride"],
        "lightts_init_kwargs": CONFIG["lightts_init_kwargs"],
        "batch_size": CONFIG["batch_size"],
        "eval_batch_size": CONFIG["eval_batch_size"],
        "lr": CONFIG["lr"],
        "weight_decay": CONFIG["weight_decay"],
        "grad_clip_norm": CONFIG.get("grad_clip_norm"),
        "milestones": CONFIG["milestones"],
        "gamma": CONFIG["gamma"],
        "early_stopping_patience": CONFIG["early_stopping_patience"],
        "device": CONFIG["device"],
        "target_col": dataset_info["target_col"],
        "target_index": dataset_info["target_index"],
        "num_features": dataset_info["num_features"],
    }
    save_json(os.path.join(run_dir, "run_config.json"), run_config)

    best_checkpoint_path = os.path.join(run_dir, "best.pt")
    last_checkpoint_path = os.path.join(run_dir, "last.pt")

    history_rows = []
    best_epoch = 0
    best_val_mae = math.inf
    patience_counter = 0

    def _save_checkpoint(path, extra):
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_name": CONFIG["model_name"],
            "input_len": int(input_len),
            "output_len": int(output_len),
            "num_features": int(dataset_info["num_features"]),
            "target_col": dataset_info["target_col"],
            "target_index": int(dataset_info["target_index"]),
            "x_mean": scaler_stats["x_mean"],
            "x_std": scaler_stats["x_std"],
            "y_mean": float(scaler_stats["y_mean"]),
            "y_std": float(scaler_stats["y_std"]),
            "model_meta": model.get_model_meta(),
            **extra,
        }
        torch.save(ckpt, path)

    for epoch in range(1, int(CONFIG["num_epochs"]) + 1):
        train_norm_mse = run_one_epoch(model, loaders["train"], optimizer, CONFIG["device"])
        val_metrics = evaluate_loader(
            model, loaders["val"], CONFIG["device"],
            scaler_stats["y_mean"], scaler_stats["y_std"]
        )
        test_metrics = evaluate_loader(
            model, loaders["test"], CONFIG["device"],
            scaler_stats["y_mean"], scaler_stats["y_std"]
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_norm_MSE": float(train_norm_mse),
            "val_norm_MSE": val_metrics["norm_MSE"],
            "val_MSE": val_metrics["MSE"],
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAPE": val_metrics["MAPE"],
            "val_R2": val_metrics["R2"],
            "test_MSE": test_metrics["MSE"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAPE": test_metrics["MAPE"],
            "test_R2": test_metrics["R2"],
        }
        history_rows.append(row)

        val_mae = val_metrics["MAE"]
        if val_mae is not None and float(val_mae) < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(best_checkpoint_path, {
                "best_epoch": int(best_epoch),
                "best_val_MAE": float(best_val_mae),
            })
        else:
            patience_counter += 1

        log_info(
            f"target={current_target} L{input_len}H{output_len} "
            f"epoch={epoch:03d} "
            f"train_norm_MSE={format_metric(train_norm_mse)} "
            f"val_MAE={format_metric(val_mae)} "
            f"test_MAE={format_metric(test_metrics['MAE'])}"
        )

        if patience_counter >= int(CONFIG["early_stopping_patience"]):
            log_info(
                f"早停触发: target={current_target}, L{input_len}H{output_len}, "
                f"best_epoch={best_epoch}, best_val_MAE={format_metric(best_val_mae)}"
            )
            break

    if not os.path.isfile(best_checkpoint_path):
        raise ValueError(
            f"训练结束后未生成 best checkpoint：{best_checkpoint_path}，"
            f"请检查验证指标是否异常。"
        )

    _save_checkpoint(last_checkpoint_path, {"last_epoch": len(history_rows)})
    save_training_history(history_rows, os.path.join(run_dir, "epoch_history.csv"))

    best_ckpt = torch.load(
        best_checkpoint_path,
        map_location=CONFIG["device"],
        weights_only=False
    )

    model = build_model(
        model_name=best_ckpt["model_name"],
        num_features=int(best_ckpt["num_features"]),
        input_len=int(best_ckpt["input_len"]),
        output_len=int(best_ckpt["output_len"]),
        model_meta=best_ckpt.get("model_meta"),
    ).to(CONFIG["device"])
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_metrics = evaluate_loader(
        model, loaders["val"], CONFIG["device"],
        scaler_stats["y_mean"], scaler_stats["y_std"]
    )
    test_metrics = evaluate_loader(
        model, loaders["test"], CONFIG["device"],
        scaler_stats["y_mean"], scaler_stats["y_std"]
    )

    def _strip_norm(m):
        return {k: v for k, v in m.items() if k != "norm_MSE"}

    save_json(os.path.join(run_dir, "val_metrics.json"), _strip_norm(val_metrics))
    save_json(os.path.join(run_dir, "test_metrics.json"), _strip_norm(test_metrics))
    save_json(os.path.join(run_dir, "data_check.json"), {
        "target": current_target,
        "dataset_dir": dataset_info["dataset_dir"],
        "target_col": dataset_info["target_col"],
        "target_index": dataset_info["target_index"],
        "num_features": dataset_info["num_features"],
        "window_info": window_info,
        "feature_nan_strategy": CONFIG["feature_nan_strategy"],
        "split_nan_report": dataset_info.get("nan_report", {}),
        "model_meta": best_ckpt.get("model_meta"),
    })

    return {
        "target": current_target,
        "dataset_name": dataset_info["dataset_name"],
        "input_len": input_len,
        "output_len": output_len,
        "selection_metric": CONFIG["select_metric"],
        "selection_score": float(val_metrics["MAE"]),
        "run_dir": run_dir,
        "best_checkpoint": best_checkpoint_path,
        "val_metrics_path": os.path.join(run_dir, "val_metrics.json"),
        "test_metrics_path": os.path.join(run_dir, "test_metrics.json"),
        "best_epoch": int(best_ckpt["best_epoch"]),
        "train_windows": window_info["train_windows"],
        "val_windows": window_info["val_windows"],
        "test_windows": window_info["test_windows"],
        "val_MSE": val_metrics["MSE"],
        "val_MAE": val_metrics["MAE"],
        "val_RMSE": val_metrics["RMSE"],
        "val_MAPE": val_metrics["MAPE"],
        "val_R2": val_metrics["R2"],
        "test_MSE": test_metrics["MSE"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_R2": test_metrics["R2"],
    }


# =============================================================================
# 最优实验管理
# =============================================================================

def pick_best_row(rows):
    if not rows:
        raise ValueError("rows 为空，无法选择最优实验")

    mode = CONFIG["select_metric_mode"].lower()
    if mode == "min":
        return min(rows, key=lambda x: x["selection_score"])
    if mode == "max":
        return max(rows, key=lambda x: x["selection_score"])

    raise ValueError(f"不支持的 select_metric_mode: {CONFIG['select_metric_mode']}")


def copy_best_run(best_row, current_target):
    if not CONFIG["copy_best_run_to_best_root"]:
        return None

    src_dir = best_row["run_dir"]
    dst_dir = os.path.join(CONFIG["best_run_root"], sanitize_name(current_target))
    remove_dir_if_exists(dst_dir)
    ensure_dir(os.path.dirname(dst_dir))
    shutil.copytree(src_dir, dst_dir)
    save_json(os.path.join(dst_dir, "best_run_info.json"), best_row)
    return dst_dir


def remove_non_best_runs(rows, best_row):
    if not CONFIG["remove_non_best_runs"]:
        return

    best_abs = os.path.abspath(best_row["run_dir"])
    for row in rows:
        run_abs = os.path.abspath(row["run_dir"])
        if run_abs == best_abs:
            continue
        remove_dir_if_exists(run_abs)
        log_info(f"已删除非最优实验目录: {run_abs}")


# =============================================================================
# 单目标训练主流程
# =============================================================================

def train_one_target(current_target):
    dataset_info = validate_single_target_dataset(current_target)
    split_arrays = load_split_arrays(dataset_info["dataset_dir"])
    processed_arrays, _, nan_report = preprocess_split_arrays(
        split_arrays=split_arrays,
        target_index=dataset_info["target_index"],
    )
    dataset_info["nan_report"] = nan_report

    log_info("=" * 80)
    log_info(f"开始训练目标: {current_target}")
    log_info(f"数据集目录: {dataset_info['dataset_dir']}")
    log_info(
        f"样本数 train/val/test = "
        f"{dataset_info['train_samples']}/"
        f"{dataset_info['val_samples']}/"
        f"{dataset_info['test_samples']}"
    )
    log_info(f"变量数(特征+目标): {dataset_info['num_features']}")
    log_info(f"缺失值处理报告: {nan_report}")
    log_info("=" * 80)

    all_rows = []
    skipped_combos = []

    for input_len in CONFIG["input_lens"]:
        for output_len in CONFIG["output_lens"]:
            is_valid, window_info = evaluate_window_combo(dataset_info, input_len, output_len)

            if not is_valid:
                skipped_combos.append({
                    "target": current_target,
                    "input_len": input_len,
                    "output_len": output_len,
                    **window_info,
                })
                log_info(
                    f"跳过非法组合: target={current_target}, "
                    f"input_len={input_len}, output_len={output_len}, "
                    f"window_info={window_info}"
                )
                continue

            row = train_and_collect_one_run(
                current_target=current_target,
                dataset_info=dataset_info,
                processed_arrays=processed_arrays,
                input_len=input_len,
                output_len=output_len,
                window_info=window_info,
            )
            all_rows.append(row)

    if not all_rows:
        raise ValueError(
            f"目标 {current_target} 没有任何合法的 input_len/output_len 组合。\n"
            f"train/val/test 样本数 = "
            f"{dataset_info['train_samples']}/"
            f"{dataset_info['val_samples']}/"
            f"{dataset_info['test_samples']}。\n"
            f"请缩小 input_lens/output_lens 或重新导出更长的数据集。"
        )

    summary_dir = os.path.join(dataset_info["dataset_dir"], "model_selection")
    ensure_dir(summary_dir)

    write_summary_csv(all_rows, os.path.join(summary_dir, "all_runs_summary.csv"))
    save_json(os.path.join(summary_dir, "skipped_combos.json"), skipped_combos)

    best_row = pick_best_row(all_rows)
    best_row = run_offline_analysis(best_row, dataset_info, processed_arrays)

    save_json(os.path.join(summary_dir, "best_run_summary.json"), best_row)

    best_copy_dir = copy_best_run(best_row, current_target)
    remove_non_best_runs(all_rows, best_row)

    log_info("=" * 80)
    log_info(f"目标 {current_target} 训练完成，最优结果如下:")
    log_info(
        f"最优参数: input_len={best_row['input_len']}, "
        f"output_len={best_row['output_len']}, "
        f"{best_row['selection_metric']}={format_metric(best_row['selection_score'])}"
    )
    if best_row.get("val_R2") is not None:
        log_info(f"val_R2  (原始量纲): {format_metric(best_row['val_R2'])}")
    if best_row.get("test_R2") is not None:
        log_info(f"test_R2 (原始量纲): {format_metric(best_row['test_R2'])}")
    log_info(f"最优实验目录: {best_row['run_dir']}")
    if best_copy_dir is not None:
        log_info(f"最优实验副本:  {best_copy_dir}")
    if best_row.get("offline_target_analysis_summary"):
        log_info(f"离线分析摘要:  {best_row['offline_target_analysis_summary']}")
    log_info("=" * 80)

    return {
        "target": current_target,
        "dataset_name": dataset_info["dataset_name"],
        "dataset_dir": dataset_info["dataset_dir"],
        "best_input_len": best_row["input_len"],
        "best_output_len": best_row["output_len"],
        "selection_metric": best_row["selection_metric"],
        "selection_score": best_row["selection_score"],
        "best_run_dir": best_row["run_dir"],
        "best_checkpoint": best_row["best_checkpoint"],
        "valid_combo_count": len(all_rows),
        "skipped_combo_count": len(skipped_combos),
        "val_MAE": best_row.get("val_MAE"),
        "val_R2": best_row.get("val_R2"),
        "test_MAE": best_row.get("test_MAE"),
        "test_R2": best_row.get("test_R2"),
        "offline_target_analysis_summary": best_row.get("offline_target_analysis_summary"),
    }


# =============================================================================
# 全目标遍历入口
# =============================================================================

def train_all_targets():
    all_target_rows = []
    failed_targets = []

    for target in CONFIG["target_cols"]:
        try:
            row = train_one_target(target)
            all_target_rows.append(row)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed_targets.append({"target": target, "error": str(e)})
            log_info(f"[失败] 目标 {target} 训练失败: {e}")

    summary_dir = os.path.join(
        CONFIG["single_target_output_root"],
        "lightts_single_target_training_summary",
    )
    ensure_dir(summary_dir)

    if all_target_rows:
        write_summary_csv(
            all_target_rows,
            os.path.join(summary_dir, "all_targets_best_summary.csv"),
        )

    save_json(
        os.path.join(summary_dir, "all_targets_best_summary.json"),
        {"success_targets": all_target_rows, "failed_targets": failed_targets},
    )

    log_info("=" * 80)
    log_info("全部目标训练完成。")
    log_info(f"成功目标数: {len(all_target_rows)}")
    log_info(f"失败目标数: {len(failed_targets)}")
    if failed_targets:
        log_info("失败详情:")
        for item in failed_targets:
            log_info(f"  target={item['target']}, error={item['error']}")
    log_info("=" * 80)


def main():
    set_seed(CONFIG["seed"])
    add_basicts_src_to_path()
    train_all_targets()


if __name__ == "__main__":
    main()

