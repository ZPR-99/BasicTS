import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.data import BasicTSTabularDataset
from basicts.models.DLinear import DLinear, DLinearConfig


# =========================================================
# 基础配置
# =========================================================
DATA_DIR = r"D:\云天化\软仪表\BasicTS\datasets\二期磷酸\深度学习\SO3"
CKPT_DIR = r"D:\云天化\软仪表\BasicTS\checkpoints\phosphate_dlinear\SO3"
GPUS = "0" if torch.cuda.is_available() else None


# =========================================================
# 运行前检查
# =========================================================
def load_meta(data_dir):
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"缺少 meta.json: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def check_dataset_files(data_dir):
    required_files = [
        "train_data.npy", "train_label.npy", "train_timestamps.npy",
        "val_data.npy", "val_label.npy", "val_timestamps.npy",
        "test_data.npy", "test_label.npy", "test_timestamps.npy",
        "meta.json",
    ]
    for name in required_files:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少数据文件: {path}")


def check_nan(data_dir):
    check_files = [
        "train_data.npy", "train_label.npy",
        "val_data.npy", "val_label.npy",
        "test_data.npy", "test_label.npy",
    ]
    for name in check_files:
        path = os.path.join(data_dir, name)
        arr = np.load(path)
        if np.isnan(arr).any():
            raise ValueError(f"{name} 中存在 NaN，请先处理缺失值后再训练 DLinear。")


# =========================================================
# DLinear backbone + 回归头
# =========================================================
@dataclass
class DLinearRegressorConfig:
    input_len: int = 1
    output_len: int = 1
    num_features: int = 1
    target_dim: int = 1
    moving_avg: int = 1
    stride: int = 1
    individual: bool = False


class DLinearRegressor(nn.Module):
    """
    适配当前 tabular 风格输入的 DLinear 回归模型：
    inputs  : [B, 1, F]
    outputs : [B, 1, 1]

    做法：
    1. 官方 DLinear 先输出 [B, 1, F]
    2. 再通过线性头把最后一维 F 压到 1
    """

    def __init__(self, model_config):
        super().__init__()

        backbone_config = DLinearConfig(
            input_len=model_config.input_len,
            output_len=model_config.output_len,
            num_features=model_config.num_features,
            moving_avg=model_config.moving_avg,
            stride=model_config.stride,
            individual=model_config.individual,
        )

        self.backbone = DLinear(backbone_config)
        self.head = nn.Linear(model_config.num_features, model_config.target_dim)

    def forward(self, inputs, **kwargs):
        """
        inputs: [B, 1, F]
        backbone output: [B, 1, F]
        final output: [B, 1, 1]
        """
        x = self.backbone(inputs)
        x = self.head(x)
        return x


# =========================================================
# 构建训练配置
# =========================================================
def build_cfg():
    check_dataset_files(DATA_DIR)
    check_nan(DATA_DIR)

    meta = load_meta(DATA_DIR)
    num_features = int(meta["num_features"])

    model_config = DLinearRegressorConfig(
        input_len=1,
        output_len=1,
        num_features=num_features,
        target_dim=1,
        moving_avg=1,
        stride=1,
        individual=False,
    )

    cfg = BasicTSForecastingConfig(
        model=DLinearRegressor,
        model_config=model_config,

        dataset_name="SO3",
        dataset_type=BasicTSTabularDataset,
        dataset_params={
            "dataset_name": "SO3",
            "data_file_path": DATA_DIR,
            "memmap": False,
            "use_timestamps": True,
            "with_meta_check": True,
        },

        gpus=GPUS,
        seed=42,

        # 不再让 BasicTS 默认去切窗口
        input_len=1,
        output_len=1,

        # 先关闭 scaler，避免对你当前自定义特征再次做默认归一化时引入不确定性
        scaler=None,
        null_val=0.0,

        num_epochs=100,

        train_batch_size=128,
        val_batch_size=256,
        test_batch_size=256,

        loss="MSE",
        metrics=["RMSE", "MAE", "MAPE", "R2"],
        target_metric="RMSE",
        best_metric="min",

        optimizer=torch.optim.Adam,
        optimizer_params={
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },

        lr_scheduler=None,
        lr_scheduler_params=None,

        val_interval=1,
        test_interval=1,
        eval_after_train=True,
        save_results=True,

        ckpt_save_dir=CKPT_DIR,
    )

    return cfg


# =========================================================
# 主函数
# =========================================================
def main():
    cfg = build_cfg()
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()

