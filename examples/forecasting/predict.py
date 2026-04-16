import argparse
import json
import os

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "二期磷酸", "SO3")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "forecasting")

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.scaler import NaNSafeZScoreScaler


def build_parser():
    parser = argparse.ArgumentParser(description="BasicTS forecasting prediction export entry")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="数据集目录，目录下应包含 train/val/test_data.npy、*_timestamps.npy、meta.json",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SO3",
        help="数据集名称，用于日志和结果目录命名",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=24,
        help="输入窗口长度",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="预测窗口长度",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='GPU 设备，如 "0"、"0,1"；CPU 导出可不传',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="评估 batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers，Windows 下建议 0",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="待导出预测结果的 checkpoint 路径",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="checkpoint 根目录，仅用于配置构造",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="导出的 CSV 文件路径；为空则默认写入 ckpt 同目录下的 test_results/predictions.csv",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过重新评估，直接读取现有 test_results/*.npy 生成 CSV",
    )

    parser.set_defaults(rescale=True)
    parser.add_argument(
        "--no-rescale",
        dest="rescale",
        action="store_false",
        help="关闭反归一化。默认开启，建议保持开启以导出原始量纲预测值。",
    )

    return parser


def check_dataset_dir(data_dir):
    required_files = [
        "train_data.npy",
        "val_data.npy",
        "test_data.npy",
        "train_timestamps.npy",
        "val_timestamps.npy",
        "test_timestamps.npy",
        "meta.json",
    ]

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    if missing_files:
        raise FileNotFoundError(
            f"数据集目录缺少必要文件: {missing_files}\n"
            f"当前目录: {data_dir}"
        )


def check_ckpt_path(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 文件不存在: {ckpt_path}")


def build_config(args):
    model_config = DLinearConfig(
        input_len=args.input_len,
        output_len=args.output_len,
    )

    cfg = BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name=args.dataset_name,

        gpus=args.gpus,
        seed=args.seed,

        input_len=args.input_len,
        output_len=args.output_len,
        use_timestamps=True,
        memmap=False,

        dataset_params={
            "data_file_path": args.data_dir,
            "memmap": False,
        },

        scaler=NaNSafeZScoreScaler,
        norm_each_channel=True,
        rescale=args.rescale,
        null_val=float("nan"),
        null_to_num=0.0,

        batch_size=args.batch_size,
        test_batch_size=args.batch_size,

        train_data_num_workers=args.num_workers,
        val_data_num_workers=args.num_workers,
        test_data_num_workers=args.num_workers,

        train_data_prefetch=False,
        val_data_prefetch=False,
        test_data_prefetch=False,

        save_results=True,
        eval_after_train=False,

        ckpt_save_dir=args.ckpt_dir,
    )

    return cfg


def load_meta(data_dir):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def run_evaluation(args):
    cfg = build_config(args)
    BasicTSLauncher.launch_evaluation(
        cfg=cfg,
        ckpt_path=args.ckpt_path,
        gpus=args.gpus,
        batch_size=args.batch_size,
    )


def build_prediction_dataframe(data_dir, result_dir):
    meta = load_meta(data_dir)

    prediction_path = os.path.join(result_dir, "prediction.npy")
    targets_path = os.path.join(result_dir, "targets.npy")

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"缺少 prediction.npy: {prediction_path}")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"缺少 targets.npy: {targets_path}")

    prediction = np.load(prediction_path)
    targets = np.load(targets_path)

    if prediction.shape != targets.shape:
        raise ValueError(
            f"prediction 与 targets 形状不一致: {prediction.shape} vs {targets.shape}"
        )

    sample_count = prediction.shape[0]
    horizon_count = prediction.shape[1]
    channel_count = prediction.shape[2]

    target_channel = meta["target_channel"][0]
    target_name = meta["target_name"]

    rows = []
    for sample_idx in range(sample_count):
        for horizon_idx in range(horizon_count):
            rows.append({
                "sample_index": sample_idx,
                "horizon_index": horizon_idx + 1,
                "y_true": float(targets[sample_idx, horizon_idx, target_channel]),
                "y_pred": float(prediction[sample_idx, horizon_idx, target_channel]),
            })

    df = pd.DataFrame(rows)

    source_csv_name = meta.get("files", {}).get("source_table_csv")
    source_csv_path = None
    if source_csv_name:
        source_csv_path = os.path.join(data_dir, source_csv_name)

    if source_csv_path and os.path.exists(source_csv_path):
        source_df = pd.read_csv(source_csv_path)
        timestamp_col = meta.get("timestamp_col", "Timestamp")

        if timestamp_col in source_df.columns:
            source_df[timestamp_col] = pd.to_datetime(source_df[timestamp_col], errors="coerce")

            train_len = meta["split"]["train_len"]
            val_len = meta["split"]["val_len"]
            test_len = meta["split"]["test_len"]
            input_len = meta["input_len"]
            output_len = meta["output_len"]

            test_start = train_len + val_len
            test_end = test_start + test_len

            test_source_df = source_df.iloc[test_start:test_end].reset_index(drop=True)

            timestamp_rows = []
            for sample_idx in range(sample_count):
                for horizon_idx in range(horizon_count):
                    target_local_idx = input_len + sample_idx + horizon_idx
                    if 0 <= target_local_idx < len(test_source_df):
                        ts_value = test_source_df.iloc[target_local_idx][timestamp_col]
                    else:
                        ts_value = pd.NaT

                    timestamp_rows.append({
                        "sample_index": sample_idx,
                        "horizon_index": horizon_idx + 1,
                        "Timestamp": ts_value,
                        "target_local_index_in_test_split": target_local_idx,
                        "target_global_index_in_full_series": test_start + target_local_idx,
                    })

            timestamp_df = pd.DataFrame(timestamp_rows)
            df = df.merge(timestamp_df, on=["sample_index", "horizon_index"], how="left")

    df["target_name"] = target_name
    df["channel_index"] = target_channel
    df["prediction_minus_true"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()

    ordered_cols = [
        "target_name",
        "channel_index",
        "sample_index",
        "horizon_index",
        "Timestamp",
        "target_local_index_in_test_split",
        "target_global_index_in_full_series",
        "y_true",
        "y_pred",
        "prediction_minus_true",
        "abs_error",
    ]
    existing_cols = [c for c in ordered_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    return df


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gpus is not None and str(args.gpus).strip() == "":
        args.gpus = None

    args.data_dir = os.path.abspath(args.data_dir)
    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    args.ckpt_path = os.path.abspath(args.ckpt_path)

    check_dataset_dir(args.data_dir)
    check_ckpt_path(args.ckpt_path)

    experiment_dir = os.path.dirname(args.ckpt_path)
    result_dir = os.path.join(experiment_dir, "test_results")

    if not args.skip_eval:
        run_evaluation(args)

    df = build_prediction_dataframe(args.data_dir, result_dir)

    if args.output_csv is None:
        output_csv = os.path.join(result_dir, "predictions.csv")
    else:
        output_csv = os.path.abspath(args.output_csv)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"预测结果已导出: {output_csv}")
    print(f"结果行数: {len(df)}")
    print(f"评估结果目录: {result_dir}")


if __name__ == "__main__":
    main()
    