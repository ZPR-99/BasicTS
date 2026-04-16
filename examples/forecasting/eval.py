import argparse
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "二期磷酸", "SO3")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "forecasting")

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.scaler import NaNSafeZScoreScaler


def build_parser():
    parser = argparse.ArgumentParser(description="BasicTS forecasting evaluation entry")

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
        help='GPU 设备，如 "0"、"0,1"；CPU 评估可不传',
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
        help="待评估 checkpoint 路径",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="checkpoint 根目录，仅用于配置构造",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="是否保存 test_results 下的 inputs/prediction/targets.npy",
    )

    parser.set_defaults(rescale=True)
    parser.add_argument(
        "--no-rescale",
        dest="rescale",
        action="store_false",
        help="关闭反归一化。默认开启，建议保持开启以输出原始量纲结果。",
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

        save_results=args.save_results,
        eval_after_train=False,

        ckpt_save_dir=args.ckpt_dir,
    )

    return cfg


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

    cfg = build_config(args)
    BasicTSLauncher.launch_evaluation(
        cfg=cfg,
        ckpt_path=args.ckpt_path,
        gpus=args.gpus,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

    