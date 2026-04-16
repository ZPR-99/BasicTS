from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.data import BasicTSTabularDataset

from your_model import YourModel, YourModelConfig


def main():
    model_config = YourModelConfig(
        input_dim=300,   # 改成你的特征维度
        hidden_dim=128,
        output_dim=1,
    )

    cfg = BasicTSForecastingConfig(
        model=YourModel,
        model_config=model_config,

        dataset_name="二期磷酸/SO3",
        dataset_type=BasicTSTabularDataset,
        dataset_params={
            "dataset_name": "二期磷酸/SO3",
            "data_file_path": "datasets/二期磷酸/SO3",
            "memmap": False,
            "use_timestamps": True,
            "with_meta_check": True,
        },

        gpus="0",
        save_results=True,
        metrics=["RMSE", "MAE", "MAPE", "R2"],
        target_metric="RMSE",
        best_metric="min",
    )

    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()