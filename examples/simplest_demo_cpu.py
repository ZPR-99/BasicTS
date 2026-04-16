from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DLinear import DLinear, DLinearConfig


def main():

    model_config = DLinearConfig(input_len=336, output_len=336)

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="ETTh1",
        dataset_params={"data_file_path": r"D:\云天化\软仪表\BasicTS\datasets\ETTh1"},
        gpus=None#"0"
    ))


if __name__ == "__main__":
    main()
