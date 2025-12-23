import os
import numpy as np
import pandas as pd


# =========================
# 配置参数
# =========================
INPUT_FILE = r"D:\云天化\软仪表\天安\二期磷酸\model\date\raw_date\Laboratory_data.csv"
OUTPUT_DIR = r"D:\云天化\软仪表\天安\二期磷酸\model\date\raw_date"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Laboratory_cleaned.csv")

TIMESTAMP_COL = "Timestamp"

CONFIG = {
    # 基础清洗
    "treat_zero_as_nan": True,
    "drop_all_null_columns": True,
    "drop_duplicate_timestamp": True,

    # 1) 滚动均值 3-Sigma 清洗
    "enable_rolling_sigma": True,
    "sigma_window": "24h",
    "sigma_min_periods": 5,
    "sigma_multiplier": 3,

    # 2) 全局 IQR 清洗
    "enable_iqr": True,
    "iqr_multiplier": 2.5,

    # 3) 全局中位数百分比范围裁剪（新增，放在 IQR 后、Quantile 前）
    "enable_median_pct_clip": False,
    "median_pct_low": 0.5,   # 低于中位数 20% 之外裁掉
    "median_pct_high": 0.5,  # 高于中位数 20% 之外裁掉
    "median_pct_min_valid_count": 10,

    # 4) 全局分位数裁剪（最后兜底）
    "enable_quantile_clip": True,
    "quantile_low": 0.001,      # P0.5
    "quantile_high": 0.999,     # P99.5
    "quantile_min_valid_count": 10,
}


# =========================
# 日志打印
# =========================
def log(msg: str):
    print(msg)


# =========================
# 获取数值列
# =========================
def get_numeric_columns(df: pd.DataFrame, timestamp_col: str):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != timestamp_col]
    return cols


# =========================
# 0 值转 NaN
# =========================
def convert_zero_to_nan(df: pd.DataFrame, cols):
    if not CONFIG["treat_zero_as_nan"]:
        return df

    log("1. 执行 0 值转 NaN ...")

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        zero_mask = s.eq(0)
        if zero_mask.any():
            df.loc[zero_mask, col] = np.nan

    return df


# =========================
# 滚动均值 3-Sigma 清洗
# =========================
def rolling_sigma_clean(df: pd.DataFrame, cols):
    if not CONFIG["enable_rolling_sigma"]:
        return df

    log(
        f"2. 执行滚动均值 3-Sigma 清洗 "
        f"(window={CONFIG['sigma_window']}, sigma={CONFIG['sigma_multiplier']}) ..."
    )

    try:
        work = df[cols].apply(pd.to_numeric, errors="coerce")

        rolling_obj = work.rolling(
            window=CONFIG["sigma_window"],
            min_periods=CONFIG["sigma_min_periods"]
        )
        means = rolling_obj.mean()
        stds = rolling_obj.std()

        lower = means - CONFIG["sigma_multiplier"] * stds
        upper = means + CONFIG["sigma_multiplier"] * stds

        mask = (work < lower) | (work > upper)
        df[cols] = work.mask(mask)

    except Exception as e:
        log(f"[SigmaClean] 处理失败: {e}")

    return df


# =========================
# 全局 IQR 清洗
# =========================
def iqr_clean(df: pd.DataFrame, cols):
    if not CONFIG["enable_iqr"]:
        return df

    log(f"3. 执行全局 IQR 清洗 (multiplier={CONFIG['iqr_multiplier']}) ...")

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        if len(valid) == 0:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - CONFIG["iqr_multiplier"] * iqr
        upper = q3 + CONFIG["iqr_multiplier"] * iqr

        mask = (s < lower) | (s > upper)
        if mask.any():
            df.loc[mask, col] = np.nan

    return df


# =========================
# 全局中位数百分比范围裁剪
# 说明：
# 以全局中位数为中心，按百分比给左右边界
# lower = median * (1 - median_pct_low)
# upper = median * (1 + median_pct_high)
# =========================
def median_pct_clip_clean(df: pd.DataFrame, cols):
    if not CONFIG["enable_median_pct_clip"]:
        return df

    log(
        f"4. 执行全局中位数百分比范围裁剪 "
        f"(low={CONFIG['median_pct_low']:.2%}, high={CONFIG['median_pct_high']:.2%}) ..."
    )

    low_pct = CONFIG["median_pct_low"]
    high_pct = CONFIG["median_pct_high"]
    min_count = CONFIG["median_pct_min_valid_count"]

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        if len(valid) < min_count:
            continue

        med = valid.median()
        if pd.isna(med):
            continue

        # 中位数为 0 时，这种百分比范围无意义，直接跳过
        if med == 0:
            continue

        if med > 0:
            lower = med * (1 - low_pct)
            upper = med * (1 + high_pct)
        else:
            # 若中位数为负，保证 lower < upper
            lower = med * (1 + high_pct)
            upper = med * (1 - low_pct)
            if lower > upper:
                lower, upper = upper, lower

        mask = (s < lower) | (s > upper)
        if mask.any():
            df.loc[mask, col] = np.nan

    return df


# =========================
# 全局分位数裁剪
# =========================
def quantile_clip_clean(df: pd.DataFrame, cols):
    if not CONFIG["enable_quantile_clip"]:
        return df

    q_low = CONFIG["quantile_low"]
    q_high = CONFIG["quantile_high"]
    min_count = CONFIG["quantile_min_valid_count"]

    log(
        f"5. 执行全局分位数裁剪 "
        f"(P{q_low * 100:.3f} ~ P{q_high * 100:.3f}) ..."
    )

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        if len(valid) < min_count:
            continue

        low = valid.quantile(q_low)
        high = valid.quantile(q_high)

        if pd.isna(low) or pd.isna(high) or low > high:
            continue

        mask = (s < low) | (s > high)
        if mask.any():
            df.loc[mask, col] = np.nan

    return df


# =========================
# 删除全空列
# =========================
def drop_all_null_columns(df: pd.DataFrame):
    if not CONFIG["drop_all_null_columns"]:
        return df

    all_null_cols = [c for c in df.columns if c != TIMESTAMP_COL and df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        log(f"删除全空列数量: {len(all_null_cols)}")
    else:
        log("无全空列需要删除")

    return df


# =========================
# 主流程
# =========================
def main():
    log("=== 开始数据清洗 ===")
    log(f"输入文件: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取数据
    df = pd.read_csv(INPUT_FILE)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"未找到时间列: {TIMESTAMP_COL}")

    # 2. 处理时间列
    log("处理时间列 ...")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])

    if CONFIG["drop_duplicate_timestamp"]:
        df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="first")

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # 3. 识别数值列
    numeric_cols = get_numeric_columns(df, TIMESTAMP_COL)
    log(f"识别到数值列数量: {len(numeric_cols)}")

    if len(numeric_cols) == 0:
        raise ValueError("未识别到可清洗的数值列")

    # 4. 数值化
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # 5. 设为时间索引，便于滚动窗口
    df = df.set_index(TIMESTAMP_COL)

    # 6. 清洗流程
    df = convert_zero_to_nan(df, numeric_cols)
    df = rolling_sigma_clean(df, numeric_cols)
    df = iqr_clean(df, numeric_cols)

    existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = median_pct_clip_clean(df, existing_numeric_cols)
    df = quantile_clip_clean(df, existing_numeric_cols)

    df = drop_all_null_columns(df)

    # 7. 保存结果
    df_out = df.reset_index()
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    log("=== 数据清洗完成 ===")
    log(f"输出文件: {OUTPUT_FILE}")
    log(f"最终数据形状: {df_out.shape}")


if __name__ == "__main__":
    main()

