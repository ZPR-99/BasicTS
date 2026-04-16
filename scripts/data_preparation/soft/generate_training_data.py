import gc
import json
import os
import time

import numpy as np
import pandas as pd


CONFIG = {
    "paths": {
        "merged_input_path": r"D:\云天化\软仪表\BasicTS\datasets\raw_data\二期磷酸\merged.csv",
        "lab_input_path": r"D:\云天化\软仪表\BasicTS\datasets\raw_data\二期磷酸\Laboratory_data.csv",
        "output_root_dir": r"D:\云天化\软仪表\BasicTS\datasets\二期磷酸",
    },

    "io": {
        "timestamp_col": "Timestamp",
        "timestamp_floor_freq": "min",
        "time_frequency": "1min",
        "csv_encoding": "utf-8-sig",
    },

    "merged_clean": {
        "treat_zero_as_nan": True,
        "drop_all_null_columns": True,
        "drop_duplicate_timestamp": True,
        "enable_rolling_sigma": True,
        "sigma_window": "24h",
        "sigma_min_periods": 5,
        "sigma_multiplier": 3,
        "enable_iqr": True,
        "iqr_multiplier": 2.5,
        "enable_median_pct_clip": False,
        "median_pct_low": 0.5,
        "median_pct_high": 0.5,
        "median_pct_min_valid_count": 10,
        "enable_quantile_clip": True,
        "quantile_low": 0.001,
        "quantile_high": 0.999,
        "quantile_min_valid_count": 10,
    },

    "lab_clean": {
        "enabled": True,
        "treat_zero_as_nan": True,
        "drop_all_null_columns": True,
        # 化验表可能存在同一时刻多条记录，默认不在清洗阶段去重，留到纠偏后按(target,timestamp)聚合
        "drop_duplicate_timestamp": False,
        "enable_rolling_sigma": True,
        "sigma_window": "24h",
        "sigma_min_periods": 5,
        "sigma_multiplier": 3,
        "enable_iqr": True,
        "iqr_multiplier": 2.5,
        "enable_median_pct_clip": False,
        "median_pct_low": 0.5,
        "median_pct_high": 0.5,
        "median_pct_min_valid_count": 10,
        "enable_quantile_clip": True,
        "quantile_low": 0.001,
        "quantile_high": 0.999,
        "quantile_min_valid_count": 10,
    },

    "rt_interpolation": {
        "threshold_ratio": 0.4,
        "max_gap_threshold": 120,
        "keep_original_column_order": True,
    },

    "lab_targets": {
        "rule_hours": {
            "SO3": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            "LF302_OCMJ": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            "LF302_F4JL": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            "LF302_DA43": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            "LF302_S2YO": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        },
        "align_tol_min": 20,
    },

    "sample_targets": [
        "SO3",
        "LF302_OCMJ",
        "LF302_F4JL",
        "LF302_DA43",
        "LF302_S2YO"
    ],

    "sequence_output": {
        "predict_data_file": "predict_data.csv",
        "sequence_subdir": "深度学习时序",
        "task_type": "sequence_regression_torch",
        "consumer": "深度学习时序",
        "domain": "二期磷酸",

        # 历史窗口长度：分钟
        "input_len": 120,

        # 单目标单步输出
        "output_len": 1,

        # False 更稳妥：输入窗口截止到标签时刻前一分钟
        "include_current_minute_in_input": False,

        # 按标签时间顺序切分，避免泄露
        "train_val_test_ratio": [0.6, 0.2, 0.2],

        # 窗口中允许的最大缺失比例（超过则丢弃）
        "max_missing_ratio": 0.20,

        # 只有窗口长度达到该比例的非空值时才保留
        "min_valid_ratio": 0.80,

        # 插值后仍可能有长停机段 NaN，要求窗口时间严格连续
        "require_continuous_window": True,

        # 导出前是否使用训练集统计量填充剩余 NaN
        "nan_fill_method": "train_median_then_zero",

        # 时间特征
        "add_time_of_day": True,
        "add_day_of_week": True,
        "add_month_of_year": False,

        "metrics": ["RMSE", "MAE", "MAPE", "R2", "TDA"],
        "null_val": None,
    },
}


def log(msg):
    print(msg)


def log_step(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def force_gc(tag=""):
    gc.collect()
    if tag:
        log(f"[GC] 已释放阶段内存: {tag}")


def ensure_dir(dir_path):
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def get_timestamp_col():
    return CONFIG["io"]["timestamp_col"]


def get_timestamp_floor_freq():
    return CONFIG["io"]["timestamp_floor_freq"]


def get_time_frequency():
    return CONFIG["io"]["time_frequency"]


def get_csv_encoding():
    return CONFIG["io"]["csv_encoding"]


def get_lab_target_cols():
    return list(CONFIG["lab_targets"]["rule_hours"].keys())


def get_sample_target_cols():
    return list(CONFIG["sample_targets"])


def get_output_root_dir():
    return CONFIG["paths"]["output_root_dir"]


def get_sequence_root_dir():
    return get_output_root_dir()


def get_predict_data_output_path():
    return os.path.join(get_output_root_dir(), CONFIG["sequence_output"]["predict_data_file"])


def sanitize_columns(df, context=""):
    df = df.copy()
    raw_cols = [str(c) for c in df.columns]
    stripped_cols = [str(c).strip() for c in df.columns]
    df.columns = stripped_cols

    dup_mask = pd.Index(df.columns).duplicated()
    if dup_mask.any():
        dup_cols = pd.Index(df.columns)[dup_mask].tolist()

        collision_map = {}
        for raw, stripped in zip(raw_cols, stripped_cols):
            collision_map.setdefault(stripped, []).append(raw)

        collision_detail = {k: v for k, v in collision_map.items() if len(v) > 1}
        raise ValueError(
            f"{context}在列名清洗(strip)后出现重复列名。"
            f"\n重复列（前20项）: {dup_cols[:20]}"
            f"\n冲突来源示例: {dict(list(collision_detail.items())[:20])}"
        )

    return df


def ensure_unique_columns(df, context=""):
    dup_mask = pd.Index(df.columns).duplicated()
    if dup_mask.any():
        dup_cols = pd.Index(df.columns)[dup_mask].tolist()
        raise ValueError(f"{context}发现重复列名: {dup_cols[:20]}")
    return df


def deduplicate_preserve_order(cols, context=""):
    seen = set()
    unique_cols = []
    dup_cols = []

    for c in cols:
        if c in seen:
            dup_cols.append(c)
            continue
        seen.add(c)
        unique_cols.append(c)

    if dup_cols:
        log(f"{context}检测到重复列名并已按首次出现保留（前20项）: {dup_cols[:20]}")

    return unique_cols


def parse_and_sort_timestamp(df, timestamp_col, floor_freq=None, drop_duplicate_timestamp=True, context=""):
    if timestamp_col not in df.columns:
        raise ValueError(f"{context}缺少时间列: {timestamp_col}")

    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if floor_freq:
        ts = ts.dt.floor(floor_freq)
    df[timestamp_col] = ts

    bad_ts = int(df[timestamp_col].isna().sum())
    if bad_ts > 0:
        log(f"{context}时间列存在无法解析记录 {bad_ts} 行，这些行将被删除")
        df = df.dropna(subset=[timestamp_col])

    if drop_duplicate_timestamp:
        before = len(df)
        df = df.drop_duplicates(subset=[timestamp_col], keep="first")
        removed = before - len(df)
        if removed > 0:
            log(f"{context}重复时间戳已去除: {removed} 行")

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def get_numeric_columns(df, timestamp_col):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != timestamp_col]
    return cols


def convert_zero_to_nan(df, cols, cfg, context_name):
    if not cfg["treat_zero_as_nan"]:
        return df

    log(f"{context_name}：执行 0 值转 NaN ...")

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        zero_mask = s.eq(0)
        if zero_mask.any():
            df.loc[zero_mask, col] = np.nan

    return df


def rolling_sigma_clean(df, cols, cfg, context_name):
    if not cfg["enable_rolling_sigma"]:
        return df

    log(
        f"{context_name}：执行滚动均值 3-Sigma 清洗 "
        f"(window={cfg['sigma_window']}, sigma={cfg['sigma_multiplier']}) ..."
    )

    work = df[cols].apply(pd.to_numeric, errors="coerce")
    rolling_obj = work.rolling(
        window=cfg["sigma_window"],
        min_periods=cfg["sigma_min_periods"]
    )
    means = rolling_obj.mean()
    stds = rolling_obj.std()

    lower = means - cfg["sigma_multiplier"] * stds
    upper = means + cfg["sigma_multiplier"] * stds

    mask = (work < lower) | (work > upper)
    df[cols] = work.mask(mask)

    del work, rolling_obj, means, stds, lower, upper, mask
    force_gc(f"{context_name}::rolling_sigma_clean")
    return df


def iqr_clean(df, cols, cfg, context_name):
    if not cfg["enable_iqr"]:
        return df

    log(f"{context_name}：执行全局 IQR 清洗 (multiplier={cfg['iqr_multiplier']}) ...")

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

        lower = q1 - cfg["iqr_multiplier"] * iqr
        upper = q3 + cfg["iqr_multiplier"] * iqr

        mask = (s < lower) | (s > upper)
        if mask.any():
            df.loc[mask, col] = np.nan

    force_gc(f"{context_name}::iqr_clean")
    return df


def median_pct_clip_clean(df, cols, cfg, context_name):
    if not cfg["enable_median_pct_clip"]:
        return df

    log(
        f"{context_name}：执行全局中位数百分比范围裁剪 "
        f"(low={cfg['median_pct_low']:.2%}, high={cfg['median_pct_high']:.2%}) ..."
    )

    low_pct = cfg["median_pct_low"]
    high_pct = cfg["median_pct_high"]
    min_count = cfg["median_pct_min_valid_count"]

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        if len(valid) < min_count:
            continue

        med = valid.median()
        if pd.isna(med) or med == 0:
            continue

        if med > 0:
            lower = med * (1 - low_pct)
            upper = med * (1 + high_pct)
        else:
            lower = med * (1 + high_pct)
            upper = med * (1 - low_pct)
            if lower > upper:
                lower, upper = upper, lower

        mask = (s < lower) | (s > upper)
        if mask.any():
            df.loc[mask, col] = np.nan

    force_gc(f"{context_name}::median_pct_clip_clean")
    return df


def quantile_clip_clean(df, cols, cfg, context_name):
    if not cfg["enable_quantile_clip"]:
        return df

    q_low = cfg["quantile_low"]
    q_high = cfg["quantile_high"]
    min_count = cfg["quantile_min_valid_count"]

    log(
        f"{context_name}：执行全局分位数裁剪 "
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

    force_gc(f"{context_name}::quantile_clip_clean")
    return df


def drop_all_null_columns(df, timestamp_col, cfg, context_name):
    if not cfg["drop_all_null_columns"]:
        return df

    all_null_cols = [c for c in df.columns if c != timestamp_col and df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        log(f"{context_name}：删除全空列数量: {len(all_null_cols)}")
    else:
        log(f"{context_name}：无全空列需要删除")

    return df


def clean_dataframe_by_config(df, clean_cfg, context_name):
    timestamp_col = get_timestamp_col()

    df = sanitize_columns(df, context=context_name)
    df = ensure_unique_columns(df, context=context_name)

    df = parse_and_sort_timestamp(
        df=df,
        timestamp_col=timestamp_col,
        floor_freq=get_timestamp_floor_freq(),
        drop_duplicate_timestamp=clean_cfg["drop_duplicate_timestamp"],
        context=context_name,
    )

    numeric_cols = get_numeric_columns(df, timestamp_col)
    log(f"{context_name}：识别到可清洗数值列数量: {len(numeric_cols)}")

    if len(numeric_cols) == 0:
        raise ValueError(f"{context_name}未识别到可清洗的数值列")

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.set_index(timestamp_col)

    df = convert_zero_to_nan(df, numeric_cols, clean_cfg, context_name)
    df = rolling_sigma_clean(df, numeric_cols, clean_cfg, context_name)
    df = iqr_clean(df, numeric_cols, clean_cfg, context_name)

    existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = median_pct_clip_clean(df, existing_numeric_cols, clean_cfg, context_name)
    df = quantile_clip_clean(df, existing_numeric_cols, clean_cfg, context_name)
    df = drop_all_null_columns(df, timestamp_col, clean_cfg, context_name)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    df_out = df.reset_index()
    log(f"{context_name}：清洗后形状: {df_out.shape[0]} 行 x {df_out.shape[1]} 列")

    del df
    force_gc(f"{context_name}::clean_dataframe_by_config")
    return df_out


def clean_merged_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"merged.csv 输入文件不存在: {input_path}")

    log_step("步骤1：读取并清洗 merged.csv")
    df = pd.read_csv(input_path, encoding=get_csv_encoding())
    return clean_dataframe_by_config(
        df=df,
        clean_cfg=CONFIG["merged_clean"],
        context_name="merged.csv",
    )


def clean_lab_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Laboratory_data.csv 输入文件不存在: {input_path}")

    log_step("步骤2：读取并清洗 Laboratory_data.csv")
    df = pd.read_csv(input_path, encoding=get_csv_encoding())

    if not CONFIG["lab_clean"]["enabled"]:
        df = sanitize_columns(df, context="Laboratory_data.csv")
        df = ensure_unique_columns(df, context="Laboratory_data.csv")
        df = parse_and_sort_timestamp(
            df=df,
            timestamp_col=get_timestamp_col(),
            floor_freq=get_timestamp_floor_freq(),
            drop_duplicate_timestamp=False,
            context="Laboratory_data.csv",
        )
        for col in get_lab_target_cols():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
        return df

    return clean_dataframe_by_config(
        df=df,
        clean_cfg=CONFIG["lab_clean"],
        context_name="Laboratory_data.csv",
    )


def get_na_runs_mask(series, gap_threshold):
    is_na = series.isna()

    if not is_na.any():
        return pd.Series(False, index=series.index)

    groups = is_na.ne(is_na.shift(fill_value=False)).cumsum()
    run_lengths = is_na.groupby(groups).transform("sum")
    long_gap_mask = is_na & (run_lengths > gap_threshold)
    return long_gap_mask


def smart_interpolate_column(series, gap_threshold):
    if series.isna().sum() == 0:
        return series

    long_gap_mask = get_na_runs_mask(series, gap_threshold)
    interpolated = series.interpolate(
        method="time",
        limit_direction="both",
        limit_area="inside"
    )
    final_series = interpolated.mask(long_gap_mask, np.nan)
    return final_series


def interpolate_merged_data(cleaned_df):
    timestamp_col = get_timestamp_col()
    threshold_ratio = CONFIG["rt_interpolation"]["threshold_ratio"]
    gap_threshold = CONFIG["rt_interpolation"]["max_gap_threshold"]

    log_step("步骤3：对 cleaned merged.csv 做插值与补齐分钟时间轴")

    df = cleaned_df.copy()
    df = sanitize_columns(df, context="cleaned merged.csv")
    df = ensure_unique_columns(df, context="cleaned merged.csv")
    original_columns = df.columns.tolist()

    df = parse_and_sort_timestamp(
        df=df,
        timestamp_col=timestamp_col,
        floor_freq=None,
        drop_duplicate_timestamp=True,
        context="cleaned merged.csv"
    )

    df = df.set_index(timestamp_col).sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=get_time_frequency())
    missing_ts_count = int(len(full_index.difference(df.index)))
    if missing_ts_count > 0:
        log(f"检测到缺失时间戳 {missing_ts_count} 个，按频率 {get_time_frequency()} 补齐时间轴")
    df = df.reindex(full_index)
    df.index.name = timestamp_col

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_ratios = df[numeric_cols].notna().mean()
    cols_to_process = valid_ratios[valid_ratios >= threshold_ratio].index.tolist()
    cols_skipped = [c for c in numeric_cols if c not in cols_to_process]

    log(f"有效率 >= {threshold_ratio:.0%} 的待处理列数: {len(cols_to_process)}")
    log(f"跳过列数: {len(cols_skipped)}")

    total_start_nulls = int(df[cols_to_process].isna().sum().sum()) if cols_to_process else 0

    for i, col in enumerate(cols_to_process, start=1):
        df[col] = smart_interpolate_column(df[col], gap_threshold).astype(np.float64)

        if i % 10 == 0 or i == len(cols_to_process):
            log(f"插值进度: {i}/{len(cols_to_process)}")

    total_end_nulls = int(df[cols_to_process].isna().sum().sum()) if cols_to_process else 0
    total_filled = total_start_nulls - total_end_nulls

    all_nan_cols = [col for col in df.columns if col != timestamp_col and df[col].isna().all()]
    if all_nan_cols:
        log(f"删除全空列数: {len(all_nan_cols)}")
        df = df.drop(columns=all_nan_cols)

    if CONFIG["rt_interpolation"]["keep_original_column_order"]:
        final_columns = [c for c in original_columns if c in df.columns]
        df = df[final_columns]

    df = df.reset_index()

    log(f"merged.csv 插值后形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    log(f"插值前空值总数(待处理列): {total_start_nulls}")
    log(f"插值后空值总数(待处理列): {total_end_nulls}")
    log(f"实际填充数量: {total_filled}")

    force_gc("interpolate_merged_data")
    return df


def prepare_rt_table_df(df):
    timestamp_col = get_timestamp_col()

    df = sanitize_columns(df, context="插值后的 merged.csv")
    df = ensure_unique_columns(df, context="插值后的 merged.csv")
    df = parse_and_sort_timestamp(
        df=df,
        timestamp_col=timestamp_col,
        floor_freq=get_timestamp_floor_freq(),
        drop_duplicate_timestamp=True,
        context="插值后的 merged.csv"
    )

    for col in df.columns:
        if col != timestamp_col:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    return df


def prepare_lab_table_df(df):
    timestamp_col = get_timestamp_col()

    df = sanitize_columns(df, context="清洗后的 Laboratory_data.csv")
    df = ensure_unique_columns(df, context="清洗后的 Laboratory_data.csv")
    df = parse_and_sort_timestamp(
        df=df,
        timestamp_col=timestamp_col,
        floor_freq=get_timestamp_floor_freq(),
        drop_duplicate_timestamp=False,
        context="清洗后的 Laboratory_data.csv"
    )

    for col in get_lab_target_cols():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    return df


def build_candidate_times(ts, candidate_hours):
    day0 = ts.normalize()
    days = [day0 - pd.Timedelta(days=1), day0, day0 + pd.Timedelta(days=1)]
    candidates = []
    for d in days:
        for h in candidate_hours:
            candidates.append(d + pd.Timedelta(hours=h))
    return candidates


def align_lab_time(ts, target_name):
    tol = pd.Timedelta(minutes=CONFIG["lab_targets"]["align_tol_min"])
    candidate_hours = CONFIG["lab_targets"]["rule_hours"][target_name]
    candidates = build_candidate_times(ts, candidate_hours)

    backward_candidates = [c for c in candidates if c <= ts]
    if not backward_candidates:
        return pd.NaT, False

    aligned_time = max(backward_candidates)
    diff = ts - aligned_time

    if diff <= tol:
        return aligned_time, True
    return pd.NaT, False


def build_aligned_lab_long_table(lab_df):
    target_cols = [c for c in get_lab_target_cols() if c in lab_df.columns]
    timestamp_col = get_timestamp_col()

    lab_long = lab_df.melt(
        id_vars=[timestamp_col],
        value_vars=target_cols,
        var_name="target_name",
        value_name="y",
    ).dropna(subset=["y"]).reset_index(drop=True)

    if lab_long.empty:
        raise ValueError("Laboratory_data.csv 在去除空标签后为空，无法继续。")

    aligned = lab_long.apply(
        lambda r: align_lab_time(r[timestamp_col], r["target_name"]),
        axis=1,
        result_type="expand",
    )
    aligned.columns = ["Timestamp_aligned", "align_ok"]
    lab_long = pd.concat([lab_long, aligned], axis=1)

    log(f"化验长表原始样本数: {len(lab_long)}")
    log(f"纠偏成功样本数: {int(lab_long['align_ok'].sum())}")
    log(f"纠偏失败样本数: {int((~lab_long['align_ok']).sum())}")

    lab_long = lab_long[lab_long["align_ok"]].copy().reset_index(drop=True)
    if lab_long.empty:
        raise ValueError("化验值纠偏后无有效样本，请检查规则时刻或容忍窗口配置。")

    lab_long[timestamp_col] = lab_long["Timestamp_aligned"]
    lab_long = lab_long.drop(columns=["Timestamp_aligned", "align_ok"])

    dup_cnt = int(lab_long.duplicated(subset=["target_name", timestamp_col]).sum())
    if dup_cnt > 0:
        log(f"纠偏后重复(target_name, {timestamp_col})样本数: {dup_cnt}，将按均值聚合")
        lab_long = lab_long.groupby(["target_name", timestamp_col], as_index=False)["y"].mean()

    lab_long = lab_long.sort_values(["target_name", timestamp_col]).reset_index(drop=True)
    return lab_long


def attach_lab_by_target_asof(base_df, lab_long_df):
    timestamp_col = get_timestamp_col()

    base_df = base_df.sort_values(timestamp_col).reset_index(drop=True)
    final_df = base_df.copy()

    feature_lookup = base_df[[timestamp_col]].copy()
    feature_lookup = feature_lookup.rename(columns={timestamp_col: "feature_timestamp"})
    feature_lookup["lookup_ts"] = feature_lookup["feature_timestamp"]
    feature_lookup = feature_lookup.sort_values("lookup_ts").reset_index(drop=True)

    target_names = sorted(lab_long_df["target_name"].unique().tolist())

    for target_name in target_names:
        sub = lab_long_df[lab_long_df["target_name"] == target_name].copy()
        if sub.empty:
            continue

        sub = sub[[timestamp_col, "y"]].rename(columns={"y": target_name})
        sub = sub.sort_values(timestamp_col).reset_index(drop=True)

        left_df = sub.rename(columns={timestamp_col: "lookup_ts"})

        mapped = pd.merge_asof(
            left_df.sort_values("lookup_ts"),
            feature_lookup,
            on="lookup_ts",
            direction="backward",
            allow_exact_matches=True,
        )

        before_cnt = int(sub[target_name].notna().sum())
        unmatched_cnt = int(mapped["feature_timestamp"].isna().sum())

        if unmatched_cnt > 0:
            log(f"{target_name}: 有 {unmatched_cnt} 条纠偏后样本未找到可落点的分钟表时间，将被丢弃")

        mapped = mapped.dropna(subset=["feature_timestamp"]).copy()
        if mapped.empty:
            log(f"{target_name}: 未成功插入任何样本")
            continue

        mapped = mapped.rename(columns={"feature_timestamp": timestamp_col})
        mapped = mapped[[timestamp_col, target_name]]

        dup_cnt = int(mapped.duplicated(subset=[timestamp_col]).sum())
        if dup_cnt > 0:
            log(f"{target_name}: 有 {dup_cnt} 条样本落到相同分钟行，将按均值聚合")
            mapped = mapped.groupby(timestamp_col, as_index=False)[target_name].mean()
        else:
            mapped = mapped.sort_values(timestamp_col).reset_index(drop=True)

        after_cnt = int(mapped[target_name].notna().sum())
        log(f"{target_name}: 纠偏后样本数={before_cnt}, 成功插入样本数={after_cnt}")

        final_df = final_df.merge(mapped, on=timestamp_col, how="left")
        final_df = ensure_unique_columns(final_df, context=f"{target_name} 合并后 predict_data")

        del sub, left_df, mapped
        force_gc(f"attach_lab_by_target_asof::{target_name}")

    del feature_lookup
    force_gc("attach_lab_by_target_asof")
    return final_df


def select_input_columns_from_predict_data(predict_df):
    timestamp_col = get_timestamp_col()
    target_cols = [c for c in get_lab_target_cols() if c in predict_df.columns]

    cols = []
    all_null_cols = []

    for c in predict_df.columns:
        if c == timestamp_col:
            continue
        if c in target_cols:
            continue

        if predict_df[c].notna().sum() == 0:
            all_null_cols.append(c)
            continue

        cols.append(c)

    cols = deduplicate_preserve_order(cols, context="predict_input_cols")

    log(f"原始点位总数（不含时间和目标）: {len(cols) + len(all_null_cols)}")
    log(f"最终保留序列输入点位数: {len(cols)}")

    if all_null_cols:
        log(f"以下原始点位全为空，已跳过（前50项）: {all_null_cols[:50]}")

    return cols


def build_temporal_feature_matrix(timestamp_series):
    cfg = CONFIG["sequence_output"]

    ts = pd.to_datetime(timestamp_series, errors="coerce")
    if ts.isna().any():
        raise ValueError("时间序列存在无法解析的时间戳，无法生成标准化时间特征。")

    feature_list = []
    desc = []

    unix_ts = (ts.astype("int64") // 10 ** 9).astype(np.int64)
    feature_list.append(unix_ts.to_numpy().reshape(-1, 1).astype(np.float32))
    desc.append("unix_timestamp_seconds")

    if cfg["add_time_of_day"]:
        time_of_day = (ts.dt.hour * 60 + ts.dt.minute) / 1440.0
        feature_list.append(time_of_day.to_numpy().reshape(-1, 1).astype(np.float32))
        desc.append("time_of_day")

    if cfg["add_day_of_week"]:
        day_of_week = ts.dt.dayofweek / 7.0
        feature_list.append(day_of_week.to_numpy().reshape(-1, 1).astype(np.float32))
        desc.append("day_of_week")

    if cfg["add_month_of_year"]:
        month_of_year = (ts.dt.month - 1) / 12.0
        feature_list.append(month_of_year.to_numpy().reshape(-1, 1).astype(np.float32))
        desc.append("month_of_year")

    timestamps = np.concatenate(feature_list, axis=1)
    return timestamps, desc


def split_dataframe_by_ratio(df, ratio_list):
    total = float(sum(ratio_list))
    norm_ratios = [x / total for x in ratio_list]

    n = len(df)
    train_len = int(n * norm_ratios[0])
    val_len = int(n * norm_ratios[1])
    test_len = n - train_len - val_len

    train_df = df.iloc[:train_len].copy().reset_index(drop=True)
    val_df = df.iloc[train_len: train_len + val_len].copy().reset_index(drop=True)
    test_df = df.iloc[train_len + val_len:].copy().reset_index(drop=True)

    return train_df, val_df, test_df, {
        "train_len": train_len,
        "val_len": val_len,
        "test_len": test_len,
        "normalized_ratio": norm_ratios,
    }


def check_window_continuity(window_ts):
    if len(window_ts) <= 1:
        return True

    expected_delta = pd.to_timedelta(get_time_frequency())
    deltas = window_ts.diff().dropna()
    return bool((deltas == expected_delta).all())


def calc_train_fill_values(train_x):
    if train_x.size == 0:
        return None

    flat_x = train_x.reshape(-1, train_x.shape[-1]).astype(np.float32)
    fill_values = np.nanmedian(flat_x, axis=0)

    if np.isscalar(fill_values):
        fill_values = np.array([fill_values], dtype=np.float32)

    fill_values = np.where(np.isnan(fill_values), 0.0, fill_values).astype(np.float32)
    return fill_values


def apply_fill_values_to_3d_array(x_arr, fill_values):
    if x_arr.size == 0:
        return x_arr

    if fill_values is None:
        return np.where(np.isnan(x_arr), 0.0, x_arr).astype(np.float32)

    out = x_arr.copy()
    nan_mask = np.isnan(out)
    if nan_mask.any():
        for f_idx in range(out.shape[-1]):
            feature_nan_mask = nan_mask[:, :, f_idx]
            if feature_nan_mask.any():
                out[:, :, f_idx][feature_nan_mask] = fill_values[f_idx]
        out = np.where(np.isnan(out), 0.0, out)

    return out.astype(np.float32)


def build_single_target_sequence_samples(predict_df, target_col, input_cols):
    timestamp_col = get_timestamp_col()
    seq_cfg = CONFIG["sequence_output"]

    input_len = int(seq_cfg["input_len"])
    output_len = int(seq_cfg["output_len"])
    include_current = bool(seq_cfg["include_current_minute_in_input"])
    max_missing_ratio = float(seq_cfg["max_missing_ratio"])
    min_valid_ratio = float(seq_cfg["min_valid_ratio"])
    require_continuous = bool(seq_cfg["require_continuous_window"])

    if output_len != 1:
        raise ValueError("当前脚本仅支持单步标签输出，请将 CONFIG['sequence_output']['output_len'] 设为 1。")

    work_df = predict_df[[timestamp_col] + input_cols + [target_col]].copy()
    work_df = work_df.sort_values(timestamp_col).reset_index(drop=True)

    endpoint_df = work_df.loc[work_df[target_col].notna(), [timestamp_col, target_col]].copy()
    endpoint_df = endpoint_df.sort_values(timestamp_col).reset_index(drop=True)

    if endpoint_df.empty:
        return None

    time_to_index = pd.Series(work_df.index.to_numpy(), index=work_df[timestamp_col]).to_dict()
    train_ep, val_ep, test_ep, split_info = split_dataframe_by_ratio(
        endpoint_df,
        seq_cfg["train_val_test_ratio"]
    )

    stats = {
        "candidate_endpoints": int(len(endpoint_df)),
        "dropped_no_history": 0,
        "dropped_not_continuous": 0,
        "dropped_missing_ratio": 0,
        "dropped_low_valid_ratio": 0,
        "built_samples": 0,
    }

    def build_subset_samples(sub_endpoint_df, subset_name):
        xs = []
        ys = []
        x_ts_list = []
        y_ts_list = []
        ts_desc = []

        for _, row in sub_endpoint_df.iterrows():
            label_ts = row[timestamp_col]
            label_val = float(row[target_col])

            label_idx = time_to_index.get(label_ts, None)
            if label_idx is None:
                stats["dropped_no_history"] += 1
                continue

            if include_current:
                end_idx = label_idx
                start_idx = end_idx - input_len + 1
            else:
                end_idx = label_idx - 1
                start_idx = end_idx - input_len + 1

            if start_idx < 0 or end_idx < start_idx:
                stats["dropped_no_history"] += 1
                continue

            window_df = work_df.iloc[start_idx: end_idx + 1].copy()
            if len(window_df) != input_len:
                stats["dropped_no_history"] += 1
                continue

            if require_continuous and (not check_window_continuity(window_df[timestamp_col])):
                stats["dropped_not_continuous"] += 1
                continue

            x_df = window_df[input_cols].copy()
            total_cells = int(x_df.shape[0] * x_df.shape[1])
            valid_cells = int(x_df.notna().sum().sum())
            valid_ratio = valid_cells / total_cells if total_cells > 0 else 0.0
            missing_ratio = 1.0 - valid_ratio

            if missing_ratio > max_missing_ratio:
                stats["dropped_missing_ratio"] += 1
                continue

            if valid_ratio < min_valid_ratio:
                stats["dropped_low_valid_ratio"] += 1
                continue

            x = x_df.to_numpy(dtype=np.float32)
            y = np.array([[[label_val]]], dtype=np.float32)

            x_ts_features, ts_desc = build_temporal_feature_matrix(window_df[timestamp_col])
            y_ts_features, _ = build_temporal_feature_matrix(pd.Series([label_ts]))

            xs.append(x)
            ys.append(y[0])
            x_ts_list.append(x_ts_features.astype(np.float32))
            y_ts_list.append(y_ts_features.reshape(1, -1).astype(np.float32))

        if xs:
            x_arr = np.stack(xs, axis=0).astype(np.float32)
            y_arr = np.stack(ys, axis=0).astype(np.float32)
            x_ts_arr = np.stack(x_ts_list, axis=0).astype(np.float32)
            y_ts_arr = np.stack(y_ts_list, axis=0).astype(np.float32)
        else:
            x_arr = np.empty((0, input_len, len(input_cols)), dtype=np.float32)
            y_arr = np.empty((0, 1, 1), dtype=np.float32)
            x_ts_arr = np.empty((0, input_len, 0), dtype=np.float32)
            y_ts_arr = np.empty((0, 1, 0), dtype=np.float32)

        log(f"{target_col} | {subset_name}: endpoints={len(sub_endpoint_df)}, samples={len(x_arr)}")
        return x_arr, y_arr, x_ts_arr, y_ts_arr, ts_desc

    train_x, train_y, train_ts, train_label_ts, ts_desc = build_subset_samples(train_ep, "train")
    val_x, val_y, val_ts, val_label_ts, _ = build_subset_samples(val_ep, "val")
    test_x, test_y, test_ts, test_label_ts, _ = build_subset_samples(test_ep, "test")

    fill_method = seq_cfg["nan_fill_method"]
    if fill_method == "train_median_then_zero":
        fill_values = calc_train_fill_values(train_x)
        train_x = apply_fill_values_to_3d_array(train_x, fill_values)
        val_x = apply_fill_values_to_3d_array(val_x, fill_values)
        test_x = apply_fill_values_to_3d_array(test_x, fill_values)
    elif fill_method == "zero":
        train_x = apply_fill_values_to_3d_array(train_x, np.zeros(train_x.shape[-1], dtype=np.float32) if train_x.size > 0 else None)
        val_x = apply_fill_values_to_3d_array(val_x, np.zeros(val_x.shape[-1], dtype=np.float32) if val_x.size > 0 else None)
        test_x = apply_fill_values_to_3d_array(test_x, np.zeros(test_x.shape[-1], dtype=np.float32) if test_x.size > 0 else None)
    elif fill_method in [None, "", "keep_nan"]:
        pass
    else:
        raise ValueError(f"不支持的 nan_fill_method: {fill_method}")

    stats["built_samples"] = int(len(train_x) + len(val_x) + len(test_x))

    return {
        "target_col": target_col,
        "input_cols": input_cols,
        "split_info": split_info,
        "stats": stats,
        "ts_desc": ts_desc,
        "train": {
            "x": train_x,
            "y": train_y,
            "x_ts": train_ts,
            "y_ts": train_label_ts,
        },
        "val": {
            "x": val_x,
            "y": val_y,
            "x_ts": val_ts,
            "y_ts": val_label_ts,
        },
        "test": {
            "x": test_x,
            "y": test_y,
            "x_ts": test_ts,
            "y_ts": test_label_ts,
        },
    }


def build_dataset_meta(target_name, sample_result):
    seq_cfg = CONFIG["sequence_output"]

    num_samples = int(
        len(sample_result["train"]["x"]) +
        len(sample_result["val"]["x"]) +
        len(sample_result["test"]["x"])
    )

    return {
        "name": target_name,
        "domain": seq_cfg["domain"],
        "task_type": seq_cfg["task_type"],
        "consumer": seq_cfg["consumer"],
        "frequency": get_time_frequency(),
        "sample_mode": "endpoint_sequence_to_one",
        "target_name": target_name,
        "input_len": int(seq_cfg["input_len"]),
        "output_len": int(seq_cfg["output_len"]),
        "include_current_minute_in_input": bool(seq_cfg["include_current_minute_in_input"]),
        "num_samples": num_samples,
        "num_features": int(len(sample_result["input_cols"])),
        "num_timestamp_features": int(len(sample_result["ts_desc"])),
        "feature_columns": list(sample_result["input_cols"]),
        "timestamps_description": list(sample_result["ts_desc"]),
        "split": {
            "train_len": int(len(sample_result["train"]["x"])),
            "val_len": int(len(sample_result["val"]["x"])),
            "test_len": int(len(sample_result["test"]["x"])),
            "train_val_test_ratio": [float(x) for x in sample_result["split_info"]["normalized_ratio"]],
        },
        "regular_settings": {
            "train_val_test_ratio": [float(x) for x in sample_result["split_info"]["normalized_ratio"]],
            "norm_each_channel": False,
            "rescale": False,
            "metrics": seq_cfg["metrics"],
            "null_val": seq_cfg["null_val"],
            "max_missing_ratio": float(seq_cfg["max_missing_ratio"]),
            "min_valid_ratio": float(seq_cfg["min_valid_ratio"]),
            "require_continuous_window": bool(seq_cfg["require_continuous_window"]),
            "nan_fill_method": seq_cfg["nan_fill_method"],
        },
        "build_stats": sample_result["stats"],
        "files": {
            "train_data": "train_data.npy",
            "val_data": "val_data.npy",
            "test_data": "test_data.npy",
            "train_label": "train_label.npy",
            "val_label": "val_label.npy",
            "test_label": "test_label.npy",
            "train_timestamps": "train_timestamps.npy",
            "val_timestamps": "val_timestamps.npy",
            "test_timestamps": "test_timestamps.npy",
            "train_label_timestamps": "train_label_timestamps.npy",
            "val_label_timestamps": "val_label_timestamps.npy",
            "test_label_timestamps": "test_label_timestamps.npy",
        },
    }


def save_single_target_sequence_dataset(sample_result):
    target_name = sample_result["target_col"]
    target_dir = os.path.join(get_sequence_root_dir(), target_name)
    ensure_dir(target_dir)

    np.save(os.path.join(target_dir, "train_data.npy"), sample_result["train"]["x"])
    np.save(os.path.join(target_dir, "val_data.npy"), sample_result["val"]["x"])
    np.save(os.path.join(target_dir, "test_data.npy"), sample_result["test"]["x"])

    np.save(os.path.join(target_dir, "train_label.npy"), sample_result["train"]["y"])
    np.save(os.path.join(target_dir, "val_label.npy"), sample_result["val"]["y"])
    np.save(os.path.join(target_dir, "test_label.npy"), sample_result["test"]["y"])

    np.save(os.path.join(target_dir, "train_timestamps.npy"), sample_result["train"]["x_ts"])
    np.save(os.path.join(target_dir, "val_timestamps.npy"), sample_result["val"]["x_ts"])
    np.save(os.path.join(target_dir, "test_timestamps.npy"), sample_result["test"]["x_ts"])

    np.save(os.path.join(target_dir, "train_label_timestamps.npy"), sample_result["train"]["y_ts"])
    np.save(os.path.join(target_dir, "val_label_timestamps.npy"), sample_result["val"]["y_ts"])
    np.save(os.path.join(target_dir, "test_label_timestamps.npy"), sample_result["test"]["y_ts"])

    meta = build_dataset_meta(target_name, sample_result)
    with open(os.path.join(target_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    log(
        f"{target_name}: 深度学习时序数据集已输出到 {target_dir} | "
        f"train/val/test = "
        f"{len(sample_result['train']['x'])}/"
        f"{len(sample_result['val']['x'])}/"
        f"{len(sample_result['test']['x'])}"
    )

    return {
        "target_name": target_name,
        "target_dir": target_dir,
        "num_samples": meta["num_samples"],
        "num_features": meta["num_features"],
        "task_type": meta["task_type"],
        "consumer": meta["consumer"],
    }


def process_targets_and_export_sequence(predict_df, input_cols):
    dataset_summaries = []

    log_step("步骤6：按目标逐个构建时序样本并导出")

    for target_col in get_sample_target_cols():
        if target_col not in predict_df.columns:
            log(f"警告：predict_data 中不存在目标列 {target_col}，跳过")
            continue

        sample_count = int(predict_df[target_col].notna().sum())
        if sample_count == 0:
            log(f"{target_col}: 无目标样本，跳过")
            continue

        log_step(f"{target_col}：开始构建时序样本")
        sample_result = build_single_target_sequence_samples(
            predict_df=predict_df,
            target_col=target_col,
            input_cols=input_cols,
        )

        if sample_result is None:
            log(f"{target_col}: 样本构建失败，跳过")
            continue

        if sample_result["stats"]["built_samples"] == 0:
            log(f"{target_col}: 未构建出有效样本，跳过")
            log(f"{target_col} 样本构建统计: {sample_result['stats']}")
            continue

        summary = save_single_target_sequence_dataset(sample_result)
        dataset_summaries.append(summary)

        log(f"{target_col} 样本构建统计: {sample_result['stats']}")
        force_gc(f"{target_col}::process_targets_and_export_sequence")

    return dataset_summaries


def main():
    start_time = time.time()
    paths = CONFIG["paths"]
    timestamp_col = get_timestamp_col()

    log_step("统一数据处理脚本启动")
    log(f"merged.csv 输入路径: {paths['merged_input_path']}")
    log(f"Laboratory_data.csv 输入路径: {paths['lab_input_path']}")
    log(f"predict_data.csv 输出路径: {get_predict_data_output_path()}")
    log(f"深度学习时序输出目录: {get_sequence_root_dir()}")

    ensure_dir(get_output_root_dir())
    ensure_dir(get_sequence_root_dir())

    merged_cleaned_df = clean_merged_data(paths["merged_input_path"])
    lab_cleaned_df = clean_lab_data(paths["lab_input_path"])
    merged_interpolated_df = interpolate_merged_data(merged_cleaned_df)

    del merged_cleaned_df
    force_gc("main::after_interpolate")

    log_step("步骤4：使用插值后的 merged.csv + Laboratory_data.csv 构造 predict_data 总表")
    rt_df = prepare_rt_table_df(merged_interpolated_df)
    lab_df = prepare_lab_table_df(lab_cleaned_df)

    del merged_interpolated_df, lab_cleaned_df
    force_gc("main::after_prepare_tables")

    log(f"插值后 merged.csv 形状: {rt_df.shape[0]} 行 x {rt_df.shape[1]} 列")
    log(f"清洗后 Laboratory_data.csv 形状: {lab_df.shape[0]} 行 x {lab_df.shape[1]} 列")

    lab_long_df = build_aligned_lab_long_table(lab_df)
    del lab_df
    force_gc("main::after_build_lab_long")

    predict_df = attach_lab_by_target_asof(
        base_df=rt_df,
        lab_long_df=lab_long_df,
    )
    del lab_long_df, rt_df
    force_gc("main::after_attach_lab")

    predict_df = ensure_unique_columns(predict_df, context="predict_data")
    predict_df = predict_df.sort_values(timestamp_col).reset_index(drop=True)

    for c in [x for x in get_lab_target_cols() if x in predict_df.columns]:
        log(f"{c} 非空样本数: {int(predict_df[c].notna().sum())}")

    predict_data_output_path = get_predict_data_output_path()
    predict_df.to_csv(predict_data_output_path, index=False, encoding=get_csv_encoding())
    log(f"predict_data.csv 已保存: {predict_data_output_path}")

    log_step("步骤5：从 predict_data 中选择原始输入点位")
    input_cols = select_input_columns_from_predict_data(predict_df)
    if not input_cols:
        raise ValueError("除时间列和目标列外，其余原始点位均为空，无法构造时序样本。")

    dataset_summaries = process_targets_and_export_sequence(
        predict_df=predict_df,
        input_cols=input_cols,
    )

    del predict_df, input_cols
    force_gc("main::final_cleanup")

    log_step("完成")
    log("当前版本已按以下顺序执行：")
    log("1) merged.csv -> 清洗")
    log("2) Laboratory_data.csv -> 清洗")
    log("3) cleaned merged.csv -> 分钟级插值与补齐时间轴")
    log("4) interpolated merged.csv + cleaned Laboratory_data.csv -> 化验纠偏对齐 -> predict_data.csv")
    log("5) 基于 predict_data.csv，以目标非空时刻为终点切历史窗口")
    log("6) 导出真正的 sequence regression 数据集（不再导出表格回归数据）")
    log(f"成功输出时序数据集目标数: {len(dataset_summaries)}")
    log(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
