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

    "forecasting_output": {
        "save_source_csv": True,
        "source_csv_name": "source_table.csv",

        # 参考 ETTh1 full-series 方式：整段连续序列切 train/val/test，不预切样本
        "train_val_test_ratio": [0.6, 0.2, 0.2],

        # 供 BasicTSForecastingDataset 使用的滑窗参数，仅记录到 meta 中
        "input_len": 12,
        "output_len": 1,

        # 时间特征，风格参考 ETTh1
        "add_time_of_day": True,
        "add_day_of_week": True,
        "add_day_of_month": True,
        "add_day_of_year": True,

        # 数据集描述
        "domain": "二期磷酸",
        "task_type": "forecasting_full_series",
        "consumer": "BasicTSForecastingDataset",
        "metrics": ["MAE", "MSE"],
        "norm_each_channel": True,
        "rescale": False,
        "null_val": None,
        "graph_file_path": None,
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


def get_output_root_dir():
    return CONFIG["paths"]["output_root_dir"]


def get_dataset_root_dir():
    return get_output_root_dir()


def get_target_dir(target_name):
    return os.path.join(get_output_root_dir(), target_name)


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

    log_step("步骤2：对 cleaned merged.csv 做插值与补齐分钟时间轴")

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

    if df.empty:
        raise ValueError("cleaned merged.csv 为空，无法执行插值。")

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


def load_lab_data_without_cleaning(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Laboratory_data.csv 输入文件不存在: {input_path}")

    log_step("步骤3：读取 Laboratory_data.csv（仅做时间解析与对齐准备，不做清洗）")

    df = pd.read_csv(input_path, encoding=get_csv_encoding())
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

    if not target_cols:
        raise ValueError("Laboratory_data.csv 中未找到任何目标列，无法构造目标表。")

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


def build_single_target_table(base_df, lab_long_df, target_name):
    timestamp_col = get_timestamp_col()

    sub = lab_long_df[lab_long_df["target_name"] == target_name].copy()
    if sub.empty:
        log(f"{target_name}: 纠偏后无有效样本，跳过建表")
        return None

    sub = sub[[timestamp_col, "y"]].rename(columns={"y": target_name})
    sub = sub.sort_values(timestamp_col).reset_index(drop=True)

    feature_lookup = base_df[[timestamp_col]].copy()
    feature_lookup = feature_lookup.rename(columns={timestamp_col: "feature_timestamp"})
    feature_lookup["lookup_ts"] = feature_lookup["feature_timestamp"]
    feature_lookup = feature_lookup.sort_values("lookup_ts").reset_index(drop=True)

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
        log(f"{target_name}: 未成功插入任何样本，跳过建表")
        del sub, feature_lookup, left_df, mapped
        force_gc(f"build_single_target_table::{target_name}::empty")
        return None

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

    final_df = base_df.merge(mapped, on=timestamp_col, how="left")
    final_df = ensure_unique_columns(final_df, context=f"{target_name} 单目标表")

    before_rows = len(final_df)
    final_df = final_df.dropna(subset=[target_name], how="any").reset_index(drop=True)
    removed_rows = before_rows - len(final_df)
    log(f"{target_name}: 按‘当前目标为空就删除’规则移除行数: {removed_rows}")

    keep_cols = [timestamp_col] + [c for c in base_df.columns if c != timestamp_col] + [target_name]
    final_df = final_df[keep_cols].copy()
    final_df = final_df.sort_values(timestamp_col).reset_index(drop=True)

    del sub, feature_lookup, left_df, mapped
    force_gc(f"build_single_target_table::{target_name}")
    return final_df


def infer_frequency_minutes(ts_series):
    ts = pd.to_datetime(pd.Series(ts_series), errors="coerce").dropna().sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return None, False

    diff_minutes = ts.diff().dropna().dt.total_seconds() / 60.0
    diff_minutes = diff_minutes[diff_minutes > 0]
    if len(diff_minutes) == 0:
        return None, False

    freq_minutes = int(round(diff_minutes.mode().iloc[0]))
    is_regular = bool((diff_minutes.round().astype(int) == freq_minutes).all())
    return freq_minutes, is_regular


def add_temporal_features(df):
    cfg = CONFIG["forecasting_output"]
    ts = pd.to_datetime(df[get_timestamp_col()], errors="coerce")

    timestamps = []
    desc = []

    if cfg["add_time_of_day"]:
        tod = ((ts.dt.hour * 60 + ts.dt.minute) / 1440.0).to_numpy()
        timestamps.append(tod)
        desc.append("time_of_day")

    if cfg["add_day_of_week"]:
        dow = (ts.dt.dayofweek / 7.0).to_numpy()
        timestamps.append(dow)
        desc.append("day_of_week")

    if cfg["add_day_of_month"]:
        dom = ((ts.dt.day - 1) / 31.0).to_numpy()
        timestamps.append(dom)
        desc.append("day_of_month")

    if cfg["add_day_of_year"]:
        doy = ((ts.dt.dayofyear - 1) / 366.0).to_numpy()
        timestamps.append(doy)
        desc.append("day_of_year")

    if len(timestamps) == 0:
        raise ValueError("未启用任何时间特征，timestamps 为空。")

    timestamps = np.stack(timestamps, axis=-1).astype(np.float32)
    return timestamps, desc


def split_full_series(data, timestamps, ratio_list):
    total = float(sum(ratio_list))
    norm_ratios = [x / total for x in ratio_list]

    n = data.shape[0]
    train_len = int(n * norm_ratios[0])
    val_len = int(n * norm_ratios[1])
    test_len = n - train_len - val_len

    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len: train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len:].astype(np.float32)

    train_timestamps = timestamps[:train_len].astype(np.float32)
    val_timestamps = timestamps[train_len: train_len + val_len].astype(np.float32)
    test_timestamps = timestamps[train_len + val_len:].astype(np.float32)

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_timestamps": train_timestamps,
        "val_timestamps": val_timestamps,
        "test_timestamps": test_timestamps,
        "split": {
            "train_len": int(train_len),
            "val_len": int(val_len),
            "test_len": int(test_len),
            "train_val_test_ratio": [float(x) for x in norm_ratios],
        }
    }


def build_forecasting_meta(target_name, source_df, data, timestamps, ts_desc, split_info, inferred_freq_minutes, is_regular):
    cfg = CONFIG["forecasting_output"]
    feature_columns = [c for c in source_df.columns if c != get_timestamp_col()]
    target_channel = [len(feature_columns) - 1]

    meta = {
        "name": target_name,
        "domain": cfg["domain"],
        "task_type": cfg["task_type"],
        "consumer": cfg["consumer"],
        "timestamp_col": get_timestamp_col(),
        "target_name": target_name,
        "target_channel": target_channel,
        "input_len": int(cfg["input_len"]),
        "output_len": int(cfg["output_len"]),
        "frequency_minutes": inferred_freq_minutes,
        "is_regular_frequency": is_regular,
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "timestamps_shape": [int(timestamps.shape[0]), int(timestamps.shape[1])],
        "timestamps_description": list(ts_desc),
        "num_time_steps": int(data.shape[0]),
        "num_vars": int(data.shape[1]),
        "feature_columns": feature_columns,
        "has_graph": cfg["graph_file_path"] is not None,
        "split": split_info,
        "regular_settings": {
            "train_val_test_ratio": split_info["train_val_test_ratio"],
            "norm_each_channel": bool(cfg["norm_each_channel"]),
            "rescale": bool(cfg["rescale"]),
            "metrics": list(cfg["metrics"]),
            "null_val": cfg["null_val"],
        },
        "files": {
            "train_data": "train_data.npy",
            "val_data": "val_data.npy",
            "test_data": "test_data.npy",
            "train_timestamps": "train_timestamps.npy",
            "val_timestamps": "val_timestamps.npy",
            "test_timestamps": "test_timestamps.npy",
            "source_table_csv": CONFIG["forecasting_output"]["source_csv_name"] if CONFIG["forecasting_output"]["save_source_csv"] else None,
        },
        "notes": [
            "该目录按 ETTh1 风格导出为 full-series forecasting 数据集。",
            "train/val/test 保存的是整段连续序列，而不是预切窗样本。",
            "目标列位于 feature_columns 的最后一列，可用 target_channel 定位。",
            "如果时间间隔并非严格规则，BasicTSForecastingDataset 仍按行滑窗；是否进一步规则化可在后续版本处理。",
        ],
    }
    return meta


def save_forecasting_dataset(target_name, target_df):
    target_dir = get_target_dir(target_name)
    ensure_dir(target_dir)

    if CONFIG["forecasting_output"]["save_source_csv"]:
        target_df.to_csv(
            os.path.join(target_dir, CONFIG["forecasting_output"]["source_csv_name"]),
            index=False,
            encoding=get_csv_encoding()
        )

    data_df = target_df[[c for c in target_df.columns if c != get_timestamp_col()]].copy()
    data = data_df.to_numpy(dtype=np.float32)

    timestamps, ts_desc = add_temporal_features(target_df)
    inferred_freq_minutes, is_regular = infer_frequency_minutes(target_df[get_timestamp_col()])

    split_result = split_full_series(
        data=data,
        timestamps=timestamps,
        ratio_list=CONFIG["forecasting_output"]["train_val_test_ratio"],
    )

    np.save(os.path.join(target_dir, "train_data.npy"), split_result["train_data"])
    np.save(os.path.join(target_dir, "val_data.npy"), split_result["val_data"])
    np.save(os.path.join(target_dir, "test_data.npy"), split_result["test_data"])

    np.save(os.path.join(target_dir, "train_timestamps.npy"), split_result["train_timestamps"])
    np.save(os.path.join(target_dir, "val_timestamps.npy"), split_result["val_timestamps"])
    np.save(os.path.join(target_dir, "test_timestamps.npy"), split_result["test_timestamps"])

    meta = build_forecasting_meta(
        target_name=target_name,
        source_df=target_df,
        data=data,
        timestamps=timestamps,
        ts_desc=ts_desc,
        split_info=split_result["split"],
        inferred_freq_minutes=inferred_freq_minutes,
        is_regular=is_regular,
    )

    with open(os.path.join(target_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    log(f"{target_name}: forecasting 数据集已保存到 {target_dir}")
    log(
        f"{target_name}: train/val/test = "
        f"{split_result['split']['train_len']}/"
        f"{split_result['split']['val_len']}/"
        f"{split_result['split']['test_len']}"
    )
    log(f"{target_name}: inferred_frequency_minutes = {inferred_freq_minutes}, is_regular_frequency = {is_regular}")

    return {
        "target_name": target_name,
        "target_dir": target_dir,
        "num_time_steps": int(data.shape[0]),
        "num_vars": int(data.shape[1]),
        "inferred_frequency_minutes": inferred_freq_minutes,
        "is_regular_frequency": is_regular,
    }


def main():
    start_time = time.time()
    paths = CONFIG["paths"]
    timestamp_col = get_timestamp_col()

    log_step("统一数据处理脚本启动（ETTh1 风格 forecasting 数据集导出）")
    log(f"merged.csv 输入路径: {paths['merged_input_path']}")
    log(f"Laboratory_data.csv 输入路径: {paths['lab_input_path']}")
    log(f"forecasting 输出目录: {get_dataset_root_dir()}")

    ensure_dir(get_output_root_dir())
    ensure_dir(get_dataset_root_dir())

    merged_cleaned_df = clean_merged_data(paths["merged_input_path"])
    merged_interpolated_df = interpolate_merged_data(merged_cleaned_df)

    del merged_cleaned_df
    force_gc("main::after_interpolate")

    rt_df = prepare_rt_table_df(merged_interpolated_df)
    del merged_interpolated_df
    force_gc("main::after_prepare_rt")

    lab_df = load_lab_data_without_cleaning(paths["lab_input_path"])
    log(f"插值后 merged.csv 形状: {rt_df.shape[0]} 行 x {rt_df.shape[1]} 列")
    log(f"Laboratory_data.csv 形状: {lab_df.shape[0]} 行 x {lab_df.shape[1]} 列")

    log_step("步骤4：构造对齐后的化验长表")
    lab_long_df = build_aligned_lab_long_table(lab_df)
    del lab_df
    force_gc("main::after_build_lab_long")

    log_step("步骤5：按目标逐个生成单目标表 -> forecasting full-series 数据集")
    saved_targets = []
    feature_cols = [c for c in rt_df.columns if c != timestamp_col]
    log(f"当前分钟级特征列数: {len(feature_cols)}")

    for target_name in get_lab_target_cols():
        log_step(f"{target_name}：开始生成 forecasting 数据集")

        target_df = build_single_target_table(
            base_df=rt_df,
            lab_long_df=lab_long_df,
            target_name=target_name,
        )

        if target_df is None:
            continue

        if target_name not in target_df.columns:
            log(f"{target_name}: 单目标列不存在，跳过保存")
            del target_df
            force_gc(f"main::{target_name}::missing_target")
            continue

        non_null_cnt = int(target_df[target_name].notna().sum())
        log(f"{target_name}: 非空样本数: {non_null_cnt}")

        summary = save_forecasting_dataset(target_name, target_df)
        saved_targets.append(summary)

        del target_df
        force_gc(f"main::{target_name}::saved")

    del rt_df, lab_long_df
    force_gc("main::final_cleanup")

    log_step("完成")
    log("当前版本流程：")
    log("1) merged.csv -> 清洗")
    log("2) cleaned merged.csv -> 分钟级插值与补齐时间轴")
    log("3) Laboratory_data.csv -> 仅做时间解析与目标列数值化，不做清洗")
    log("4) 插值后的 merged.csv + Laboratory_data.csv -> 化验纠偏对齐")
    log("5) 为每个目标生成：时间戳 + 特征 + 单目标 的单目标表")
    log("6) 参考 ETTh1，将单目标表导出为 BasicTSForecastingDataset 所需的 full-series 格式")
    log("7) 每个目标目录输出 train/val/test_data.npy、*_timestamps.npy、meta.json")

    log(f"成功输出目标数: {len(saved_targets)}")
    if saved_targets:
        log(f"成功输出目标摘要: {saved_targets}")

    log(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()

