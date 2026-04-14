
import gc
import json
import os
import time

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


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

    "feature_matrix": {
        "keep_raw_inputs": True,
        "lag_minutes": [1, 30, 60, 90, 120, 180, 240, 360],
        "window_minutes": [15, 30, 60, 90, 120, 180, 360],
        "rolling_min_periods": 1,
        "progress_log_every": 20,
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

    "xgb_selection": {
        "enabled": True,
        "top_k": 300,
        "importance_type": "gain",
        "model_params": {
            "n_estimators": 1200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.2,
            "reg_lambda": 2,
            "random_state": 42,
            "n_jobs": 1,
            "tree_method": "hist",
        }
    },

    "standardized_output": {
        "train_val_test_ratio": [0.6, 0.2, 0.2],
        "add_time_of_day": True,
        "add_day_of_week": True,
        "domain": "二期磷酸",
        "metrics": ["MAE", "MSE"],
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


def get_lab_target_cols():
    return list(CONFIG["lab_targets"]["rule_hours"].keys())


def get_sample_target_cols():
    return list(CONFIG["sample_targets"])


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


def convert_zero_to_nan(df, cols):
    cfg = CONFIG["merged_clean"]
    if not cfg["treat_zero_as_nan"]:
        return df

    log("1. 对 merged.csv 执行 0 值转 NaN ...")

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        zero_mask = s.eq(0)
        if zero_mask.any():
            df.loc[zero_mask, col] = np.nan

    return df


def rolling_sigma_clean(df, cols):
    cfg = CONFIG["merged_clean"]
    if not cfg["enable_rolling_sigma"]:
        return df

    log(
        f"2. 对 merged.csv 执行滚动均值 3-Sigma 清洗 "
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
    force_gc("rolling_sigma_clean")
    return df


def iqr_clean(df, cols):
    cfg = CONFIG["merged_clean"]
    if not cfg["enable_iqr"]:
        return df

    log(f"3. 对 merged.csv 执行全局 IQR 清洗 (multiplier={cfg['iqr_multiplier']}) ...")

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

    force_gc("iqr_clean")
    return df


def median_pct_clip_clean(df, cols):
    cfg = CONFIG["merged_clean"]
    if not cfg["enable_median_pct_clip"]:
        return df

    log(
        f"4. 对 merged.csv 执行全局中位数百分比范围裁剪 "
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

    force_gc("median_pct_clip_clean")
    return df


def quantile_clip_clean(df, cols):
    cfg = CONFIG["merged_clean"]
    if not cfg["enable_quantile_clip"]:
        return df

    q_low = cfg["quantile_low"]
    q_high = cfg["quantile_high"]
    min_count = cfg["quantile_min_valid_count"]

    log(
        f"5. 对 merged.csv 执行全局分位数裁剪 "
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

    force_gc("quantile_clip_clean")
    return df


def drop_all_null_columns(df):
    cfg = CONFIG["merged_clean"]
    timestamp_col = get_timestamp_col()

    if not cfg["drop_all_null_columns"]:
        return df

    all_null_cols = [c for c in df.columns if c != timestamp_col and df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        log(f"删除 merged.csv 全空列数量: {len(all_null_cols)}")
    else:
        log("merged.csv 无全空列需要删除")

    return df


def clean_merged_data(input_path):
    timestamp_col = get_timestamp_col()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"merged.csv 输入文件不存在: {input_path}")

    log_step("步骤1：对 merged.csv 做清洗")

    df = pd.read_csv(input_path)
    df = sanitize_columns(df, context="merged.csv")
    df = ensure_unique_columns(df, context="merged.csv")

    df = parse_and_sort_timestamp(
        df=df,
        timestamp_col=timestamp_col,
        floor_freq=get_timestamp_floor_freq(),
        drop_duplicate_timestamp=CONFIG["merged_clean"]["drop_duplicate_timestamp"],
        context="merged.csv"
    )

    numeric_cols = get_numeric_columns(df, timestamp_col)
    log(f"merged.csv 识别到可清洗数值列数量: {len(numeric_cols)}")

    if len(numeric_cols) == 0:
        raise ValueError("merged.csv 未识别到可清洗的数值列")

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.set_index(timestamp_col)

    df = convert_zero_to_nan(df, numeric_cols)
    df = rolling_sigma_clean(df, numeric_cols)
    df = iqr_clean(df, numeric_cols)

    existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = median_pct_clip_clean(df, existing_numeric_cols)
    df = quantile_clip_clean(df, existing_numeric_cols)
    df = drop_all_null_columns(df)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    df_out = df.reset_index()
    log(f"merged.csv 清洗后形状: {df_out.shape[0]} 行 x {df_out.shape[1]} 列")
    del df
    force_gc("clean_merged_data")
    return df_out


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

    log_step("步骤2：对 cleaned merged.csv 做插值")

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
        df[col] = smart_interpolate_column(df[col], gap_threshold).astype(np.float32)

        if i % 10 == 0 or i == len(cols_to_process):
            log(f"插值进度: {i}/{len(cols_to_process)}")

    total_end_nulls = int(df[cols_to_process].isna().sum().sum()) if cols_to_process else 0
    total_filled = total_start_nulls - total_end_nulls

    all_nan_cols = [
        col for col in df.columns
        if col != timestamp_col and df[col].isna().all()
    ]
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


def prepare_lab_table_df(file_path):
    timestamp_col = get_timestamp_col()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Laboratory_data.csv 输入文件不存在: {file_path}")

    df = pd.read_csv(file_path)
    df = sanitize_columns(df, context="Laboratory_data.csv")
    df = ensure_unique_columns(df, context="Laboratory_data.csv")

    df = parse_and_sort_timestamp(
        df=df,
        timestamp_col=timestamp_col,
        floor_freq=get_timestamp_floor_freq(),
        drop_duplicate_timestamp=False,
        context="Laboratory_data.csv"
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
        lab_long = (
            lab_long.groupby(["target_name", timestamp_col], as_index=False)["y"]
            .mean()
        )

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
        final_df = ensure_unique_columns(final_df, context=f"{target_name} 合并后数据表")

        del sub, left_df, mapped
        force_gc(f"attach_lab_by_target_asof::{target_name}")

    del feature_lookup
    force_gc("attach_lab_by_target_asof")
    return final_df


def select_input_columns_from_merged(merged_df):
    timestamp_col = get_timestamp_col()
    target_cols = [c for c in get_lab_target_cols() if c in merged_df.columns]

    cols = []
    all_null_cols = []

    for c in merged_df.columns:
        if c == timestamp_col:
            continue
        if c in target_cols:
            continue

        if merged_df[c].notna().sum() == 0:
            all_null_cols.append(c)
            continue

        cols.append(c)

    cols = deduplicate_preserve_order(cols, context="merged_input_cols")

    log(f"原始点位总数（不含时间和目标）: {len(cols) + len(all_null_cols)}")
    log(f"最终保留用于构造特征的原始点位数: {len(cols)}")

    if all_null_cols:
        log(f"以下原始点位全为空，已跳过（前50项）: {all_null_cols[:50]}")

    return cols


def build_target_training_table(rt_indexed, merged_with_target_df, target_col, input_cols):
    timestamp_col = get_timestamp_col()
    feature_cfg = CONFIG["feature_matrix"]
    keep_raw_inputs = feature_cfg["keep_raw_inputs"]
    lag_minutes = feature_cfg["lag_minutes"]
    window_minutes = feature_cfg["window_minutes"]
    rolling_min_periods = feature_cfg["rolling_min_periods"]
    progress_log_every = feature_cfg["progress_log_every"]

    target_rows = merged_with_target_df.loc[
        merged_with_target_df[target_col].notna(),
        [timestamp_col, target_col]
    ].copy()

    if target_rows.empty:
        return pd.DataFrame(columns=[timestamp_col, target_col])

    target_rows[timestamp_col] = pd.to_datetime(target_rows[timestamp_col], errors="coerce")
    target_rows[target_col] = pd.to_numeric(target_rows[target_col], errors="coerce", downcast="float")
    target_rows = target_rows.dropna(subset=[timestamp_col, target_col])
    target_rows = target_rows.drop_duplicates(subset=[timestamp_col], keep="first")
    target_rows = target_rows.sort_values(timestamp_col).reset_index(drop=True)

    sample_ts = pd.DatetimeIndex(target_rows[timestamp_col])
    feature_data = {
        timestamp_col: target_rows[timestamp_col].tolist(),
        target_col: target_rows[target_col].astype(np.float32).to_numpy(),
    }

    total_cols = len(input_cols)
    for idx, col in enumerate(input_cols, start=1):
        if idx == 1 or idx % progress_log_every == 0 or idx == total_cols:
            log(f"{target_col} 特征构造进度: {idx}/{total_cols} -> {col}")

        signal = pd.to_numeric(rt_indexed[col], errors="coerce").astype(np.float32)

        if keep_raw_inputs:
            feature_data[col] = signal.reindex(sample_ts).astype(np.float32).to_numpy()

        for lag in lag_minutes:
            if lag <= 0:
                continue
            lagged = signal.shift(lag)
            feature_data[f"{col}_lag_{lag}m"] = lagged.reindex(sample_ts).astype(np.float32).to_numpy()
            del lagged

        for w in window_minutes:
            if w <= 0:
                continue

            rolling_obj = signal.rolling(window=w, min_periods=rolling_min_periods)
            feature_data[f"{col}_ma_{w}m"] = rolling_obj.mean().reindex(sample_ts).astype(np.float32).to_numpy()
            feature_data[f"{col}_std_{w}m"] = rolling_obj.std().reindex(sample_ts).astype(np.float32).to_numpy()
            feature_data[f"{col}_max_{w}m"] = rolling_obj.max().reindex(sample_ts).astype(np.float32).to_numpy()
            feature_data[f"{col}_min_{w}m"] = rolling_obj.min().reindex(sample_ts).astype(np.float32).to_numpy()
            del rolling_obj

        del signal
        if idx % 10 == 0 or idx == total_cols:
            force_gc(f"{target_col}::feature_build::{idx}")

    target_df = pd.DataFrame(feature_data)
    target_df = ensure_unique_columns(target_df, context=f"{target_col}_target_training_table")
    return target_df


def clean_target_training_table(target_df, target_col):
    timestamp_col = get_timestamp_col()

    if target_df.empty:
        return target_df, []

    log(f"{target_col}: 特征构造完成后开始清洗缩容")

    target_df = target_df.replace([np.inf, -np.inf], np.nan)
    target_df[target_col] = pd.to_numeric(target_df[target_col], errors="coerce", downcast="float")
    target_df = target_df.dropna(subset=[timestamp_col, target_col]).reset_index(drop=True)

    feature_cols = [
        c for c in target_df.columns
        if c not in [timestamp_col, target_col]
    ]

    valid_feature_cols = []
    dropped_all_nan_cols = []

    for col in feature_cols:
        if target_df[col].notna().sum() > 0:
            valid_feature_cols.append(col)
        else:
            dropped_all_nan_cols.append(col)

    if dropped_all_nan_cols:
        log(f"{target_col}: 清洗时删除全空特征列数={len(dropped_all_nan_cols)}")

    keep_cols = [timestamp_col, target_col] + valid_feature_cols
    keep_cols = deduplicate_preserve_order(keep_cols, context=f"{target_col}_clean_keep_cols")
    target_df = target_df.loc[:, keep_cols].copy()

    for col in valid_feature_cols:
        target_df[col] = pd.to_numeric(target_df[col], errors="coerce", downcast="float")

    log(f"{target_col}: 清洗缩容后样本表形状: {target_df.shape[0]} 行 x {target_df.shape[1]} 列")
    force_gc(f"{target_col}::clean_target_training_table")
    return target_df, valid_feature_cols


def fit_xgb_and_select_features(X, y, feature_cols):
    if XGBRegressor is None:
        raise ImportError("未安装 xgboost，无法执行 XGBoost 特征筛选。请先安装：pip install xgboost")

    cfg = CONFIG["xgb_selection"]
    top_k = int(cfg["top_k"])
    importance_type = cfg["importance_type"]

    X = X[feature_cols].copy()
    X = X.replace([float("inf"), float("-inf")], np.nan)
    X = X.astype(np.float32)
    y = pd.to_numeric(y, errors="coerce").astype(np.float32)

    model_params = dict(cfg["model_params"])
    model_params["importance_type"] = importance_type

    model = XGBRegressor(**model_params)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    selected_cols = importance_df["feature"].tolist()[:top_k]

    del X, y, model
    force_gc("fit_xgb_and_select_features")
    return selected_cols, importance_df


def build_temporal_feature_matrix(timestamp_series):
    cfg = CONFIG["standardized_output"]

    ts = pd.to_datetime(timestamp_series, errors="coerce")
    if ts.isna().any():
        raise ValueError("时间序列存在无法解析的时间戳，无法生成标准化时间特征。")

    feature_list = []
    desc = []

    unix_ts = (ts.astype("int64") // 10**9).astype(np.int64)
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

    timestamps = np.concatenate(feature_list, axis=1)
    return timestamps, desc


def split_dataframe_by_ratio(df, ratio_list):
    total = float(sum(ratio_list))
    norm_ratios = [x / total for x in ratio_list]
    train_ratio, val_ratio, test_ratio = norm_ratios

    n = len(df)
    train_len = int(n * train_ratio)
    val_len = int(n * val_ratio)
    test_len = n - train_len - val_len

    train_df = df.iloc[:train_len].copy()
    val_df = df.iloc[train_len: train_len + val_len].copy()
    test_df = df.iloc[train_len + val_len:].copy()

    return train_df, val_df, test_df, {
        "train_len": train_len,
        "val_len": val_len,
        "test_len": test_len,
        "normalized_ratio": [train_ratio, val_ratio, test_ratio],
    }


def export_single_target_dataset(target_name, sample_df, target_dir):
    cfg = CONFIG["standardized_output"]
    timestamp_col = get_timestamp_col()

    ensure_dir(target_dir)

    sample_df = sample_df.copy()
    sample_df[timestamp_col] = pd.to_datetime(sample_df[timestamp_col], errors="coerce")
    sample_df = sample_df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    if sample_df.empty:
        log(f"{target_name}: 样本为空，跳过")
        return None

    feature_cols = [
        c for c in sample_df.columns
        if c not in [timestamp_col, target_name]
    ]
    feature_cols = deduplicate_preserve_order(feature_cols, context=f"{target_name}_standardized_feature_cols")

    if not feature_cols:
        log(f"{target_name}: 无可用特征列，跳过")
        return None

    train_df, val_df, test_df, split_info = split_dataframe_by_ratio(sample_df, cfg["train_val_test_ratio"])

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        log(
            f"{target_name}: 按比例切分后存在空集合 "
            f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})，跳过"
        )
        return None

    def to_xy_timestamps(sub_df):
        x = sub_df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        y = pd.to_numeric(sub_df[target_name], errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
        ts_features, ts_desc = build_temporal_feature_matrix(sub_df[timestamp_col])
        return x, y, ts_features, ts_desc

    train_x, train_y, train_ts, ts_desc = to_xy_timestamps(train_df)
    val_x, val_y, val_ts, _ = to_xy_timestamps(val_df)
    test_x, test_y, test_ts, _ = to_xy_timestamps(test_df)

    np.save(os.path.join(target_dir, "train_data.npy"), train_x)
    np.save(os.path.join(target_dir, "val_data.npy"), val_x)
    np.save(os.path.join(target_dir, "test_data.npy"), test_x)

    np.save(os.path.join(target_dir, "train_label.npy"), train_y)
    np.save(os.path.join(target_dir, "val_label.npy"), val_y)
    np.save(os.path.join(target_dir, "test_label.npy"), test_y)

    np.save(os.path.join(target_dir, "train_timestamps.npy"), train_ts)
    np.save(os.path.join(target_dir, "val_timestamps.npy"), val_ts)
    np.save(os.path.join(target_dir, "test_timestamps.npy"), test_ts)

    meta = {
        "name": target_name,
        "domain": cfg["domain"],
        "task_type": "tabular_regression",
        "frequency": get_time_frequency(),
        "shape": [int(sample_df.shape[0]), int(len(feature_cols))],
        "timestamps_shape": [int(sample_df.shape[0]), int(len(ts_desc))],
        "timestamps_description": ts_desc,
        "num_samples": int(sample_df.shape[0]),
        "num_features": int(len(feature_cols)),
        "target_name": target_name,
        "split": {
            "train_len": int(split_info["train_len"]),
            "val_len": int(split_info["val_len"]),
            "test_len": int(split_info["test_len"]),
            "train_val_test_ratio": [float(x) for x in split_info["normalized_ratio"]],
        },
        "regular_settings": {
            "train_val_test_ratio": [float(x) for x in split_info["normalized_ratio"]],
            "norm_each_channel": False,
            "rescale": False,
            "metrics": cfg["metrics"],
            "null_val": cfg["null_val"],
        },
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
        },
    }

    meta_path = os.path.join(target_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    log(f"{target_name}: 标准化数据集已输出到 {target_dir}")
    log(f"{target_name}: train/val/test = {split_info['train_len']}/{split_info['val_len']}/{split_info['test_len']}")

    del train_df, val_df, test_df, train_x, val_x, test_x, train_y, val_y, test_y, train_ts, val_ts, test_ts
    force_gc(f"export_single_target_dataset::{target_name}")
    return {
        "target_name": target_name,
        "target_dir": target_dir,
        "num_samples": int(sample_df.shape[0]),
        "num_features": int(len(feature_cols)),
    }


def process_targets_and_export(rt_indexed, merged_with_target_df, input_cols, output_root_dir):
    sample_targets = get_sample_target_cols()
    dataset_summaries = []
    timestamp_col = get_timestamp_col()

    log_step("步骤4：按目标逐个构造特征 -> 清洗缩容 -> 筛选 -> 导出")

    for target_col in sample_targets:
        if target_col not in merged_with_target_df.columns:
            log(f"警告：当前表中不存在目标列 {target_col}，跳过")
            continue

        sample_count = int(merged_with_target_df[target_col].notna().sum())
        if sample_count == 0:
            log(f"{target_col}: 无目标样本，跳过")
            continue

        log_step(f"{target_col}：构造目标样本特征表")
        target_full_df = build_target_training_table(
            rt_indexed=rt_indexed,
            merged_with_target_df=merged_with_target_df,
            target_col=target_col,
            input_cols=input_cols,
        )
        log(f"{target_col}: 原始目标样本特征表形状 = {target_full_df.shape[0]} 行 x {target_full_df.shape[1]} 列")

        target_full_df, valid_feature_cols = clean_target_training_table(target_full_df, target_col)

        if target_full_df.empty:
            log(f"{target_col}: 清洗缩容后无有效样本，跳过")
            del target_full_df
            force_gc(f"{target_col}::after_clean_empty")
            continue

        if not valid_feature_cols:
            log(f"{target_col}: 清洗缩容后无有效特征，跳过")
            del target_full_df
            force_gc(f"{target_col}::after_clean_no_feature")
            continue

        log_step(f"{target_col}：开始 XGBoost 筛选")
        X = target_full_df[valid_feature_cols].copy()
        y = pd.to_numeric(target_full_df[target_col], errors="coerce").reset_index(drop=True)

        selected_cols, _ = fit_xgb_and_select_features(
            X=X,
            y=y,
            feature_cols=valid_feature_cols,
        )

        del X, y
        force_gc(f"{target_col}::after_xgb_fit")

        if not selected_cols:
            log(f"{target_col}: 未选出特征，跳过")
            del target_full_df
            force_gc(f"{target_col}::no_selected_cols")
            continue

        selected_keep_cols = [timestamp_col, target_col] + selected_cols
        selected_keep_cols = deduplicate_preserve_order(selected_keep_cols, context=f"{target_col}_selected_keep_cols")
        selected_sample_df = target_full_df.loc[:, selected_keep_cols].copy()

        del target_full_df
        force_gc(f"{target_col}::after_select_sample_df")

        target_dir = os.path.join(output_root_dir, target_col)
        summary = export_single_target_dataset(
            target_name=target_col,
            sample_df=selected_sample_df,
            target_dir=target_dir,
        )

        del selected_sample_df, valid_feature_cols, selected_cols
        force_gc(f"{target_col}::loop_end")

        if summary is not None:
            dataset_summaries.append(summary)

    return dataset_summaries


def main():
    start_time = time.time()
    paths = CONFIG["paths"]
    timestamp_col = get_timestamp_col()

    log_step("统一数据处理脚本启动")
    log(f"merged.csv 输入路径: {paths['merged_input_path']}")
    log(f"Laboratory_data.csv 输入路径: {paths['lab_input_path']}")
    log(f"最终输出目录: {paths['output_root_dir']}")

    ensure_dir(paths["output_root_dir"])

    merged_cleaned_df = clean_merged_data(paths["merged_input_path"])
    merged_interpolated_df = interpolate_merged_data(merged_cleaned_df)

    del merged_cleaned_df
    force_gc("main::after_interpolate")

    log_step("步骤3：使用插值后的 merged.csv + Laboratory_data.csv 构造训练样本")
    rt_df = prepare_rt_table_df(merged_interpolated_df)
    del merged_interpolated_df
    force_gc("main::after_prepare_rt")

    lab_df = prepare_lab_table_df(paths["lab_input_path"])

    log(f"插值后 merged.csv 形状: {rt_df.shape[0]} 行 x {rt_df.shape[1]} 列")
    log(f"Laboratory_data.csv 形状: {lab_df.shape[0]} 行 x {lab_df.shape[1]} 列")

    lab_long_df = build_aligned_lab_long_table(lab_df)
    del lab_df
    force_gc("main::after_build_lab_long")

    merged_with_target_df = attach_lab_by_target_asof(
        base_df=rt_df,
        lab_long_df=lab_long_df,
    )
    del lab_long_df, rt_df
    force_gc("main::after_attach_lab")

    merged_with_target_df = ensure_unique_columns(merged_with_target_df, context="目标对齐后的数据表")
    merged_with_target_df = merged_with_target_df.sort_values(timestamp_col).reset_index(drop=True)

    for c in [x for x in get_lab_target_cols() if x in merged_with_target_df.columns]:
        log(f"{c} 非空样本数: {int(merged_with_target_df[c].notna().sum())}")

    input_cols = select_input_columns_from_merged(merged_with_target_df)
    if not input_cols:
        raise ValueError("除时间列和目标列外，其余原始点位均为空，无法构造特征。")

    rt_indexed = merged_with_target_df[[timestamp_col] + input_cols].copy()
    rt_indexed = rt_indexed.set_index(timestamp_col).sort_index()

    dataset_summaries = process_targets_and_export(
        rt_indexed=rt_indexed,
        merged_with_target_df=merged_with_target_df,
        input_cols=input_cols,
        output_root_dir=paths["output_root_dir"],
    )

    del rt_indexed, merged_with_target_df, input_cols
    force_gc("main::final_cleanup")

    log_step("完成")
    log("当前版本已按以下顺序执行：")
    log("1) merged.csv -> 清洗")
    log("2) cleaned merged.csv -> 插值")
    log("3) interpolated merged.csv + Laboratory_data.csv -> 目标对齐")
    log("4) 每个目标单独：构造特征 -> 清洗缩容(去目标空行/全空特征列) -> XGBoost筛选 -> 导出")
    log("5) 每个阶段结束立即释放内存")
    log(f"成功输出目标数: {len(dataset_summaries)}")
    log(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
