"""
脚本功能说明：
1. 读取分钟级输入表和化验表，完成时间对齐与合并，生成原始总表 predict_data.csv
2. predict_data.csv 仅保留时间列、原始点位列和目标列，不在该阶段生成衍生特征
3. 基于 predict_data.csv 的原始点位构造统一时序特征，并进行 XGBoost 特征筛选
4. 最终输出：
   - 各目标 XXX_importance.csv（特征重要性表）
   - 各目标 XXX_samples.csv（训练样本表）
   - 所有目标筛后保留特征去重汇总表 all_targets_selected_features.csv
   - 所有目标对应原始点位去重清单表 all_targets_selected_raw_points.csv
"""
import os
import re
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# =========================================================
# 配置参数
# =========================================================
CONFIG = {
    "paths": {
        "rt_path": r"D:\云天化\软仪表\天安\二期磷酸\model\date\input_date\merged_interpolated.csv",
        "lab_path": r"D:\云天化\软仪表\天安\二期磷酸\model\date\raw_date\Laboratory_data1.csv",
        "output_dir": r"D:\云天化\软仪表\天安\二期磷酸\model\date\feature\2",
        "final_output_file": "predict_data.csv",
        "xgb_selected_subdir": "xgb_selected",
    },

    "io": {
        "timestamp_col": "Timestamp",
        "timestamp_floor_freq": "min",
        "csv_encoding": "utf-8-sig",
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

    "sample_output": {
        "xgb_training_suffix": "samples",
        "xgb_feature_importance_suffix": "importance",
        "all_targets_selected_features_file": "all_targets_selected_features.csv",
        "all_targets_raw_points_table_file": "all_targets_selected_raw_points.csv",
    },

    "xgb_selection": {
        "enabled": True,
        "top_k": 300,
        "importance_type": "gain",
        "model_params": {
            "n_estimators": 2000,
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
}


# =========================================================
# 日志输出函数
# =========================================================
def log_step(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def log_info(msg):
    print(msg)


# =========================================================
# 配置读取辅助函数
# =========================================================
def get_timestamp_col():
    return CONFIG["io"]["timestamp_col"]


def get_timestamp_floor_freq():
    return CONFIG["io"]["timestamp_floor_freq"]


def get_csv_encoding():
    return CONFIG["io"]["csv_encoding"]


def get_lab_target_cols():
    return list(CONFIG["lab_targets"]["rule_hours"].keys())


def get_sample_target_cols():
    return list(CONFIG["sample_targets"])


# =========================================================
# 数据处理工具函数
# =========================================================
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
        log_info(f"{context}检测到重复列名并已按首次出现保留（前20项）: {dup_cols[:20]}")

    return unique_cols


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


def read_and_prepare_rt_table(file_path):
    """
    读取并预处理分钟级输入表
    """
    timestamp_col = get_timestamp_col()
    timestamp_floor_freq = get_timestamp_floor_freq()

    df = pd.read_csv(file_path)
    df = sanitize_columns(df, context="原始分钟级输入表")
    df = ensure_unique_columns(df, context="原始分钟级输入表")

    if timestamp_col not in df.columns:
        raise ValueError(f"分钟级输入表缺少时间列: {timestamp_col}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce").dt.floor(timestamp_floor_freq)
    bad_ts = int(df[timestamp_col].isna().sum())
    if bad_ts > 0:
        log_info(f"警告：分钟级输入表时间列存在无法解析记录 {bad_ts} 行，这些行将被删除")
        df = df.dropna(subset=[timestamp_col])

    df = (
        df.sort_values(timestamp_col)
        .drop_duplicates(subset=[timestamp_col], keep="first")
        .reset_index(drop=True)
    )

    for col in df.columns:
        if col != timestamp_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def read_and_prepare_lab_table(file_path):
    """
    读取并预处理化验表
    """
    timestamp_col = get_timestamp_col()
    timestamp_floor_freq = get_timestamp_floor_freq()

    df = pd.read_csv(file_path)
    df = sanitize_columns(df, context="化验表")
    df = ensure_unique_columns(df, context="化验表")

    if timestamp_col not in df.columns:
        raise ValueError(f"化验表缺少时间列: {timestamp_col}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce").dt.floor(timestamp_floor_freq)
    bad_ts = int(df[timestamp_col].isna().sum())
    if bad_ts > 0:
        log_info(f"警告：化验表时间列存在无法解析记录 {bad_ts} 行，这些行将被删除")
        df = df.dropna(subset=[timestamp_col])

    for col in get_lab_target_cols():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# 化验值纠偏相关
# =========================================================
def build_candidate_times(ts, candidate_hours):
    """
    根据给定时间戳和候选小时列表,生成候选时间点
    """
    day0 = ts.normalize()
    days = [day0 - pd.Timedelta(days=1), day0, day0 + pd.Timedelta(days=1)]
    candidates = []
    for d in days:
        for h in candidate_hours:
            candidates.append(d + pd.Timedelta(hours=h))
    return candidates


def align_lab_time(ts, target_name):
    """
    仅允许将化验时间对齐到不晚于当前时间戳的最近规则时刻，避免未来信息泄露。
    """
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
    """
    返回纠偏后的化验长表:Timestamp, target_name, y
    """
    target_cols = [c for c in get_lab_target_cols() if c in lab_df.columns]
    timestamp_col = get_timestamp_col()

    if timestamp_col not in lab_df.columns:
        raise ValueError(f"化验表缺少 {timestamp_col} 列")

    lab_long = lab_df.melt(
        id_vars=[timestamp_col],
        value_vars=target_cols,
        var_name="target_name",
        value_name="y",
    ).dropna(subset=["y"]).reset_index(drop=True)

    if lab_long.empty:
        raise ValueError("化验表在去除空标签后为空，无法继续。")

    aligned = lab_long.apply(
        lambda r: align_lab_time(r[timestamp_col], r["target_name"]),
        axis=1,
        result_type="expand",
    )
    aligned.columns = ["Timestamp_aligned", "align_ok"]
    lab_long = pd.concat([lab_long, aligned], axis=1)

    log_info(f"化验长表原始样本数: {len(lab_long)}")
    log_info(f"纠偏成功样本数: {int(lab_long['align_ok'].sum())}")
    log_info(f"纠偏失败样本数: {int((~lab_long['align_ok']).sum())}")

    lab_long = lab_long[lab_long["align_ok"]].copy().reset_index(drop=True)
    if lab_long.empty:
        raise ValueError("化验值纠偏后无有效样本，请检查规则时刻或容忍窗口配置。")

    lab_long[timestamp_col] = lab_long["Timestamp_aligned"]
    lab_long = lab_long.drop(columns=["Timestamp_aligned", "align_ok"])

    dup_cnt = int(lab_long.duplicated(subset=["target_name", timestamp_col]).sum())
    if dup_cnt > 0:
        log_info(f"纠偏后重复(target_name, {timestamp_col})样本数: {dup_cnt}，将按均值聚合")
        lab_long = (
            lab_long.groupby(["target_name", timestamp_col], as_index=False)["y"]
            .mean()
        )

    lab_long = lab_long.sort_values(["target_name", timestamp_col]).reset_index(drop=True)
    return lab_long


# =========================================================
# 生成 predict_data.csv
# =========================================================
def attach_lab_by_target_asof(base_df, lab_long_df):
    """
    将纠偏后的化验值逐目标落到原始分钟级表上，生成 predict_data.csv。
    此阶段只做原始表合并，不做衍生特征构造。
    """
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
            log_info(f"{target_name}: 有 {unmatched_cnt} 条纠偏后样本未找到可落点的分钟表时间，将被丢弃")

        mapped = mapped.dropna(subset=["feature_timestamp"]).copy()
        if mapped.empty:
            log_info(f"{target_name}: 未成功插入任何样本")
            continue

        mapped = mapped.rename(columns={"feature_timestamp": timestamp_col})
        mapped = mapped[[timestamp_col, target_name]]

        dup_cnt = int(mapped.duplicated(subset=[timestamp_col]).sum())
        if dup_cnt > 0:
            log_info(f"{target_name}: 有 {dup_cnt} 条样本落到相同分钟行，将按均值聚合")
            mapped = mapped.groupby(timestamp_col, as_index=False)[target_name].mean()
        else:
            mapped = mapped.sort_values(timestamp_col).reset_index(drop=True)

        after_cnt = int(mapped[target_name].notna().sum())
        log_info(f"{target_name}: 纠偏后样本数={before_cnt}, 成功插入 merged_data 样本数={after_cnt}")

        final_df = final_df.merge(mapped, on=timestamp_col, how="left")
        final_df = ensure_unique_columns(final_df, context=f"{target_name} 合并后 merged_data")

    return final_df


# =========================================================
# predict_data.csv -> 特征工程
# =========================================================
def select_input_columns_from_merged(merged_df):
    """
    从 predict_data.csv 中选择用于构造特征的原始输入列：
    除时间列、目标列外，只剔除全空列。
    """
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

    log_info(f"merged_data 原始点位总数（不含时间和目标）: {len(cols) + len(all_null_cols)}")
    log_info(f"最终保留用于构造特征的原始点位数: {len(cols)}")

    if all_null_cols:
        log_info(f"以下原始点位全为空，已跳过（前50项）: {all_null_cols[:50]}")

    return cols


def create_unified_feature_matrix(
    rt_indexed,
    input_cols,
    keep_raw_inputs,
    lag_minutes,
    window_minutes,
):
    """
    统一分钟级特征矩阵：
    1) 原始输入列（可选保留）
    2) lag 特征
    3) rolling mean / std / max / min
    """
    timestamp_col = get_timestamp_col()
    rolling_min_periods = CONFIG["feature_matrix"]["rolling_min_periods"]
    progress_log_every = CONFIG["feature_matrix"]["progress_log_every"]

    feature_map = {
        timestamp_col: pd.Series(rt_indexed.index.to_numpy())
    }

    total_cols = len(input_cols)
    for idx, col in enumerate(input_cols, start=1):
        if idx == 1 or idx % progress_log_every == 0 or idx == total_cols:
            log_info(f"统一特征矩阵构造进度: {idx}/{total_cols} -> {col}")

        signal = rt_indexed[col].astype(float)

        if keep_raw_inputs:
            feature_map[col] = pd.Series(signal.to_numpy())

        for lag in lag_minutes:
            if lag <= 0:
                continue
            feature_map[f"{col}_lag_{lag}m"] = pd.Series(signal.shift(lag).to_numpy())

        for w in window_minutes:
            if w <= 0:
                continue

            rolling_obj = signal.rolling(window=w, min_periods=rolling_min_periods)
            feature_map[f"{col}_ma_{w}m"] = pd.Series(rolling_obj.mean().to_numpy())
            feature_map[f"{col}_std_{w}m"] = pd.Series(rolling_obj.std().to_numpy())
            feature_map[f"{col}_max_{w}m"] = pd.Series(rolling_obj.max().to_numpy())
            feature_map[f"{col}_min_{w}m"] = pd.Series(rolling_obj.min().to_numpy())

    feature_df = pd.DataFrame(feature_map)
    feature_df = ensure_unique_columns(feature_df, context="统一分钟级特征矩阵")
    feature_df = feature_df.sort_values(timestamp_col).reset_index(drop=True)

    return feature_df


def attach_targets_from_merged(feature_df, merged_df):
    """
    统一特征矩阵构造完成后，将 predict_data.csv 中的目标列精确合并回特征表。
    """
    timestamp_col = get_timestamp_col()
    target_cols = [c for c in get_lab_target_cols() if c in merged_df.columns]

    if not target_cols:
        raise ValueError("predict_data.csv 中不存在任何目标列，无法进行后续 XGBoost 筛选。")

    target_df = merged_df[[timestamp_col] + target_cols].copy()
    target_df = target_df.sort_values(timestamp_col).reset_index(drop=True)

    final_df = feature_df.merge(target_df, on=timestamp_col, how="left")
    final_df = ensure_unique_columns(final_df, context="特征表回并目标列后")
    final_df = final_df.sort_values(timestamp_col).reset_index(drop=True)
    return final_df


# =========================================================
# 样本抽取相关
# =========================================================
def get_feature_columns_for_training(final_df):
    """
    训练特征列 = 除时间列、所有目标列之外的其它列
    """
    timestamp_col = get_timestamp_col()
    all_target_cols = [c for c in get_lab_target_cols() if c in final_df.columns]

    feature_cols = [
        c for c in final_df.columns
        if c not in [timestamp_col] + all_target_cols
    ]
    feature_cols = deduplicate_preserve_order(feature_cols, context="training_feature_cols")
    return feature_cols


def build_target_slice_for_xgb(final_df, target_col):
    """
    只为当前目标构造一份目标非空子表，避免对整张超大宽表做重复复制。
    同时剔除在该目标样本中完全全空的特征列，降低XGBoost筛选时的内存和计算量。
    """
    timestamp_col = get_timestamp_col()

    if target_col not in final_df.columns:
        raise ValueError(f"最终表中不存在目标列: {target_col}")

    feature_cols = get_feature_columns_for_training(final_df)

    target_mask = final_df[target_col].notna()

    if not target_mask.any():
        return pd.DataFrame(columns=[timestamp_col, target_col]), []

    base_cols = [timestamp_col, target_col] + feature_cols
    base_cols = deduplicate_preserve_order(base_cols, context=f"{target_col}_base_cols")

    target_df = final_df.loc[target_mask, base_cols].copy().reset_index(drop=True)

    valid_feature_cols = []
    dropped_all_nan_cols = []

    for col in feature_cols:
        if target_df[col].notna().sum() > 0:
            valid_feature_cols.append(col)
        else:
            dropped_all_nan_cols.append(col)

    if dropped_all_nan_cols:
        log_info(
            f"{target_col} 在目标样本内全空特征数={len(dropped_all_nan_cols)}，"
            f"已剔除（前20项）: {dropped_all_nan_cols[:20]}"
        )

    keep_cols = [timestamp_col, target_col] + valid_feature_cols
    keep_cols = deduplicate_preserve_order(keep_cols, context=f"{target_col}_target_keep_cols")

    target_df = target_df.loc[:, keep_cols].copy()
    target_df = ensure_unique_columns(target_df, context=f"{target_col}_target_slice_for_xgb")

    return target_df, valid_feature_cols


def fit_xgb_and_select_features(X, y, feature_cols):
    """
    训练XGBoost模型并选择重要特征
    """
    if XGBRegressor is None:
        raise ImportError(
            "未安装 xgboost，无法执行 XGBoost 特征筛选。请先安装：pip install xgboost"
        )

    cfg = CONFIG["xgb_selection"]
    top_k = int(cfg["top_k"])
    importance_type = cfg["importance_type"]
    selection_label = f"xgb_top{top_k}"

    X = X[feature_cols].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    model_params = dict(cfg["model_params"])
    model_params["importance_type"] = importance_type

    model = XGBRegressor(**model_params)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    selected_cols = importance_df["feature"].tolist()[:top_k]

    selected_columns_df = pd.DataFrame({
        "feature": selected_cols,
        "selection_source": [selection_label] * len(selected_cols)
    })

    return selected_cols, importance_df, selected_columns_df


# =========================================================
# 所有目标筛后特征去重汇总与原始点位表
# =========================================================
def get_feature_source_raw_columns(feature_name, raw_input_cols):
    """
    获取特征对应的原始点位列名
    """
    if feature_name in raw_input_cols:
        return [feature_name]

    suffix_patterns = [
        r"(.+)_lag_\d+m$",
        r"(.+)_ma_\d+m$",
        r"(.+)_std_\d+m$",
        r"(.+)_max_\d+m$",
        r"(.+)_min_\d+m$",
    ]

    for pattern in suffix_patterns:
        matched = re.match(pattern, feature_name)
        if matched:
            base_col = matched.group(1)
            if base_col in raw_input_cols:
                return [base_col]

    return []


def save_xgb_selected_tables(final_df, raw_input_cols, output_dir):
    """
    仅生成以下结果：
    1) 各目标名_importance.csv
    2) 各目标名_xgb_training_samples.csv
    3) 所有目标筛后保留特征去重汇总表
    4) 所有目标对应原始点位去重清单表
    """
    os.makedirs(output_dir, exist_ok=True)

    paths_cfg = CONFIG["paths"]
    csv_encoding = get_csv_encoding()

    xgb_selected_dir = os.path.join(output_dir, paths_cfg["xgb_selected_subdir"])
    os.makedirs(xgb_selected_dir, exist_ok=True)

    sample_targets = get_sample_target_cols()
    xgb_training_suffix = CONFIG["sample_output"]["xgb_training_suffix"]
    xgb_feature_importance_suffix = CONFIG["sample_output"]["xgb_feature_importance_suffix"]
    all_targets_selected_features_file = CONFIG["sample_output"]["all_targets_selected_features_file"]
    all_targets_raw_points_table_file = CONFIG["sample_output"]["all_targets_raw_points_table_file"]

    xgb_enabled = CONFIG["xgb_selection"]["enabled"]

    log_step("步骤6：生成 XGBoost 筛选结果及汇总表")

    if not xgb_enabled:
        log_info("XGBoost筛选已关闭，跳过结果生成")
        return

    timestamp_col = get_timestamp_col()
    selected_feature_records = []
    selection_label = f"xgb_top{CONFIG['xgb_selection']['top_k']}"

    for target_col in sample_targets:
        if target_col not in final_df.columns:
            log_info(f"警告：最终表中不存在目标列 {target_col}，跳过")
            continue

        target_df, feature_cols = build_target_slice_for_xgb(final_df, target_col)

        if target_df.empty:
            log_info(f"[XGBoost筛选] {target_col}: 无目标样本，跳过")
            continue

        if not feature_cols:
            log_info(f"[XGBoost筛选] {target_col}: 无可用特征列，跳过")
            continue

        X = target_df[feature_cols].copy()
        y = pd.to_numeric(target_df[target_col], errors="coerce").reset_index(drop=True)
        X = X.reset_index(drop=True)
        xgb_base_df = target_df.reset_index(drop=True)

        if X.empty:
            log_info(f"[XGBoost筛选] {target_col}: 无样本，跳过")
            continue

        selected_cols, importance_df, selected_columns_df = fit_xgb_and_select_features(
            X=X,
            y=y,
            feature_cols=feature_cols,
        )

        if not selected_cols:
            log_info(f"[XGBoost筛选] {target_col}: 未选出特征，跳过")
            continue

        importance_path = os.path.join(
            xgb_selected_dir,
            f"{target_col}_{xgb_feature_importance_suffix}.csv"
        )
        importance_df.to_csv(
            importance_path,
            index=False,
            encoding=csv_encoding
        )

        xgb_keep_cols = [timestamp_col, target_col] + selected_cols
        xgb_keep_cols = deduplicate_preserve_order(
            xgb_keep_cols,
            context=f"{target_col}_xgb_keep_cols"
        )
        xgb_df = xgb_base_df.loc[:, xgb_keep_cols].copy()

        xgb_training_path = os.path.join(
            xgb_selected_dir,
            f"{target_col}_{xgb_training_suffix}.csv"
        )
        xgb_df.to_csv(
            xgb_training_path,
            index=False,
            encoding=csv_encoding
        )

        for _, row in selected_columns_df.iterrows():
            feature_name = row["feature"]
            selection_source = row["selection_source"]
            source_raw_columns = get_feature_source_raw_columns(feature_name, raw_input_cols)

            selected_feature_records.append({
                "target_name": target_col,
                "feature": feature_name,
                "selection_source": selection_source,
                "source_raw_columns": "|".join(source_raw_columns),
            })

        xgb_top_count = int((selected_columns_df["selection_source"] == selection_label).sum())

        log_info(
            f"[XGBoost筛选] {target_col}: "
            f"目标样本数={len(target_df)}, "
            f"原始特征数={len(feature_cols)}, "
            f"XGB筛选特征数={xgb_top_count}, "
            f"最终特征数={len(selected_cols)}, "
            f"训练样本数={len(xgb_df)}"
        )
        log_info(f"  特征重要性文件: {importance_path}")
        log_info(f"  筛后训练样本文件: {xgb_training_path}")

    if selected_feature_records:
        selected_feature_df = pd.DataFrame(selected_feature_records)
        selected_feature_df = selected_feature_df.sort_values(["feature", "target_name"]).reset_index(drop=True)

        dedup_feature_df = (
            selected_feature_df.groupby("feature", as_index=False)
            .agg({
                "selection_source": lambda x: "|".join(sorted(set([str(v) for v in x if pd.notna(v)]))),
                "source_raw_columns": lambda x: "|".join(
                    sorted(
                        set(
                            [
                                item
                                for v in x
                                if pd.notna(v) and str(v) != ""
                                for item in str(v).split("|")
                                if item != ""
                            ]
                        )
                    )
                ),
                "target_name": lambda x: "|".join(sorted(set([str(v) for v in x if pd.notna(v)]))),
            })
            .rename(columns={"target_name": "selected_by_targets"})
        )

        dedup_feature_path = os.path.join(xgb_selected_dir, all_targets_selected_features_file)
        dedup_feature_df.to_csv(
            dedup_feature_path,
            index=False,
            encoding=csv_encoding
        )
        log_info(f"所有目标筛后保留特征去重汇总表: {dedup_feature_path}")

        dedup_raw_points = deduplicate_preserve_order(
            [
                item.strip()
                for v in dedup_feature_df["source_raw_columns"].dropna().tolist()
                if str(v).strip() != ""
                for item in str(v).split("|")
                if item.strip() != ""
            ],
            context="all_targets_selected_raw_points"
        )

        raw_points_df = pd.DataFrame({
            "raw_point": dedup_raw_points
        })

        raw_points_table_path = os.path.join(xgb_selected_dir, all_targets_raw_points_table_file)
        raw_points_df.to_csv(
            raw_points_table_path,
            index=False,
            encoding=csv_encoding
        )
        log_info(f"所有目标对应原始点位去重清单表: {raw_points_table_path}")


# =========================================================
# 主流程
# =========================================================
def main():
    """
    主函数:执行完整的数据处理流程

    流程:
        1. 读取分钟级输入表和化验表
        2. 合并对齐生成merged_data.csv
        3. 从merged_data.csv选择原始点位
        4. 构造统一分钟级特征矩阵
        5. 将目标列合并回特征表
        6. 执行XGBoost特征筛选并保存结果
    """
    paths = CONFIG["paths"]
    timestamp_col = get_timestamp_col()
    csv_encoding = get_csv_encoding()

    os.makedirs(paths["output_dir"], exist_ok=True)
    final_output_path = os.path.join(paths["output_dir"], paths["final_output_file"])

    log_step("步骤1：读取两张输入表")
    rt_df = read_and_prepare_rt_table(paths["rt_path"])
    lab_df = read_and_prepare_lab_table(paths["lab_path"])

    log_info(f"分钟级输入表形状: {rt_df.shape[0]} 行 x {rt_df.shape[1]} 列")
    log_info(f"化验表形状: {lab_df.shape[0]} 行 x {lab_df.shape[1]} 列")

    log_step("步骤2：两表合并对齐，先生成 predict_data.csv")
    lab_long_df = build_aligned_lab_long_table(lab_df)
    merged_df = attach_lab_by_target_asof(
        base_df=rt_df,
        lab_long_df=lab_long_df,
    )
    merged_df = ensure_unique_columns(merged_df, context="merged_data")
    merged_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)

    log_info(f"merged_data 形状: {merged_df.shape[0]} 行 x {merged_df.shape[1]} 列")
    for c in [x for x in get_lab_target_cols() if x in merged_df.columns]:
        log_info(f"{c} 非空样本数: {int(merged_df[c].notna().sum())}")

    merged_df.to_csv(final_output_path, index=False, encoding=csv_encoding)
    log_info(f"predict_data.csv 已保存: {final_output_path}")

    log_step("步骤3：从 predict_data.csv 中选择原始点位并准备构造特征")
    input_cols = select_input_columns_from_merged(merged_df)
    if not input_cols:
        raise ValueError("除时间列和目标列外，其余原始点位均为空，无法构造特征。")

    log_info(f"最终用于构造特征的原始点位数: {len(input_cols)}")
    log_info("最终用于构造特征的原始点位如下：")
    print(input_cols)

    log_step("步骤4：基于 predict_data.csv 构造统一分钟级特征矩阵")
    rt_indexed = merged_df[[timestamp_col] + input_cols].copy()
    rt_indexed = rt_indexed.set_index(timestamp_col).sort_index()

    feature_cfg = CONFIG["feature_matrix"]
    feature_df = create_unified_feature_matrix(
        rt_indexed=rt_indexed,
        input_cols=input_cols,
        keep_raw_inputs=feature_cfg["keep_raw_inputs"],
        lag_minutes=feature_cfg["lag_minutes"],
        window_minutes=feature_cfg["window_minutes"],
    )

    log_info(f"统一分钟级特征矩阵形状: {feature_df.shape[0]} 行 x {feature_df.shape[1]} 列")
    log_info(f"统一特征列数（不含时间列）: {feature_df.shape[1] - 1}")

    log_step("步骤5：将 predict_data.csv 中目标列精确并回特征表")
    final_df = attach_targets_from_merged(
        feature_df=feature_df,
        merged_df=merged_df,
    )
    final_df = final_df.sort_values(timestamp_col).reset_index(drop=True)
    log_info(f"最终特征总表形状: {final_df.shape[0]} 行 x {final_df.shape[1]} 列")

    save_xgb_selected_tables(
        final_df=final_df,
        raw_input_cols=input_cols,
        output_dir=paths["output_dir"],
    )

    log_step("完成")
    log_info("本脚本当前已完成：")
    log_info("1) 先读取两张表并合并对齐，生成 predict_data.csv")
    log_info("2) 基于 predict_data.csv 构造统一时序特征")
    log_info("3) 对各目标执行 XGBoost 特征筛选")
    log_info("4) 仅保留各目标 importance、xgb_training_samples、所有目标筛后特征去重汇总表、对应原始点位去重清单表")


if __name__ == "__main__":
    main()

