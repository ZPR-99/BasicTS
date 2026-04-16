import os
import time
from datetime import datetime

import numpy as np
import pandas as pd


# =========================
# 配置区
# =========================
CONFIG = {
    # 输入：清洗后的数据
    "input_file_path": r"D:\云天化\软仪表\天安\二期磷酸\model\date\clean_date\merged_cleaned.csv",

    # 输出：插值后的数据
    "output_file_path": r"D:\云天化\软仪表\天安\二期磷酸\model\date\input_date\merged_interpolated.csv",

    # 输出：插值统计报告
    "report_file_path": r"D:\云天化\软仪表\天安\二期磷酸\model\date\input_date\merged_interpolation_report.csv",

    # 列有效数据比例阈值：低于该比例的列不做插值
    "threshold_ratio": 0.4,

    # 连续缺失阈值：<= 120 认为是短时丢包，可插值；> 120 认为是停机/故障，保留 NaN
    "max_gap_threshold": 120,

    # 时间列名
    "timestamp_col": "Timestamp",

    # 是否保留原始列顺序
    "keep_original_column_order": True,

    # 期望时间频率（用于补齐缺失时间戳）
    "time_frequency": "1min",
}


# =========================
# 日志函数
# =========================
def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# =========================
# 工具函数
# =========================
def ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_na_runs_mask(series: pd.Series, gap_threshold: int) -> pd.Series:
    """
    返回“长缺失段”的掩码：
    连续 NaN 段长度 > gap_threshold 的位置标 True
    """
    is_na = series.isna()

    if not is_na.any():
        return pd.Series(False, index=series.index)

    groups = is_na.ne(is_na.shift(fill_value=False)).cumsum()
    run_lengths = is_na.groupby(groups).transform("sum")

    long_gap_mask = is_na & (run_lengths > gap_threshold)
    return long_gap_mask


def calc_gap_stats(series: pd.Series, gap_threshold: int):
    """
    统计连续缺失段信息
    """
    is_na = series.isna()
    if not is_na.any():
        return {
            "na_count": 0,
            "na_ratio": 0.0,
            "gap_count": 0,
            "max_gap": 0,
            "long_gap_count": 0,
        }

    groups = is_na.ne(is_na.shift(fill_value=False)).cumsum()
    gap_lengths = is_na.groupby(groups).sum()
    gap_lengths = gap_lengths[gap_lengths > 0]

    return {
        "na_count": int(is_na.sum()),
        "na_ratio": float(is_na.mean()),
        "gap_count": int(len(gap_lengths)),
        "max_gap": int(gap_lengths.max()) if len(gap_lengths) > 0 else 0,
        "long_gap_count": int((gap_lengths > gap_threshold).sum()),
    }


def smart_interpolate_column(series: pd.Series, gap_threshold: int) -> pd.Series:
    """
    单列智能插值逻辑：
    1. 仅对区间内部缺失值做线性插值（不做两端外推）
    2. 对连续缺失长度 > gap_threshold 的停机段恢复为 NaN
    """
    if series.isna().sum() == 0:
        return series

    # 长停机段掩码
    long_gap_mask = get_na_runs_mask(series, gap_threshold)

    # 只对内部区间插值，避免头尾“硬补”
    interpolated = series.interpolate(
        method="time",
        limit_direction="both",
        limit_area="inside"
    )

    # 长停机段回滚为 NaN
    final_series = interpolated.mask(long_gap_mask, np.nan)

    return final_series


# =========================
# 主流程
# =========================
def process_data():
    start_time = time.time()
    log("=== 开始插值任务 ===")

    input_path = CONFIG["input_file_path"]
    output_path = CONFIG["output_file_path"]
    report_path = CONFIG["report_file_path"]
    timestamp_col = CONFIG["timestamp_col"]
    threshold_ratio = CONFIG["threshold_ratio"]
    gap_threshold = CONFIG["max_gap_threshold"]

    if not os.path.exists(input_path):
        log(f"输入文件不存在: {input_path}", level="ERROR")
        return

    log(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)

    log(f"原始数据形状: {df.shape}")

    if timestamp_col not in df.columns:
        log(f"缺少时间列: {timestamp_col}", level="ERROR")
        return

    original_columns = df.columns.tolist()

    # 1. 时间列解析、排序、去重
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    bad_ts = df[timestamp_col].isna().sum()
    if bad_ts > 0:
        log(f"警告: 时间列存在 {bad_ts} 条无法解析记录，将删除", level="WARNING")
        df = df.dropna(subset=[timestamp_col])

    before_dedup = len(df)
    df = (
        df.sort_values(timestamp_col)
          .drop_duplicates(subset=[timestamp_col], keep="first")
    )
    dedup_removed = before_dedup - len(df)
    if dedup_removed > 0:
        log(f"时间戳重复记录已去除: {dedup_removed} 行", level="WARNING")

    # 2. 建立时间索引，并按预期频率补齐缺失时间戳
    df = df.set_index(timestamp_col).sort_index()

    time_frequency = CONFIG.get("time_frequency", "1min")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=time_frequency)
    missing_ts_count = int(len(full_index.difference(df.index)))
    if missing_ts_count > 0:
        log(f"检测到缺失时间戳 {missing_ts_count} 个，按频率 {time_frequency} 补齐时间轴", level="WARNING")
    df = df.reindex(full_index)
    df.index.name = timestamp_col

    # 3. 识别数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if timestamp_col in numeric_cols:
        numeric_cols.remove(timestamp_col)

    log(f"识别到数值列数: {len(numeric_cols)}")

    # 4. 按有效率筛选待处理列
    valid_ratios = df[numeric_cols].notna().mean()
    cols_to_process = valid_ratios[valid_ratios >= threshold_ratio].index.tolist()
    cols_skipped = [c for c in numeric_cols if c not in cols_to_process]

    log(f"有效率 >= {threshold_ratio:.0%} 的待处理列数: {len(cols_to_process)}")
    log(f"跳过列数: {len(cols_skipped)}")

    # 5. 执行逐列插值并生成报告
    report_rows = []

    total_start_nulls = int(df[cols_to_process].isna().sum().sum()) if cols_to_process else 0

    for i, col in enumerate(cols_to_process, start=1):
        s_before = df[col].copy()

        before_stats = calc_gap_stats(s_before, gap_threshold)
        s_after = smart_interpolate_column(s_before, gap_threshold)
        after_stats = calc_gap_stats(s_after, gap_threshold)

        filled_count = int(s_before.isna().sum() - s_after.isna().sum())

        df[col] = s_after

        report_rows.append({
            "column": col,
            "valid_ratio": round(float(valid_ratios[col]), 6),

            "na_count_before": before_stats["na_count"],
            "na_ratio_before": round(before_stats["na_ratio"], 6),
            "gap_count_before": before_stats["gap_count"],
            "max_gap_before": before_stats["max_gap"],
            "long_gap_count_before": before_stats["long_gap_count"],

            "na_count_after": after_stats["na_count"],
            "na_ratio_after": round(after_stats["na_ratio"], 6),
            "gap_count_after": after_stats["gap_count"],
            "max_gap_after": after_stats["max_gap"],
            "long_gap_count_after": after_stats["long_gap_count"],

            "filled_count": filled_count,
            "processed": 1,
        })

        if i % 10 == 0 or i == len(cols_to_process):
            log(f"插值进度: {i}/{len(cols_to_process)}")

    # 对跳过列也记录到报告里
    for col in cols_skipped:
        s = df[col]
        stats = calc_gap_stats(s, gap_threshold)
        report_rows.append({
            "column": col,
            "valid_ratio": round(float(valid_ratios[col]), 6),

            "na_count_before": stats["na_count"],
            "na_ratio_before": round(stats["na_ratio"], 6),
            "gap_count_before": stats["gap_count"],
            "max_gap_before": stats["max_gap"],
            "long_gap_count_before": stats["long_gap_count"],

            "na_count_after": stats["na_count"],
            "na_ratio_after": round(stats["na_ratio"], 6),
            "gap_count_after": stats["gap_count"],
            "max_gap_after": stats["max_gap"],
            "long_gap_count_after": stats["long_gap_count"],

            "filled_count": 0,
            "processed": 0,
        })

    report_df = pd.DataFrame(report_rows)

    total_end_nulls = int(df[cols_to_process].isna().sum().sum()) if cols_to_process else 0
    total_filled = total_start_nulls - total_end_nulls

    # 6. 删除全空列（保留时间列）
    all_nan_cols = [
        col for col in df.columns
        if col != timestamp_col and df[col].isna().all()
    ]

    if all_nan_cols:
        log(f"删除全空列数: {len(all_nan_cols)}")
        log(f"全空列: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # 在报告中标记是否被删除
    report_df["dropped_all_nan"] = report_df["column"].isin(all_nan_cols).astype(int)

    # 排序报告
    report_df = report_df.sort_values(
        by=["dropped_all_nan", "filled_count", "na_count_after"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # 7. 恢复列顺序（仅保留未删除列）
    if CONFIG["keep_original_column_order"]:
        final_columns = [c for c in original_columns if c in df.columns]
        df = df[final_columns]

    df = df.reset_index()

    # 8. 保存结果
    ensure_parent_dir(output_path)
    ensure_parent_dir(report_path)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    # 9. 总结
    log("=== 插值完成 ===")
    log(f"输出文件: {output_path}")
    log(f"报告文件: {report_path}")
    log(f"最终数据形状: {df.shape}")
    log(f"处理列数: {len(cols_to_process)}")
    log(f"跳过列数: {len(cols_skipped)}")
    log(f"删除全空列数: {len(all_nan_cols)}")
    log(f"插值前空值总数(待处理列): {total_start_nulls}")
    log(f"插值后空值总数(待处理列): {total_end_nulls}")
    log(f"实际填充数量: {total_filled}")
    log(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    process_data()

