#!/usr/bin/env python3
"""Calculate Accuracy Under Parallelism (AUP).

Examples:
    python calculate_aup.py 1:82.5 2:82.0 4:79.0 --alpha 5
    python calculate_aup.py --input results.csv --alpha 5
    python calculate_aup.py --input results.json --alpha 5 --json

CSV files may contain ``tpf,accuracy`` headers or simply two numeric columns.
JSON files may contain ``[[tpf, accuracy], ...]`` or objects with
``tpf``/``accuracy`` fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


Point = tuple[float, float]
TPF_KEYS = ("tpf", "rho", "parallelism")
ACCURACY_KEYS = ("accuracy", "acc", "y")


def _first_present(record: dict, keys: Iterable[str]) -> object:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    raise ValueError(f"缺少字段，候选字段为：{', '.join(keys)}")


def parse_inline(values: list[str]) -> list[Point]:
    points: list[Point] = []
    for value in values:
        try:
            rho, accuracy = value.split(":", maxsplit=1)
            points.append((float(rho), float(accuracy)))
        except ValueError as exc:
            raise ValueError(
                f"无法解析 {value!r}，行内数据格式应为 TPF:accuracy，例如 2.5:81.3"
            ) from exc
    return points


def load_csv(path: Path) -> list[Point]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if not rows:
        return []

    try:
        return [(float(row[0]), float(row[1])) for row in rows if row]
    except (ValueError, IndexError):
        pass

    header = [column.strip().lower() for column in rows[0]]
    try:
        rho_index = next(header.index(key) for key in TPF_KEYS if key in header)
        acc_index = next(header.index(key) for key in ACCURACY_KEYS if key in header)
    except StopIteration as exc:
        raise ValueError(
            "CSV 表头应包含 tpf/rho/parallelism 和 accuracy/acc/y"
        ) from exc

    try:
        return [
            (float(row[rho_index]), float(row[acc_index]))
            for row in rows[1:]
            if row
        ]
    except (ValueError, IndexError) as exc:
        raise ValueError("CSV 中存在无效或缺失的数值") from exc


def load_json(path: Path) -> list[Point]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("data", data.get("points"))
    if not isinstance(data, list):
        raise ValueError("JSON 顶层应为数组，或包含 data/points 数组")

    points: list[Point] = []
    for item in data:
        if isinstance(item, dict):
            rho = _first_present(item, TPF_KEYS)
            accuracy = _first_present(item, ACCURACY_KEYS)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            rho, accuracy = item
        else:
            raise ValueError(f"无效 JSON 数据点：{item!r}")
        points.append((float(rho), float(accuracy)))
    return points


def load_points(path: Path) -> list[Point]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_csv(path)


def calculate_aup(
    points: list[Point],
    alpha: float,
    accuracy_drop: float = 5.0,
) -> dict:
    if not points:
        raise ValueError("至少需要一个数据点")
    if alpha < 0:
        raise ValueError("alpha 必须大于或等于 0")
    if accuracy_drop < 0:
        raise ValueError("accuracy-drop 必须大于或等于 0")

    ordered = sorted(points)
    if any(rho <= 0 for rho, _ in ordered):
        raise ValueError("TPF（rho）必须大于 0")
    if len({rho for rho, _ in ordered}) != len(ordered):
        raise ValueError("TPF（rho）不能重复")
    if any(not 0 <= accuracy <= 100 for _, accuracy in ordered):
        raise ValueError("accuracy 必须位于 [0, 100]")

    y_min = ordered[0][1] - accuracy_drop
    y_max = max(accuracy for _, accuracy in ordered)
    if y_max <= 0:
        raise ValueError("最高 accuracy 必须大于 0，才能计算权重")

    included = [(rho, y) for rho, y in ordered if y >= y_min]
    weights = [
        min(math.exp(-alpha * (1.0 - y / y_max)), 1.0)
        for _, y in included
    ]

    # The first rectangle is unweighted, exactly as in the paper's definition.
    aup = included[0][0] * included[0][1]
    segments = []
    for index in range(1, len(included)):
        rho_prev, y_prev = included[index - 1]
        rho, y = included[index]
        area = (rho - rho_prev) * (
            y * weights[index] + y_prev * weights[index - 1]
        ) / 2.0
        aup += area
        segments.append(area)

    return {
        "aup": aup,
        "alpha": alpha,
        "accuracy_drop": accuracy_drop,
        "y_min": y_min,
        "y_max": y_max,
        "included_points": included,
        "excluded_points": [point for point in ordered if point not in included],
        "weights": weights,
        "segment_areas": segments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 TPF-accuracy 数据计算 Accuracy Under Parallelism (AUP)。"
    )
    parser.add_argument(
        "data",
        nargs="*",
        metavar="TPF:ACC",
        help="行内数据点，例如 1:82.5 2:82.0 4:79.0",
    )
    parser.add_argument("--input", "-i", type=Path, help="CSV 或 JSON 输入文件")
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="准确率惩罚系数（默认：3.0）",
    )
    parser.add_argument(
        "--accuracy-drop",
        type=float,
        default=5.0,
        help="相对最小 TPF 准确率允许下降的百分点（默认：5.0）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.data) == bool(args.input):
        raise SystemExit("错误：必须且只能使用行内数据或 --input 文件")

    try:
        points = load_points(args.input) if args.input else parse_inline(args.data)
        result = calculate_aup(points, args.alpha, args.accuracy_drop)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"错误：{exc}") from exc

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"AUP = {result['aup']:.6f}")
    print(
        f"alpha={result['alpha']:g}, y_max={result['y_max']:g}, "
        f"y_min={result['y_min']:g}"
    )
    print(f"纳入点数：{len(result['included_points'])}")
    if result["excluded_points"]:
        excluded = ", ".join(
            f"({rho:g}, {accuracy:g})"
            for rho, accuracy in result["excluded_points"]
        )
        print(f"排除点：{excluded}")


if __name__ == "__main__":
    main()
