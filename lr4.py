"""
Лабораторная работа № 4. Вариант 17.
Последовательный, двоичный и интерполяционный поиск.
Распределение: С (степенное). n: 5 точек от 10^3 до 10^5.
k = 1500 успешных запросов. Доп. задание Б3 — точка окупаемости сортировки.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT = Path("/home/workdir/artifacts/lr4")
CODE = OUT / "code"
DIAG = OUT / "diagrams"
CODE.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)


def linear_search(a, q):
    """Последовательный поиск. Возвращает (индекс, число сравнений)."""
    for i, value in enumerate(a):
        if value == q:
            return i, i + 1
    return -1, len(a)


def binary_search(a, q):
    """Двоичный поиск в упорядоченном массиве."""
    lo, hi, cmps = 0, len(a) - 1, 0
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        cmps += 1
        if a[mid] == q:
            return mid, cmps
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, cmps


def interpolation_search(a, q):
    """Интерполяционный поиск в упорядоченном массиве."""
    lo, hi, cmps = 0, len(a) - 1, 0
    while lo <= hi and a[lo] <= q <= a[hi]:
        if a[lo] == a[hi]:
            cmps += 1
            return (lo, cmps) if a[lo] == q else (-1, cmps)
        mid = lo + (q - a[lo]) * (hi - lo) // (a[hi] - a[lo])
        # защита от выхода за границы из-за целочисленного деления
        mid = max(lo, min(mid, hi))
        cmps += 1
        if a[mid] == q:
            return mid, cmps
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, cmps


def make_power_data(n, seed=42):
    """
    Степенное распределение С: floor(10^6 * u^3), u ~ U[0,1).
    Возвращает отсортированный массив (для двоичного/интерполяционного).
    Могут быть дубликаты — это нормально для степенного закона.
    """
    rng = random.Random(seed + n)
    data = [int(1_000_000 * (rng.random() ** 3)) for _ in range(n)]
    data.sort()
    return data


def make_queries(data, k, seed=1):
    """k успешных запросов — случайные элементы из data."""
    rng = random.Random(seed)
    return [rng.choice(data) for _ in range(k)]


def benchmark(search_func, data, queries):
    """Среднее время одного запроса (с) и среднее число сравнений."""
    start = time.perf_counter()
    total_cmps = 0
    for q in queries:
        _, cmps = search_func(data, q)
        total_cmps += cmps
    elapsed = time.perf_counter() - start
    return elapsed / len(queries), total_cmps / len(queries)


def measure_sort_time(data):
    """Время сортировки копии (Timsort)."""
    a = list(data)
    t0 = time.perf_counter()
    a.sort()
    return time.perf_counter() - t0


def break_even_experiment(n, k_values, seed=7):
    """
    Б3: для фиксированного n сравниваем
    T_no_sort(k) = k * t_linear
    T_with_sort(k) = t_sort + k * t_binary
    и ищем k, где графики пересекаются.
    """
    rng = random.Random(seed)
    # неотсортированные данные степенного распределения
    raw = [int(1_000_000 * (rng.random() ** 3)) for _ in range(n)]
    sorted_data = sorted(raw)

    # калибровка среднего времени одного запроса
    queries_cal = [rng.choice(sorted_data) for _ in range(2000)]
    t_lin, _ = benchmark(linear_search, raw, queries_cal)  # на неотсортированном
    t_bin, _ = benchmark(binary_search, sorted_data, queries_cal)
    t_sort = measure_sort_time(raw)

    results = []
    for k in k_values:
        t_no = k * t_lin
        t_yes = t_sort + k * t_bin
        results.append({"k": k, "no_sort": t_no, "with_sort": t_yes})

    # точное пересечение: k * t_lin = t_sort + k * t_bin
    # k * (t_lin - t_bin) = t_sort
    if t_lin > t_bin:
        k_star = t_sort / (t_lin - t_bin)
    else:
        k_star = float("inf")

    return {
        "n": n,
        "t_linear": t_lin,
        "t_binary": t_bin,
        "t_sort": t_sort,
        "k_star": k_star,
        "curve": results,
    }


def main():
    # Вариант 17: 5 точек от 10^3 до 10^5 (логарифмическая шкала)
    SIZES = [1000, 3162, 10000, 31623, 100000]
    K = 1500
    rng = random.Random(1)

    algorithms = {
        "Последовательный": linear_search,
        "Двоичный": binary_search,
        "Интерполяционный": interpolation_search,
    }

    exp_results = {"sizes": SIZES, "k": K, "times": {}, "cmps": {}}
    for name in algorithms:
        exp_results["times"][name] = []
        exp_results["cmps"][name] = []

    print("=== Эксперимент: поиск (степенное распределение, успешные запросы) ===")
    print(f"{'n':>8}" + "".join(f"{name:>18}" for name in algorithms))
    for n in SIZES:
        data = make_power_data(n)
        queries = make_queries(data, K)
        row_t = f"{n:>8}"
        for name, func in algorithms.items():
            t, c = benchmark(func, data, queries)
            exp_results["times"][name].append(t)
            exp_results["cmps"][name].append(c)
            row_t += f"{t*1e6:>12.2f}мкс/{c:>5.1f}"
        print(row_t)

    # Б3: точка окупаемости при n = 100000
    print("\n=== Б3: точка окупаемости сортировки (n = 100000) ===")
    k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
    be = break_even_experiment(100_000, k_values)
    print(f"t_linear  = {be['t_linear']*1e6:.3f} мкс/запрос")
    print(f"t_binary  = {be['t_binary']*1e6:.3f} мкс/запрос")
    print(f"t_sort    = {be['t_sort']*1e3:.3f} мс")
    print(f"k* (пересечение) ≈ {be['k_star']:.1f}")

    out = {
        "exp": exp_results,
        "break_even": {
            "n": be["n"],
            "t_linear": be["t_linear"],
            "t_binary": be["t_binary"],
            "t_sort": be["t_sort"],
            "k_star": be["k_star"],
            "curve": be["curve"],
        },
    }
    with open(CODE / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nresults.json сохранён")

    # --- Графики ---
    font_path = "/usr/share/fonts/SlidesCarnival/google/Tinos/Tinos-Regular.ttf"
    font_bold = "/usr/share/fonts/SlidesCarnival/google/Tinos/Tinos-Bold.ttf"
    prop = fm.FontProperties(fname=font_path) if Path(font_path).exists() else None
    prop_b = fm.FontProperties(fname=font_bold) if Path(font_bold).exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Время
    for name, times in exp_results["times"].items():
        axes[0].plot(SIZES, [t * 1e6 for t in times], marker="o", label=name)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Размер массива n", fontproperties=prop)
    axes[0].set_ylabel("Среднее время запроса, мкс", fontproperties=prop)
    axes[0].set_title("Среднее время одного запроса", fontproperties=prop_b)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(prop=prop, fontsize=9)

    # Число сравнений
    for name, cmps in exp_results["cmps"].items():
        axes[1].plot(SIZES, cmps, marker="o", label=name)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Размер массива n", fontproperties=prop)
    axes[1].set_ylabel("Среднее число сравнений", fontproperties=prop)
    axes[1].set_title("Среднее число сравнений на запрос", fontproperties=prop_b)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(prop=prop, fontsize=9)

    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            if prop:
                label.set_fontproperties(prop)
            label.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(DIAG / "exp_search.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("exp_search.png сохранён")

    # Б3 график
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ks = [p["k"] for p in be["curve"]]
    no_s = [p["no_sort"] * 1e3 for p in be["curve"]]  # мс
    yes_s = [p["with_sort"] * 1e3 for p in be["curve"]]
    ax2.plot(ks, no_s, "o-", label="Без сортировки (k · t_lin)", color="#c62828")
    ax2.plot(ks, yes_s, "s-", label="С сортировкой (t_sort + k · t_bin)", color="#1565c0")
    if be["k_star"] < max(ks) * 2:
        ax2.axvline(be["k_star"], color="#2e7d32", linestyle="--",
                    label=f"k* ≈ {be['k_star']:.1f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Число запросов k", fontproperties=prop)
    ax2.set_ylabel("Суммарное время, мс", fontproperties=prop)
    ax2.set_title(f"Точка окупаемости сортировки (n = {be['n']})", fontproperties=prop_b)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(prop=prop, fontsize=9)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        if prop:
            label.set_fontproperties(prop)
        label.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(DIAG / "break_even.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("break_even.png сохранён")


if __name__ == "__main__":
    main()
