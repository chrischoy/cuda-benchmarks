#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plotting style
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "seaborn")
sns.set_palette("husl")


class MatmulBenchmarkAnalyzer:
    """Analyzer and visualizer for implicit GEMM (matmul) benchmark results."""

    def __init__(self, results_file: str):
        self.results_file = results_file
        self.data = self._load_json()
        self.metadata: Dict = self.data.get("metadata", {})
        self.methods_map: Dict[int, str] = {
            int(k): v for k, v in self.data.get("methods", {}).items()
        }
        # Parsed as {(Nrows, Ccols): {method_id: metrics_dict}}
        self.results: Dict[Tuple[int, int], Dict[int, Dict]] = self._parse_results(
            self.data.get("results", {})
        )

    def _load_json(self) -> Dict:
        try:
            with open(self.results_file, "r") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Error loading JSON: {exc}")
            return {}

    @staticmethod
    def _parse_key(size_key: str) -> Optional[Tuple[int, int]]:
        """Parse keys like 'N16384_C256' -> (16384, 256)."""
        match = re.match(r"^N(\d+)_C(\d+)$", size_key)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _parse_results(
        self, results_obj: Dict
    ) -> Dict[Tuple[int, int], Dict[int, Dict]]:
        parsed: Dict[Tuple[int, int], Dict[int, Dict]] = {}
        for size_key, methods in results_obj.items():
            size_tuple = self._parse_key(size_key)
            if size_tuple is None:
                continue
            Nrows, Ccols = size_tuple
            parsed[size_tuple] = {}
            flops = self._compute_total_flops(
                Nrows, Ccols
            )  # scalar count (not in FLOPs/s)
            for method_id_str, stats in methods.items():
                try:
                    method_id = int(method_id_str)
                except Exception:
                    # Some dumps may already have integer keys
                    method_id = method_id_str  # type: ignore
                # Normalize expected keys from generator: mean_ms, std_ms, min_ms
                mean_ms = float(stats.get("mean_ms", stats.get("mean_time_ms", np.nan)))
                std_ms = float(stats.get("std_ms", stats.get("std_time_ms", np.nan)))
                min_ms = float(stats.get("min_ms", stats.get("min_time_ms", np.nan)))

                # Compute throughput in TFLOPs/s based on min and mean times
                min_tflops = self._ms_to_tflops(flops, min_ms)
                mean_tflops = self._ms_to_tflops(flops, mean_ms)

                parsed[size_tuple][method_id] = {
                    "method": stats.get(
                        "method", self.methods_map.get(method_id, str(method_id))
                    ),
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "min_ms": min_ms,
                    "min_tflops": min_tflops,
                    "mean_tflops": mean_tflops,
                }
        return parsed

    @staticmethod
    def _compute_total_flops(Nrows: int, Ccols: int) -> float:
        """Total floating point operations for this problem size.

        Operation: for each of P=Nrows gathered rows, compute A[a,:] @ B (C x C) + bias, then scatter-add.
        GEMM-like work is vector(C) x matrix(C x C) => ~2*C*C FLOPs per gathered row.
        Total FLOPs = 2 * Nrows * Ccols * Ccols.
        """
        return 2.0 * float(Nrows) * float(Ccols) * float(Ccols)

    @staticmethod
    def _ms_to_tflops(total_flops: float, time_ms: float) -> float:
        if not np.isfinite(time_ms) or time_ms <= 0:
            return float("nan")
        flops_per_s = total_flops / (time_ms * 1e-3)
        return flops_per_s / 1e12

    # ---------------------------- Printing ---------------------------------
    def print_metadata(self) -> None:
        print("=" * 80)
        print("BENCHMARK METADATA")
        print("=" * 80)
        print(f"Timestamp: {self.metadata.get('timestamp', 'Unknown')}")
        print(f"GPU: {self.metadata.get('gpu_name', 'Unknown')}")
        print(f"PyTorch Version: {self.metadata.get('pytorch_version', 'Unknown')}")
        print(
            f"Methods: {', '.join([f'{k}:{v}' for k, v in sorted(self.methods_map.items())])}"
        )
        print()

    def print_statistics(self, stat_type: str = "all") -> None:
        print("=" * 80)
        print(f"STATISTICS ({stat_type.upper()})")
        print("=" * 80)

        sizes_sorted = sorted(self.results.keys())
        # Headers
        if stat_type == "all":
            header = (
                f"{'Size (N,C)':<16} {'Method':<22} "
                f"{'Min (ms)':>10} {'Mean (ms)':>12} {'Std (ms)':>10} "
                f"{'Min (TFLOP/s)':>15} {'Mean (TFLOP/s)':>17}"
            )
        elif stat_type == "min":
            header = f"{'Size (N,C)':<16} {'Method':<22} {'Min (ms)':>10} {'Min (TFLOP/s)':>15}"
        elif stat_type == "mean":
            header = f"{'Size (N,C)':<16} {'Method':<22} {'Mean (ms)':>12} {'Mean (TFLOP/s)':>17}"
        elif stat_type == "std":
            header = f"{'Size (N,C)':<16} {'Method':<22} {'Std (ms)':>10}"
        else:
            print("Unknown stat_type; use one of: all|min|mean|std")
            return
        print(header)
        print("-" * len(header))

        for Nrows, Ccols in sizes_sorted:
            methods = self.results[(Nrows, Ccols)]
            for method_id, data in sorted(methods.items()):
                size_str = f"{Nrows}x{Ccols}"
                method_name = data["method"]
                if stat_type == "all":
                    print(
                        f"{size_str:<16} {method_name:<22} "
                        f"{data['min_ms']:>10.3f} {data['mean_ms']:>12.3f} {data['std_ms']:>10.3f} "
                        f"{data['min_tflops']:>15.2f} {data['mean_tflops']:>17.2f}"
                    )
                elif stat_type == "min":
                    print(
                        f"{size_str:<16} {method_name:<22} "
                        f"{data['min_ms']:>10.3f} {data['min_tflops']:>15.2f}"
                    )
                elif stat_type == "mean":
                    print(
                        f"{size_str:<16} {method_name:<22} "
                        f"{data['mean_ms']:>12.3f} {data['mean_tflops']:>17.2f}"
                    )
                elif stat_type == "std":
                    print(f"{size_str:<16} {method_name:<22} {data['std_ms']:>10.3f}")

    def print_rankings(self, top_n: int = 3, metric: str = "tflops") -> None:
        """Rank methods per (N,C) by metric.

        metric: 'tflops' (descending), 'min_time' (ascending), 'mean_time' (ascending)
        """
        print("=" * 80)
        print(f"TOP {top_n} PERFORMERS BY {metric.upper()}")
        print("=" * 80)

        header = ["Size (N,C)".ljust(16)]
        for i in range(top_n):
            suffix = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i + 1}th"
            header.append(f"{suffix} Place".ljust(32))
        print(" | ".join(header) + " |")
        print("-" * (16 + 3 + (32 + 3) * top_n))

        for size in sorted(self.results.keys()):
            methods = self.results[size]
            scored: List[Tuple[float, Dict]] = []
            reverse = True
            for _, data in methods.items():
                if metric == "tflops":
                    score = data["min_tflops"]
                    reverse = True
                elif metric == "min_time":
                    score = data["min_ms"]
                    reverse = False
                elif metric == "mean_time":
                    score = data["mean_ms"]
                    reverse = False
                else:
                    continue
                scored.append((score, data))
            scored.sort(reverse=reverse, key=lambda x: x[0])

            row = [f"{size[0]}x{size[1]:<6}"]
            for i in range(top_n):
                if i < len(scored):
                    s, d = scored[i]
                    if metric == "tflops":
                        val = f"{d['min_tflops']:.2f} TF/s"
                    else:
                        key = "min_ms" if metric == "min_time" else "mean_ms"
                        val = f"{d[key]:.3f} ms"
                    row.append(f"{d['method']} ({val})".ljust(32))
                else:
                    row.append("N/A".ljust(32))
            print(" | ".join(row) + " |")

    # --------------------------- Visualization -----------------------------
    def _collect_all_methods(self) -> List[str]:
        methods = set()
        for _, md in self.results.items():
            for _, d in md.items():
                methods.add(d["method"])
        return sorted(methods)

    def plot_performance_vs_n(
        self,
        fixed_channels: List[int],
        metric: str = "tflops",
        save_path: Optional[str] = None,
        method_filter: Optional[str] = None,
    ) -> None:
        if not self.results:
            print("No results to plot")
            return

        all_methods = self._collect_all_methods()
        if method_filter:
            all_methods = [m for m in all_methods if method_filter.lower() in m.lower()]

        # X-axis: N, group by channel C in subplots
        n_values = sorted({N for (N, C) in self.results.keys() if C in fixed_channels})
        if not n_values:
            print("No matching sizes for requested channels")
            return

        fig, axes = plt.subplots(1, min(4, len(fixed_channels)), figsize=(16, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        fig.suptitle(f"Performance vs N (metric: {metric})", fontsize=14)

        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "+", "x"]

        for ax_idx, C in enumerate(fixed_channels[: len(axes)]):
            ax = axes[ax_idx]
            for m_idx, method_name in enumerate(all_methods):
                xs, ys = [], []
                for N in n_values:
                    data = self.results.get((N, C))
                    if not data:
                        continue
                    # Find method entry by name
                    md = next(
                        (d for d in data.values() if d["method"] == method_name), None
                    )
                    if md is None:
                        continue
                    xs.append(N)
                    if metric == "tflops":
                        ys.append(md["min_tflops"])
                    elif metric == "min_time":
                        ys.append(md["min_ms"])  # ms
                    elif metric == "mean_time":
                        ys.append(md["mean_ms"])  # ms
                if xs and ys:
                    ax.plot(
                        xs,
                        ys,
                        marker=markers[m_idx % len(markers)],
                        label=method_name,
                        linewidth=2,
                        markersize=4,
                    )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Rows (N)")
            if metric == "tflops":
                ax.set_ylabel("TFLOP/s (higher is better)")
            else:
                ax.set_ylabel("Time (ms, lower is better)")
            ax.set_title(f"Channels C = {C}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_performance_vs_c(
        self,
        fixed_rows: List[int],
        metric: str = "tflops",
        save_path: Optional[str] = None,
        method_filter: Optional[str] = None,
    ) -> None:
        if not self.results:
            print("No results to plot")
            return

        all_methods = self._collect_all_methods()
        if method_filter:
            all_methods = [m for m in all_methods if method_filter.lower() in m.lower()]

        c_values = sorted({C for (N, C) in self.results.keys() if N in fixed_rows})
        if not c_values:
            print("No matching sizes for requested rows")
            return

        fig, axes = plt.subplots(1, min(4, len(fixed_rows)), figsize=(16, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        fig.suptitle(f"Performance vs C (metric: {metric})", fontsize=14)

        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "+", "x"]

        for ax_idx, N in enumerate(fixed_rows[: len(axes)]):
            ax = axes[ax_idx]
            for m_idx, method_name in enumerate(all_methods):
                xs, ys = [], []
                for C in c_values:
                    data = self.results.get((N, C))
                    if not data:
                        continue
                    md = next(
                        (d for d in data.values() if d["method"] == method_name), None
                    )
                    if md is None:
                        continue
                    xs.append(C)
                    if metric == "tflops":
                        ys.append(md["min_tflops"])  # TFLOP/s
                    elif metric == "min_time":
                        ys.append(md["min_ms"])  # ms
                    elif metric == "mean_time":
                        ys.append(md["mean_ms"])  # ms
                if xs and ys:
                    ax.plot(
                        xs,
                        ys,
                        marker=markers[m_idx % len(markers)],
                        label=method_name,
                        linewidth=2,
                        markersize=4,
                    )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Channels (C)")
            if metric == "tflops":
                ax.set_ylabel("TFLOP/s (higher is better)")
            else:
                ax.set_ylabel("Time (ms, lower is better)")
            ax.set_title(f"Rows N = {N}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_heatmap(
        self,
        metric: str = "tflops",
        method_filter: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Create a single heatmap with methods on rows and sizes on columns, similar to other analyzers."""
        methods = self._collect_all_methods()
        if method_filter:
            methods = [m for m in methods if method_filter.lower() in m.lower()]
        if not methods:
            print("No methods to plot")
            return

        # Columns are sizes labeled as "N x C"
        sizes = sorted(self.results.keys())
        size_labels = [f"{N}x{C}" for (N, C) in sizes]

        data_matrix = np.full((len(methods), len(sizes)), np.nan)
        for size_idx, size in enumerate(sizes):
            mdict = self.results.get(size, {})
            for method_idx, method_name in enumerate(methods):
                md = next(
                    (d for d in mdict.values() if d["method"] == method_name), None
                )
                if md is None:
                    continue
                if metric == "tflops":
                    value = md["min_tflops"]
                elif metric == "min_time":
                    value = md["min_ms"]
                else:
                    value = md["mean_ms"]
                data_matrix[method_idx, size_idx] = value

        plt.figure(figsize=(max(12, len(sizes) * 0.8), max(6, len(methods) * 0.5)))
        mask = np.isnan(data_matrix)
        sns.heatmap(
            data_matrix,
            xticklabels=size_labels,
            yticklabels=methods,
            annot=True,
            fmt=".2f" if metric == "tflops" else ".3f",
            cmap="viridis" if metric == "tflops" else "viridis_r",
            mask=mask,
            cbar_kws={
                "label": "TFLOP/s" if metric == "tflops" else "Time (ms)",
            },
        )
        plt.title(f"Matmul Performance Heatmap — {metric}")
        plt.xlabel("Size (N x C)")
        plt.ylabel("Method")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize implicit GEMM (matmul) benchmark results"
    )
    parser.add_argument(
        "results_file", help="Path to results JSON (from test_benchmark.py)"
    )

    # Info printing
    info = parser.add_argument_group("information")
    info.add_argument("--metadata", action="store_true", help="Print metadata")
    info.add_argument(
        "--stats",
        choices=["all", "min", "mean", "std"],
        default="all",
        help="Statistics to print (default: all)",
    )
    info.add_argument(
        "--no-stats", action="store_true", help="Do not print statistics table"
    )
    info.add_argument(
        "--rankings", type=int, default=3, help="Show top-N rankings (default: 3)"
    )
    info.add_argument(
        "--no-rankings", action="store_true", help="Do not print rankings"
    )

    # Visualization
    viz = parser.add_argument_group("visualization")
    viz.add_argument(
        "--plot-vs-n",
        action="store_true",
        help="Plot performance vs N for fixed channel counts",
    )
    viz.add_argument(
        "--plot-vs-c",
        action="store_true",
        help="Plot performance vs C for fixed row counts",
    )
    viz.add_argument(
        "--plot-heatmap", action="store_true", help="Generate performance heatmap"
    )
    viz.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Channel counts C for --plot-vs-n (default: 32 64 128 256)",
    )
    viz.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[16384, 262144, 1048576],
        help="Row counts N for --plot-vs-c (default: 16384 262144 1048576)",
    )
    viz.add_argument(
        "--metric",
        choices=["tflops", "min_time", "mean_time"],
        default="tflops",
        help="Plotting and ranking metric (default: tflops)",
    )
    viz.add_argument(
        "--method-filter",
        type=str,
        default=None,
        help="Filter methods by substring for plots",
    )
    viz.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to default files under a 'plots' directory next to the results file",
    )

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: results file not found: {args.results_file}")
        return

    analyzer = MatmulBenchmarkAnalyzer(args.results_file)

    if args.metadata:
        analyzer.print_metadata()

    if not args.no_stats:
        analyzer.print_statistics(args.stats)

    if not args.no_rankings:
        analyzer.print_rankings(top_n=args.rankings, metric=args.metric)

    # Plots
    if args.plot_vs_n:
        save_path = None
        if args.save_plots:
            plot_dir = Path(args.results_file).parent / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(plot_dir / f"matmul_performance_vs_n_{args.metric}.png")
        analyzer.plot_performance_vs_n(
            fixed_channels=args.channels,
            metric=args.metric,
            save_path=save_path,
            method_filter=args.method_filter,
        )

    if args.plot_vs_c:
        save_path = None
        if args.save_plots:
            plot_dir = Path(args.results_file).parent / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(plot_dir / f"matmul_performance_vs_c_{args.metric}.png")
        analyzer.plot_performance_vs_c(
            fixed_rows=args.rows,
            metric=args.metric,
            save_path=save_path,
            method_filter=args.method_filter,
        )

    if args.plot_heatmap:
        save_path = None
        if args.save_plots:
            plot_dir = Path(args.results_file).parent / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            method_suffix = f"_{args.method_filter}" if args.method_filter else ""
            save_path = str(
                plot_dir / f"matmul_heatmap_{args.metric}{method_suffix}.png"
            )
        analyzer.plot_heatmap(
            metric=args.metric,
            method_filter=args.method_filter,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
