#!/usr/bin/env python3

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "seaborn")
sns.set_palette("husl")


class GatherBenchmarkAnalyzer:
    """Comprehensive gather benchmark results analyzer"""

    def __init__(self, results_file: str):
        """Initialize analyzer with results file"""
        self.results_file = results_file
        self.data = self._load_results()
        self.metadata = self.data.get("mxe", {})  # Changed from "metadata" to "mxe"
        self.gather_results = self._parse_results(
            self.data.get("gxa")
        )  # Changed from "gather_results" to "gxa"

    def _load_results(self) -> Dict:
        """Load results from JSON file"""
        try:
            with open(self.results_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results file: {e}")
            return {}

    def _parse_results(self, results: Optional[Dict]) -> Optional[Dict]:
        """Parse results and convert string keys back to tuples and floats"""
        if not results:
            return None

        parsed_results = {}
        for size_str, gather_ratios in results.items():
            # Convert "(1024, 8)" back to (1024, 8)
            # Remove parentheses and split by comma
            size_str_clean = size_str.strip("()")
            rows, cols = map(int, size_str_clean.split(", "))
            parsed_results[(rows, cols)] = {}

            for gather_ratio_str, methods in gather_ratios.items():
                # Convert gather ratio string back to float
                gather_ratio = float(gather_ratio_str)
                parsed_results[(rows, cols)][gather_ratio] = methods

        return parsed_results

    def print_metadata(self):
        """Print benchmark metadata"""
        print("=" * 80)
        print("GATHER BENCHMARK METADATA")
        print("=" * 80)
        print(f"Timestamp: {self.metadata.get('timestamp', 'Unknown')}")
        print(f"GPU: {self.metadata.get('gpu_name', 'Unknown')}")
        print(f"PyTorch Version: {self.metadata.get('pytorch_version', 'Unknown')}")
        print(f"Benchmark Type: {self.metadata.get('benchmark_type', 'Unknown')}")
        print()

    def print_statistics(self, stat_type: str = "all"):
        """Print comprehensive statistics

        Args:
            stat_type: "min", "mean", "std", or "all"
        """
        print("=" * 80)
        print(f"GATHER BENCHMARK STATISTICS ({stat_type.upper()})")
        print("=" * 80)

        if not self.gather_results:
            print("No gather results available")
            return

        # Create header
        if stat_type == "all":
            header = f"{'Matrix Size':<15} {'Gather %':<10} {'Method':<25} {'Min (ms)':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'BW (GB/s)':<12}"
            print(header)
            print("-" * len(header))
        elif stat_type == "min":
            header = f"{'Matrix Size':<15} {'Gather %':<10} {'Method':<25} {'Min (ms)':<12} {'BW (GB/s)':<12}"
            print(header)
            print("-" * len(header))
        elif stat_type == "mean":
            header = f"{'Matrix Size':<15} {'Gather %':<10} {'Method':<25} {'Mean (ms)':<12} {'BW (GB/s)':<12}"
            print(header)
            print("-" * len(header))
        elif stat_type == "std":
            header = f"{'Matrix Size':<15} {'Gather %':<10} {'Method':<25} {'Std (ms)':<12} {'BW (GB/s)':<12}"
            print(header)
            print("-" * len(header))

        # Print data
        for size, gather_ratios in sorted(self.gather_results.items()):
            for gather_ratio, methods in sorted(gather_ratios.items()):
                for method_id, data in methods.items():
                    if data is None:
                        continue

                    size_str = f"{size[0]}x{size[1]}"
                    gather_str = f"{gather_ratio:.1%}"
                    method_name = data["method"]
                    bandwidth = data["bandwidth_gb_s"]

                    if stat_type == "all":
                        min_time = data["min_time_ms"]
                        mean_time = data["mean_time_ms"]
                        std_time = data["std_time_ms"]
                        print(
                            f"{size_str:<15} {gather_str:<10} {method_name:<25} {min_time:<12.3f} "
                            f"{mean_time:<12.3f} {std_time:<12.3f} {bandwidth:<12.2f}"
                        )
                    elif stat_type == "min":
                        min_time = data["min_time_ms"]
                        print(
                            f"{size_str:<15} {gather_str:<10} {method_name:<25} {min_time:<12.3f} {bandwidth:<12.2f}"
                        )
                    elif stat_type == "mean":
                        mean_time = data["mean_time_ms"]
                        print(
                            f"{size_str:<15} {gather_str:<10} {method_name:<25} {mean_time:<12.3f} {bandwidth:<12.2f}"
                        )
                    elif stat_type == "std":
                        std_time = data["std_time_ms"]
                        print(
                            f"{size_str:<15} {gather_str:<10} {method_name:<25} {std_time:<12.3f} {bandwidth:<12.2f}"
                        )

    def print_rankings(
        self,
        top_n: int = 3,
        metric: str = "bandwidth",
        gather_ratio_filter: float = None,
    ):
        """Print top N performers for each matrix size and gather ratio

        Args:
            top_n: Number of top performers to show (default: 3)
            metric: "bandwidth", "min_time", or "mean_time"
            gather_ratio_filter: Filter to specific gather ratio (optional)
        """
        print("=" * 80)
        filter_str = (
            f" (Gather {gather_ratio_filter:.1%})" if gather_ratio_filter else ""
        )
        print(f"TOP {top_n} PERFORMERS BY {metric.upper()}{filter_str}")
        print("=" * 80)

        if not self.gather_results:
            print("No gather results available")
            return

        # Print header
        rank_headers = []
        for i in range(1, top_n + 1):
            if i == 1:
                rank_headers.append("1st")
            elif i == 2:
                rank_headers.append("2nd")
            elif i == 3:
                rank_headers.append("3rd")
            else:
                rank_headers.append(f"{i}th")

        header_parts = ["Matrix Size".ljust(15), "Gather %".ljust(10)]
        for h in rank_headers:
            header_parts.append(f"{h} Place".ljust(30))
        print(" | ".join(header_parts) + " |")
        print("-" * (15 + 3 + 10 + 3 + (30 + 3) * top_n))

        for size, gather_ratios in sorted(self.gather_results.items()):
            for gather_ratio, methods in sorted(gather_ratios.items()):
                # Apply gather ratio filter if specified
                if (
                    gather_ratio_filter is not None
                    and abs(gather_ratio - gather_ratio_filter) > 1e-6
                ):
                    continue

                # Get valid methods and sort by metric
                valid_methods = []
                for method_id, data in methods.items():
                    if data is None:
                        continue

                    if metric == "bandwidth":
                        sort_value = data["bandwidth_gb_s"]
                        reverse = True  # Higher is better
                    elif metric == "min_time":
                        sort_value = data["min_time_ms"]
                        reverse = False  # Lower is better
                    elif metric == "mean_time":
                        sort_value = data["mean_time_ms"]
                        reverse = False  # Lower is better
                    else:
                        continue

                    valid_methods.append((sort_value, data))

                valid_methods.sort(reverse=reverse, key=lambda x: x[0])

                # Print top N
                size_str = f"{size[0]}x{size[1]}"
                gather_str = f"{gather_ratio:.1%}"
                row_parts = [f"{size_str:<15}", f"{gather_str:<10}"]

                for i in range(top_n):
                    if i < len(valid_methods):
                        _, data = valid_methods[i]
                        method_name = data["method"]

                        if metric == "bandwidth":
                            value_str = f"{data['bandwidth_gb_s']:.2f} GB/s"
                        elif metric in ["min_time", "mean_time"]:
                            value_str = f"{data[f'{metric}_ms']:.3f} ms"

                        cell_content = f"{method_name} ({value_str})"
                        row_parts.append(f"{cell_content:<30}")
                    else:
                        row_parts.append(f"{'N/A':<30}")

                print(" | ".join(row_parts) + " |")

    def plot_performance_vs_n(
        self,
        fixed_channels: List[int] = None,
        fixed_gather_ratio: float = 0.5,
        metric: str = "bandwidth",
        save_path: str = None,
    ):
        """Plot performance vs N for fixed channel counts and gather ratio

        Args:
            fixed_channels: List of channel counts to plot (default: [64, 128, 256, 512])
            fixed_gather_ratio: Gather ratio to plot (default: 0.5)
            metric: "bandwidth", "min_time", or "mean_time"
            save_path: Path to save the plot (optional)
        """
        if fixed_channels is None:
            fixed_channels = [64, 128, 256, 512]

        if not self.gather_results:
            print("No gather results available")
            return

        # Prepare data
        plot_data = {}
        n_values = set()

        # Collect all methods and N values
        all_methods = set()
        for size, gather_ratios in self.gather_results.items():
            n, c = size
            if c in fixed_channels and fixed_gather_ratio in gather_ratios:
                n_values.add(n)
                methods = gather_ratios[fixed_gather_ratio]
                for method_id, data in methods.items():
                    if data is not None:
                        all_methods.add(data["method"])

        n_values = sorted(n_values)

        # Organize data by method and channel
        for method in all_methods:
            plot_data[method] = {}
            for c in fixed_channels:
                plot_data[method][c] = {n: None for n in n_values}

        # Fill in the data
        for size, gather_ratios in self.gather_results.items():
            n, c = size
            if c in fixed_channels and fixed_gather_ratio in gather_ratios:
                methods = gather_ratios[fixed_gather_ratio]
                for method_id, data in methods.items():
                    if data is not None:
                        method_name = data["method"]
                        if method_name in plot_data:
                            if metric == "bandwidth":
                                value = data["bandwidth_gb_s"]
                            elif metric == "min_time":
                                value = data["min_time_ms"]
                            elif metric == "mean_time":
                                value = data["mean_time_ms"]
                            else:
                                continue
                            plot_data[method_name][c][n] = value

        # Create subplots for each channel count
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Gather Performance vs Matrix Rows (N) - {metric.title()} (Gather {fixed_gather_ratio:.1%})",
            fontsize=16,
        )

        axes = axes.flatten()

        for idx, c in enumerate(fixed_channels[:4]):  # Limit to 4 subplots
            ax = axes[idx]

            # Plot each method
            for method_name, channel_data in plot_data.items():
                if c in channel_data:
                    n_vals = []
                    metric_vals = []

                    for n in n_values:
                        if channel_data[c][n] is not None:
                            n_vals.append(n)
                            metric_vals.append(channel_data[c][n])

                    if n_vals and metric_vals:
                        ax.plot(
                            n_vals,
                            metric_vals,
                            marker="o",
                            label=method_name,
                            linewidth=2,
                            markersize=4,
                        )

            ax.set_xlabel("Matrix Rows (N)")
            if metric == "bandwidth":
                ax.set_ylabel("Bandwidth (GB/s)")
            else:
                ax.set_ylabel(f"{metric.replace('_', ' ').title()} (ms)")

            ax.set_title(f"Channels = {c}")
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_performance_vs_gather_ratio(
        self,
        fixed_sizes: List[Tuple[int, int]] = None,
        metric: str = "bandwidth",
        save_path: str = None,
    ):
        """Plot performance vs gather ratio for fixed matrix sizes

        Args:
            fixed_sizes: List of matrix sizes to plot (default: [(1024, 64), (4096, 128), (16384, 256)])
            metric: "bandwidth", "min_time", or "mean_time"
            save_path: Path to save the plot (optional)
        """
        if fixed_sizes is None:
            fixed_sizes = [(1024, 64), (4096, 128), (16384, 256)]

        if not self.gather_results:
            print("No gather results available")
            return

        # Prepare data
        plot_data = {}
        gather_ratios = set()

        # Collect all methods and gather ratios
        all_methods = set()
        for size, gather_ratios_dict in self.gather_results.items():
            if size in fixed_sizes:
                gather_ratios.update(gather_ratios_dict.keys())
                for gather_ratio, methods in gather_ratios_dict.items():
                    for method_id, data in methods.items():
                        if data is not None:
                            all_methods.add(data["method"])

        gather_ratios = sorted(gather_ratios)

        # Organize data by method and size
        for method in all_methods:
            plot_data[method] = {}
            for size in fixed_sizes:
                plot_data[method][size] = {ratio: None for ratio in gather_ratios}

        # Fill in the data
        for size, gather_ratios_dict in self.gather_results.items():
            if size in fixed_sizes:
                for gather_ratio, methods in gather_ratios_dict.items():
                    for method_id, data in methods.items():
                        if data is not None:
                            method_name = data["method"]
                            if method_name in plot_data:
                                if metric == "bandwidth":
                                    value = data["bandwidth_gb_s"]
                                elif metric == "min_time":
                                    value = data["min_time_ms"]
                                elif metric == "mean_time":
                                    value = data["mean_time_ms"]
                                else:
                                    continue
                                plot_data[method_name][size][gather_ratio] = value

        # Create subplots for each matrix size
        n_sizes = len(fixed_sizes)
        n_cols = min(3, n_sizes)
        n_rows = (n_sizes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_sizes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        fig.suptitle(
            f"Gather Performance vs Gather Ratio - {metric.title()}", fontsize=16
        )

        for idx, size in enumerate(fixed_sizes):
            ax = axes[idx]

            # Plot each method
            for method_name, size_data in plot_data.items():
                if size in size_data:
                    ratio_vals = []
                    metric_vals = []

                    for ratio in gather_ratios:
                        if size_data[size][ratio] is not None:
                            ratio_vals.append(ratio)
                            metric_vals.append(size_data[size][ratio])

                    if ratio_vals and metric_vals:
                        ax.plot(
                            ratio_vals,
                            metric_vals,
                            marker="o",
                            label=method_name,
                            linewidth=2,
                            markersize=4,
                        )

            ax.set_xlabel("Gather Ratio")
            if metric == "bandwidth":
                ax.set_ylabel("Bandwidth (GB/s)")
            else:
                ax.set_ylabel(f"{metric.replace('_', ' ').title()} (ms)")

            ax.set_title(f"Matrix Size: {size[0]}x{size[1]}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_sizes, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_heatmap(
        self,
        metric: str = "bandwidth",
        gather_ratio_filter: float = 0.5,
        save_path: str = None,
        method_filter: str = None,
    ):
        """Create heatmap showing performance across all matrix sizes for a specific gather ratio

        Args:
            metric: "bandwidth", "min_time", or "mean_time"
            gather_ratio_filter: Gather ratio to filter by (default: 0.5)
            save_path: Path to save the plot (optional)
            method_filter: Filter to specific method name (substring match)
        """
        if not self.gather_results:
            print("No gather results available")
            return

        # Collect data for heatmap
        methods_to_plot = []
        if method_filter:
            # Filter methods by name
            all_methods = set()
            for size, gather_ratios in self.gather_results.items():
                if gather_ratio_filter in gather_ratios:
                    methods = gather_ratios[gather_ratio_filter]
                    for method_id, data in methods.items():
                        if data is not None:
                            method_name = data["method"]
                            if method_filter.lower() in method_name.lower():
                                all_methods.add(method_name)
            methods_to_plot = sorted(all_methods)
        else:
            # Get all methods for the specified gather ratio
            all_methods = set()
            for size, gather_ratios in self.gather_results.items():
                if gather_ratio_filter in gather_ratios:
                    methods = gather_ratios[gather_ratio_filter]
                    for method_id, data in methods.items():
                        if data is not None:
                            all_methods.add(data["method"])
            methods_to_plot = sorted(all_methods)

        # Get all sizes that have the specified gather ratio
        all_sizes = []
        for size, gather_ratios in self.gather_results.items():
            if gather_ratio_filter in gather_ratios:
                all_sizes.append(size)
        all_sizes = sorted(all_sizes)
        size_labels = [f"{n}x{c}" for n, c in all_sizes]

        # Create data matrix
        data_matrix = np.full((len(methods_to_plot), len(all_sizes)), np.nan)

        for size_idx, size in enumerate(all_sizes):
            gather_ratios = self.gather_results[size]
            if gather_ratio_filter in gather_ratios:
                methods = gather_ratios[gather_ratio_filter]
                for method_id, data in methods.items():
                    if data is not None:
                        method_name = data["method"]
                        if method_name in methods_to_plot:
                            method_idx = methods_to_plot.index(method_name)

                            if metric == "bandwidth":
                                value = data["bandwidth_gb_s"]
                            elif metric == "min_time":
                                value = data["min_time_ms"]
                            elif metric == "mean_time":
                                value = data["mean_time_ms"]
                            else:
                                continue

                            data_matrix[method_idx, size_idx] = value

        # Create heatmap
        plt.figure(figsize=(16, max(8, len(methods_to_plot) * 0.5)))

        # Create a mask for NaN values
        mask = np.isnan(data_matrix)

        sns.heatmap(
            data_matrix,
            xticklabels=size_labels,
            yticklabels=methods_to_plot,
            annot=True,
            fmt=".2f",
            cmap="viridis" if metric == "bandwidth" else "viridis_r",
            mask=mask,
            cbar_kws={
                "label": f"{metric.replace('_', ' ').title()} {'(GB/s)' if metric == 'bandwidth' else '(ms)'}"
            },
        )

        plt.title(
            f"Gather Performance Heatmap - {metric.title()} (Gather {gather_ratio_filter:.1%})"
        )
        plt.xlabel("Matrix Size (Rows x Cols)")
        plt.ylabel("Method")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to: {save_path}")
        else:
            plt.show()


def main():
    """Main analysis execution"""
    parser = argparse.ArgumentParser(
        description="Analyze CUDA gather benchmark results with comprehensive statistics and visualizations"
    )

    parser.add_argument("results_file", help="Path to the JSON results file")

    # Analysis options
    analysis_group = parser.add_argument_group("analysis options")
    analysis_group.add_argument(
        "--metadata", action="store_true", help="Print benchmark metadata"
    )
    analysis_group.add_argument(
        "--stats",
        choices=["min", "mean", "std", "all"],
        default="all",
        help="Print statistics (default: all)",
    )
    analysis_group.add_argument(
        "--rankings",
        type=int,
        default=3,
        metavar="N",
        help="Print top N rankings (default: 3)",
    )
    analysis_group.add_argument(
        "--gather-ratio",
        type=float,
        default=None,
        help="Filter rankings to specific gather ratio (e.g., 0.5 for 50%)",
    )

    # Visualization options
    viz_group = parser.add_argument_group("visualization options")
    viz_group.add_argument(
        "--plot-performance",
        action="store_true",
        help="Generate performance vs N plots",
    )
    viz_group.add_argument(
        "--plot-gather-ratio",
        action="store_true",
        help="Generate performance vs gather ratio plots",
    )
    viz_group.add_argument(
        "--plot-heatmap", action="store_true", help="Generate performance heatmap"
    )
    viz_group.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Channel counts for performance plots (default: 64 128 256 512)",
    )
    viz_group.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Matrix sizes for gather ratio plots (format: N1 C1 N2 C2 ...)",
    )
    viz_group.add_argument(
        "--metric",
        choices=["bandwidth", "min_time", "mean_time"],
        default="bandwidth",
        help="Metric to plot (default: bandwidth)",
    )
    viz_group.add_argument(
        "--method-filter", type=str, help="Filter methods by name (substring match)"
    )
    viz_group.add_argument(
        "--save-plots", type=str, help="Directory to save plots (default: display only)"
    )

    # Control options
    control_group = parser.add_argument_group("control options")
    control_group.add_argument(
        "--no-stats", action="store_true", help="Don't print statistics"
    )
    control_group.add_argument(
        "--no-rankings", action="store_true", help="Don't print rankings"
    )

    args = parser.parse_args()

    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found!")
        return

    # Initialize analyzer
    analyzer = GatherBenchmarkAnalyzer(args.results_file)

    # Print metadata if requested
    if args.metadata:
        analyzer.print_metadata()

    # Print statistics unless disabled
    if not args.no_stats:
        analyzer.print_statistics(args.stats)

    # Print rankings unless disabled
    if not args.no_rankings:
        analyzer.print_rankings(args.rankings, args.metric, args.gather_ratio)

    # Generate plots
    if args.plot_performance or args.plot_gather_ratio or args.plot_heatmap:
        if args.save_plots:
            Path(args.save_plots).mkdir(parents=True, exist_ok=True)

    if args.plot_performance:
        save_path = None
        if args.save_plots:
            save_path = f"{args.save_plots}/gather_performance_vs_n_{args.metric}.png"

        analyzer.plot_performance_vs_n(
            fixed_channels=args.channels,
            fixed_gather_ratio=0.5,  # Default to 50% gather ratio
            metric=args.metric,
            save_path=save_path,
        )

    if args.plot_gather_ratio:
        # Parse sizes argument
        fixed_sizes = None
        if args.sizes:
            if len(args.sizes) % 2 != 0:
                print(
                    "Error: --sizes must have even number of values (N1 C1 N2 C2 ...)"
                )
                return
            fixed_sizes = [
                (args.sizes[i], args.sizes[i + 1]) for i in range(0, len(args.sizes), 2)
            ]

        save_path = None
        if args.save_plots:
            save_path = (
                f"{args.save_plots}/gather_performance_vs_ratio_{args.metric}.png"
            )

        analyzer.plot_performance_vs_gather_ratio(
            fixed_sizes=fixed_sizes, metric=args.metric, save_path=save_path
        )

    if args.plot_heatmap:
        save_path = None
        if args.save_plots:
            method_suffix = f"_{args.method_filter}" if args.method_filter else ""
            save_path = (
                f"{args.save_plots}/gather_heatmap_{args.metric}{method_suffix}.png"
            )

        analyzer.plot_heatmap(
            metric=args.metric,
            gather_ratio_filter=0.5,  # Default to 50% gather ratio
            save_path=save_path,
            method_filter=args.method_filter,
        )


if __name__ == "__main__":
    main()
