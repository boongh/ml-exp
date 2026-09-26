#!/usr/bin/env python3
"""
Training Report Generator

Reads a training log folder (hyper_param.txt and training_log.csv),
generates a loss plot, and produces a LaTeX report with parameters and graph.

Usage:
    python training_report.py <log_folder> [--output OUTPUT_DIR]

Example:
    python training_report.py data/training_log/2026-09-25_15-22-25-349801
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt


def parse_hyperparameters(hyper_param_path: Path) -> Dict[str, str]:
    """Parse hyper_param.txt as key=value lines."""
    params = {}
    try:
        with open(hyper_param_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                params[key.strip()] = value.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Hyperparameters file not found: {hyper_param_path}")
    return params


def parse_training_log(csv_path: Path) -> Tuple[list, list, list]:
    """Parse training_log.csv and return (steps, train_losses, val_losses)."""
    steps = []
    train_losses = []
    val_losses = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty")

            required_fields = {"step", "train_loss", "val_loss"}
            if not required_fields.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV missing required columns. Expected {required_fields}, "
                    f"got {set(reader.fieldnames)}"
                )

            for row in reader:
                try:
                    steps.append(int(row["step"]))
                    train_losses.append(float(row["train_loss"]))
                    val_losses.append(float(row["val_loss"]))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Malformed CSV row: {row}") from e

    except FileNotFoundError:
        raise FileNotFoundError(f"Training log CSV not found: {csv_path}")

    if not steps:
        raise ValueError("No loss data in training_log.csv")

    return steps, train_losses, val_losses


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def generate_plot(
    steps: list, train_losses: list, val_losses: list, output_path: Path
) -> None:
    """Generate a loss plot and save as PNG."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", marker="o", markersize=4)
    plt.plot(steps, val_losses, label="Validation Loss", marker="s", markersize=4)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, format="png")
    plt.close()
    print(f"Plot saved: {output_path}")


def generate_latex_report(
    hyperparams: Dict[str, str], plot_path: Path, output_path: Path
) -> None:
    """Generate a LaTeX report with parameters and graph."""
    # Escape parameter values for LaTeX
    escaped_params = {k: escape_latex(v) for k, v in hyperparams.items()}

    # Compute relative path from report to plot for portability
    try:
        plot_relative = plot_path.relative_to(output_path.parent)
    except ValueError:
        plot_relative = plot_path

    # Build parameter table rows
    param_rows = "\n    ".join(
        f"{escape_latex(k)} & {v} \\\\" for k, v in sorted(escaped_params.items())
    )

    latex_content = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{float}}

\\title{{Training Report}}
\\author{{}}
\\date{{}}

\\begin{{document}}

\\maketitle

\\section{{Hyperparameters}}

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Value}} \\\\
\\midrule
{param_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Training Progress}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{plot_relative}}}
\\caption{{Training and validation loss over training steps.}}
\\end{{figure}}

\\end{{document}}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"LaTeX report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX report from a training log folder."
    )
    parser.add_argument(
        "log_folder",
        type=Path,
        help="Path to the training log folder containing hyper_param.txt and training_log.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as log_folder)",
    )
    parser.add_argument(
        "--plot-name",
        default="loss_plot.png",
        help="Name of the generated plot file (default: loss_plot.png)",
    )
    parser.add_argument(
        "--report-name",
        default="report.tex",
        help="Name of the generated LaTeX report (default: report.tex)",
    )

    args = parser.parse_args()

    # Resolve paths
    log_folder = args.log_folder.resolve()
    if not log_folder.is_dir():
        print(f"Error: Log folder not found: {log_folder}", file=sys.stderr)
        sys.exit(1)

    output_dir = (args.output or log_folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hyper_param_path = log_folder / "hyper_param.txt"
    training_log_path = log_folder / "training_log.csv"

    # Parse inputs
    try:
        hyperparams = parse_hyperparameters(hyper_param_path)
        steps, train_losses, val_losses = parse_training_log(training_log_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate outputs
    plot_path = output_dir / args.plot_name
    report_path = output_dir / args.report_name

    try:
        generate_plot(steps, train_losses, val_losses, plot_path)
        generate_latex_report(hyperparams, plot_path, report_path)
        print(f"\nReport generation complete!")
        print(f"  Log folder: {log_folder}")
        print(f"  Output directory: {output_dir}")
        print(f"  Report: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
