"""
Pretty output utilities for console display.

Provides formatted, colored output functions for better CLI experience.
"""

import sys
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str, char: str = "=", width: int = 80):
    """
    Print a large header with borders.

    Args:
        text: Header text
        char: Character to use for border
        width: Width of the header

    Example:
        ================================================================================
                                    MY HEADER TEXT
        ================================================================================
    """
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^{width}}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{char * width}{Colors.END}")
    print()


def print_section(text: str, char: str = "-", width: int = 80):
    """
    Print a medium section header.

    Args:
        text: Section text
        char: Character to use for underline
        width: Width of the section

    Example:
        My Section
        --------------------------------------------------------------------------------
    """
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.CYAN}{char * width}{Colors.END}")


def print_subsection(text: str):
    """
    Print a small subsection header.

    Args:
        text: Subsection text

    Example:
        → My Subsection
    """
    print(f"\n{Colors.BOLD}→ {text}{Colors.END}")


def print_success(text: str, prefix: str = "✓"):
    """
    Print a success message in green.

    Args:
        text: Success message
        prefix: Prefix symbol (default: checkmark)

    Example:
        ✓ Operation completed successfully
    """
    print(f"{Colors.GREEN}{prefix} {text}{Colors.END}")


def print_error(text: str, prefix: str = "✗"):
    """
    Print an error message in red.

    Args:
        text: Error message
        prefix: Prefix symbol (default: X)

    Example:
        ✗ Operation failed
    """
    print(f"{Colors.RED}{prefix} {text}{Colors.END}", file=sys.stderr)


def print_warning(text: str, prefix: str = "⚠"):
    """
    Print a warning message in yellow.

    Args:
        text: Warning message
        prefix: Prefix symbol (default: warning sign)

    Example:
        ⚠ Warning: This might take a while
    """
    print(f"{Colors.YELLOW}{prefix} {text}{Colors.END}")


def print_info(text: str, prefix: str = "ℹ"):
    """
    Print an info message in cyan.

    Args:
        text: Info message
        prefix: Prefix symbol (default: info)

    Example:
        ℹ Processing 3 benchmarks...
    """
    print(f"{Colors.CYAN}{prefix} {text}{Colors.END}")


def print_step(step_num: int, total_steps: int, text: str):
    """
    Print a step in a multi-step process.

    Args:
        step_num: Current step number
        total_steps: Total number of steps
        text: Step description

    Example:
        [1/5] Loading configuration...
    """
    print(f"{Colors.BOLD}[{step_num}/{total_steps}]{Colors.END} {text}")


def print_metric(name: str, value: any, unit: str = "", width: int = 40):
    """
    Print a metric in a formatted way.

    Args:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
        width: Width for alignment

    Example:
        Execution Time ...................... 45.2 ms
    """
    dots = "." * (width - len(name))
    value_str = f"{value} {unit}".strip()
    print(f"  {name} {Colors.CYAN}{dots}{Colors.END} {Colors.BOLD}{value_str}{Colors.END}")


def print_table(headers: list, rows: list, alignments: Optional[list] = None):
    """
    Print a simple formatted table.

    Args:
        headers: List of header strings
        rows: List of row lists
        alignments: List of alignment characters ('l', 'c', 'r')

    Example:
        ┌──────────┬───────┬────────┐
        │ Name     │ Time  │ Tokens │
        ├──────────┼───────┼────────┤
        │ Baseline │ 85.4  │ 850    │
        │ Bootstrap│ 52.1  │ 3760   │
        └──────────┴───────┴────────┘
    """
    if alignments is None:
        alignments = ['l'] * len(headers)

    # Calculate column widths
    col_widths = []
    for i in range(len(headers)):
        max_width = len(str(headers[i]))
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)

    # Print top border
    print("┌" + "┬".join("─" * w for w in col_widths) + "┐")

    # Print headers
    header_strs = []
    for i, (h, w, a) in enumerate(zip(headers, col_widths, alignments)):
        if a == 'c':
            header_strs.append(f"{str(h):^{w}}")
        elif a == 'r':
            header_strs.append(f"{str(h):>{w}}")
        else:
            header_strs.append(f"{str(h):<{w}}")
    print(f"{Colors.BOLD}│{'│'.join(header_strs)}│{Colors.END}")

    # Print separator
    print("├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Print rows
    for row in rows:
        row_strs = []
        for i, (cell, w, a) in enumerate(zip(row, col_widths, alignments)):
            cell_str = str(cell) if i < len(row) else ""
            if a == 'c':
                row_strs.append(f"{cell_str:^{w}}")
            elif a == 'r':
                row_strs.append(f"{cell_str:>{w}}")
            else:
                row_strs.append(f"{cell_str:<{w}}")
        print("│" + "│".join(row_strs) + "│")

    # Print bottom border
    print("└" + "┴".join("─" * w for w in col_widths) + "┘")


def print_progress_bar(current: int, total: int, prefix: str = "", width: int = 50):
    """
    Print a progress bar.

    Args:
        current: Current progress
        total: Total items
        prefix: Prefix text
        width: Width of the progress bar

    Example:
        Processing: [████████████████████          ] 60%
    """
    percent = int(100 * current / total)
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r{prefix} [{Colors.GREEN}{bar}{Colors.END}] {percent}%", end="", flush=True)
    if current == total:
        print()  # New line when complete


def print_key_value(key: str, value: any, indent: int = 0):
    """
    Print a key-value pair.

    Args:
        key: Key name
        value: Value
        indent: Indentation level

    Example:
          Model: gpt-4.1-mini
          Tokens: 10,190
    """
    indent_str = "  " * indent
    print(f"{indent_str}{Colors.BOLD}{key}:{Colors.END} {value}")


def print_divider(char: str = "-", width: int = 80, color: str = Colors.CYAN):
    """
    Print a simple divider line.

    Args:
        char: Character to use
        width: Width of the divider
        color: ANSI color code
    """
    print(f"{color}{char * width}{Colors.END}")


if __name__ == "__main__":
    # Demo all the pretty print functions
    print_header("PRETTY OUTPUT DEMO")

    print_section("Success, Error, Warning, Info Messages")
    print_success("Operation completed successfully!")
    print_error("Something went wrong!")
    print_warning("This is a warning message")
    print_info("This is an informational message")

    print_section("Step-by-Step Progress")
    print_step(1, 5, "Loading configuration...")
    print_step(2, 5, "Initializing models...")
    print_step(3, 5, "Running benchmarks...")

    print_section("Metrics Display")
    print_metric("Execution Time", "45.2", "ms")
    print_metric("Total Tokens", "10,190")
    print_metric("Cost", "$0.0103", "USD")

    print_section("Table Display")
    headers = ["Optimizer", "Time (ms)", "Tokens", "Cost"]
    rows = [
        ["Baseline", "85.4", "850", "$0.0009"],
        ["Bootstrap", "52.1", "3,760", "$0.0039"],
        ["MIPROv2", "45.2", "10,190", "$0.0103"],
        ["GEPA", "43.8", "16,990", "$0.0163"]
    ]
    print_table(headers, rows, alignments=['l', 'r', 'r', 'r'])

    print_section("Progress Bar")
    import time
    for i in range(1, 101):
        print_progress_bar(i, 100, "Processing")
        time.sleep(0.02)

    print_section("Key-Value Pairs")
    print_key_value("Model", "gpt-4.1-mini")
    print_key_value("Benchmark", "matrix_multiply")
    print_key_value("Size", "512x512", indent=1)
    print_key_value("Iterations", "10", indent=1)

    print_divider()
    print_success("Demo completed!")
