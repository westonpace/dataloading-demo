#!/usr/bin/env python3
"""
CLI tool to watch and display gauge statistics in real-time.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def format_bytes_per_second(bytes_per_sec: float) -> Tuple[float, str]:
    """
    Convert bytes per second to appropriate units.

    Args:
        bytes_per_sec: Speed in bytes per second

    Returns:
        Tuple of (converted value, unit string)
    """
    # Define units (using powers of 1024 for binary units)
    units = [
        (1024**4, "TiB/s"),
        (1024**3, "GiB/s"),
        (1024**2, "MiB/s"),
        (1024**1, "KiB/s"),
        (1, "B/s"),
    ]

    for threshold, unit in units:
        if bytes_per_sec >= threshold:
            return bytes_per_sec / threshold, unit

    return bytes_per_sec, "B/s"


def select_common_unit(max_speed: float) -> Tuple[float, str]:
    """
    Select an appropriate common unit for displaying speeds.

    Args:
        max_speed: Maximum speed in bytes per second

    Returns:
        Tuple of (divisor, unit string)
    """
    # Choose unit based on max speed
    if max_speed >= 1024**4:
        return 1024**4, "TiB/s"
    elif max_speed >= 1024**3:
        return 1024**3, "GiB/s"
    elif max_speed >= 1024**2:
        return 1024**2, "MiB/s"
    elif max_speed >= 1024:
        return 1024, "KiB/s"
    else:
        return 1, "B/s"


class GaugeWatcher:
    """Watches a stats file and displays gauge plots in the terminal."""

    def __init__(self, stats_file: str, refresh_interval: float = 0.5, max_points: int = 50, sync_y_axis: bool = True):
        """
        Initialize the gauge watcher.

        Args:
            stats_file: Path to the stats file to watch
            refresh_interval: How often to refresh the display (seconds)
            max_points: Maximum number of data points to keep for each gauge
            sync_y_axis: Whether to synchronize Y-axis across all gauges
        """
        self.stats_file = Path(stats_file)
        self.refresh_interval = refresh_interval
        self.max_points = max_points
        self.sync_y_axis = sync_y_axis
        self.gauge_data = defaultdict(lambda: {"times": [], "speeds": []})
        self.metadata = None
        self.last_position = 0

    def read_stats(self):
        """Read new lines from the stats file."""
        if not self.stats_file.exists():
            return

        with open(self.stats_file, "r") as f:
            f.seek(self.last_position)
            lines = f.readlines()
            self.last_position = f.tell()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # First line is metadata
                if "metadata" in data:
                    self.metadata = data["metadata"]
                    continue

                # Regular stats line
                if "gauges" in data and "elapsed" in data:
                    elapsed = data["elapsed"]
                    for gauge_name, gauge_stats in data["gauges"].items():
                        speed = gauge_stats["speed"]  # in bytes/s

                        # Store data
                        gauge_info = self.gauge_data[gauge_name]
                        gauge_info["times"].append(elapsed)
                        gauge_info["speeds"].append(speed)

                        # Keep only max_points
                        if len(gauge_info["times"]) > self.max_points:
                            gauge_info["times"].pop(0)
                            gauge_info["speeds"].pop(0)

            except json.JSONDecodeError:
                continue

    def render_plot(
        self, times: List[float], speeds: List[float], height: int = 10, width: int = 60
    ) -> Tuple[List[str], float]:
        """
        Render an ASCII plot of the speed over time.

        Args:
            times: List of time values
            speeds: List of speed values
            height: Height of the plot in lines
            width: Width of the plot in characters

        Returns:
            Tuple of (list of plot lines, max_speed)
        """
        if not times or not speeds:
            # Empty plot
            lines = []
            for _ in range(height):
                lines.append(" " * width)
            return lines, 0.0

        min_speed = 0.0
        max_speed = max(speeds) if speeds else 1.0
        if max_speed == 0:
            max_speed = 1.0

        min_time = min(times)
        max_time = max(times)
        time_range = max_time - min_time if max_time > min_time else 1.0

        # Create empty grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for t, s in zip(times, speeds):
            # Map time to x coordinate
            x = int((t - min_time) / time_range * (width - 1))
            # Map speed to y coordinate (inverted because row 0 is at top)
            y = height - 1 - int((s - min_speed) / (max_speed - min_speed) * (height - 1))

            # Clamp to grid boundaries
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            grid[y][x] = "█"

        # Fill in line segments to connect points
        if len(times) > 1:
            for i in range(len(times) - 1):
                t1, s1 = times[i], speeds[i]
                t2, s2 = times[i + 1], speeds[i + 1]

                x1 = int((t1 - min_time) / time_range * (width - 1))
                y1 = height - 1 - int((s1 - min_speed) / (max_speed - min_speed) * (height - 1))
                x2 = int((t2 - min_time) / time_range * (width - 1))
                y2 = height - 1 - int((s2 - min_speed) / (max_speed - min_speed) * (height - 1))

                # Clamp coordinates
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                # Simple line drawing using Bresenham-like algorithm
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy

                x, y = x1, y1
                while True:
                    grid[y][x] = "█"
                    if x == x2 and y == y2:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy

        # Convert grid to lines
        lines = ["".join(row) for row in grid]
        return lines, max_speed

    def render_display(self):
        """Render the complete display with all gauges."""
        output = []

        # Clear screen
        output.append("\033[2J\033[H")

        # Header
        if self.metadata:
            output.append(f"Gauge Collection: {self.metadata['collection_name']}")
            output.append(f"Interval: {self.metadata['interval']}s")
            output.append("")

        if not self.gauge_data:
            output.append("Waiting for data...")
            return "\n".join(output)

        # Get time range for synchronized x-axis
        all_times = []
        for gauge_info in self.gauge_data.values():
            all_times.extend(gauge_info["times"])

        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
        else:
            min_time = max_time = 0

        # Get global max speed for synchronized y-axis (in bytes/s)
        global_max_speed = 0.0
        if self.sync_y_axis:
            for gauge_info in self.gauge_data.values():
                if gauge_info["speeds"]:
                    global_max_speed = max(global_max_speed, max(gauge_info["speeds"]))
            if global_max_speed == 0:
                global_max_speed = 1.0

        # Select appropriate unit based on max speed
        if self.sync_y_axis:
            divisor, units = select_common_unit(global_max_speed)
        else:
            divisor, units = 1, "B/s"  # Will be overridden per gauge

        # Render each gauge
        for gauge_name, gauge_info in sorted(self.gauge_data.items()):
            times = gauge_info["times"]
            speeds = gauge_info["speeds"]  # in bytes/s

            # Determine the max speed for this gauge's plot
            if self.sync_y_axis:
                max_speed_for_plot = global_max_speed
                gauge_divisor = divisor
                gauge_units = units
            else:
                max_speed_for_plot = max(speeds) if speeds else 1.0
                if max_speed_for_plot == 0:
                    max_speed_for_plot = 1.0
                gauge_divisor, gauge_units = select_common_unit(max_speed_for_plot)

            # Convert speeds to display units
            display_speeds = [s / gauge_divisor for s in speeds]
            max_display_speed = max_speed_for_plot / gauge_divisor

            # Get latest value
            latest_value = display_speeds[-1] if display_speeds else 0.0

            # Render plot with display speeds
            plot_lines, _ = self.render_plot_with_max(times, display_speeds, max_display_speed)

            # Add gauge header
            output.append(f"=== {gauge_name} ===")

            # Add y-axis labels and plot with latest value on the right
            for i, line in enumerate(plot_lines):
                # Y-axis label (only at top and bottom)
                if i == 0:
                    label = f"{max_display_speed:7.2f} {gauge_units} │"
                elif i == len(plot_lines) - 1:
                    label = f"{0.0:7.2f} {gauge_units} │"
                else:
                    label = " " * (7 + len(gauge_units) + 2) + "│"

                # Add latest value on the middle line
                if i == len(plot_lines) // 2:
                    latest_label = f" Latest: {latest_value:.2f} {gauge_units}"
                    output.append(f"{label}{line}{latest_label}")
                else:
                    output.append(f"{label}{line}")

            # Add x-axis
            x_axis_label = " " * (7 + len(gauge_units) + 2) + "└" + "─" * 60
            output.append(x_axis_label)
            time_label = " " * (7 + len(gauge_units) + 3) + f"{min_time:.1f}s" + " " * 40 + f"{max_time:.1f}s"
            output.append(time_label)
            output.append("")

        return "\n".join(output)

    def render_plot_with_max(
        self, times: List[float], speeds: List[float], max_speed: float, height: int = 10, width: int = 60
    ) -> Tuple[List[str], float]:
        """
        Render an ASCII plot with a specified max speed.

        Args:
            times: List of time values
            speeds: List of speed values
            max_speed: Maximum speed for the y-axis
            height: Height of the plot in lines
            width: Width of the plot in characters

        Returns:
            Tuple of (list of plot lines, max_speed used)
        """
        if not times or not speeds:
            # Empty plot
            lines = []
            for _ in range(height):
                lines.append(" " * width)
            return lines, max_speed

        min_speed = 0.0

        min_time = min(times)
        max_time = max(times)
        time_range = max_time - min_time if max_time > min_time else 1.0

        # Create empty grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for t, s in zip(times, speeds):
            # Map time to x coordinate
            x = int((t - min_time) / time_range * (width - 1))
            # Map speed to y coordinate (inverted because row 0 is at top)
            y = height - 1 - int((s - min_speed) / (max_speed - min_speed) * (height - 1))

            # Clamp to grid boundaries
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            grid[y][x] = "█"

        # Fill in line segments to connect points
        if len(times) > 1:
            for i in range(len(times) - 1):
                t1, s1 = times[i], speeds[i]
                t2, s2 = times[i + 1], speeds[i + 1]

                x1 = int((t1 - min_time) / time_range * (width - 1))
                y1 = height - 1 - int((s1 - min_speed) / (max_speed - min_speed) * (height - 1))
                x2 = int((t2 - min_time) / time_range * (width - 1))
                y2 = height - 1 - int((s2 - min_speed) / (max_speed - min_speed) * (height - 1))

                # Clamp coordinates
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                # Simple line drawing using Bresenham-like algorithm
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy

                x, y = x1, y1
                while True:
                    grid[y][x] = "█"
                    if x == x2 and y == y2:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy

        # Convert grid to lines
        lines = ["".join(row) for row in grid]
        return lines, max_speed

    def watch(self):
        """Main watch loop."""
        try:
            while True:
                self.read_stats()
                display = self.render_display()
                print(display, end="", flush=True)
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\nStopped watching.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Watch and display gauge statistics in real-time"
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="/tmp/stats.jsonl",
        help="Path to the stats file (default: /tmp/stats.jsonl)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.5,
        help="Refresh interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50,
        help="Maximum number of data points to display (default: 50)",
    )
    parser.add_argument(
        "--no-sync-y-axis",
        action="store_true",
        help="Disable Y-axis synchronization across gauges (default: synchronized)",
    )
    args = parser.parse_args()

    watcher = GaugeWatcher(
        stats_file=args.stats_file,
        refresh_interval=args.refresh_interval,
        max_points=args.max_points,
        sync_y_axis=not args.no_sync_y_axis,
    )
    watcher.watch()


if __name__ == "__main__":
    main()
