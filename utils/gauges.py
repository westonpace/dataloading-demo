import threading
import time
from typing import Optional
from datetime import datetime
import json


class Gauge:
    """
    A gauge that tracks events with duration and size to compute throughput.
    All measurements are in bytes per second.

    Attributes:
        name: Name of the gauge
    """

    def __init__(self, name: str):
        self.name = name
        self._total_size = 0.0
        self._total_duration = 0.0
        self._last_reset_time = time.time()

    def record_event(self, duration: float, size: float):
        """
        Record an event with its duration and size.

        Args:
            duration: Duration of the event in seconds
            size: Size in bytes processed during the event
        """
        self._total_size += size
        self._total_duration += duration

    def get_speed(self) -> float:
        """
        Get the current speed in bytes per second.

        Returns:
            Speed calculated as total_size / total_duration (bytes/s)
        """
        if self._total_duration == 0:
            return 0.0
        return self._total_size / self._total_duration

    def reset(self):
        """Reset the gauge statistics."""
        self._total_size = 0.0
        self._total_duration = 0.0
        self._last_reset_time = time.time()

    def get_stats(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary with gauge statistics (speed in bytes/s)
        """
        return {
            "name": self.name,
            "speed": self.get_speed(),
            "total_size": self._total_size,
            "total_duration": self._total_duration,
        }


class GaugeCollection:
    """
    A collection of gauges with automatic periodic stats logging.

    The collection runs a background thread that periodically writes
    gauge statistics to a file in /tmp.

    All gauges must be created before entering the context manager.
    """

    def __init__(
        self,
        name: str = "gauges",
        interval: float = 1.0,
        stats_file: Optional[str] = None,
    ):
        """
        Initialize a gauge collection.

        Args:
            name: Name of the collection
            interval: Interval in seconds between stats writes
            stats_file: Path to the stats file (default: /tmp/stats.jsonl)
        """
        self.name = name
        self.interval = interval
        self._stats_file_path = stats_file or "/tmp/stats.jsonl"
        self._gauges = {}
        self._thread = None
        self._stop_event = threading.Event()
        self._stats_file = None
        self._start_time = None
        self._started = False

    def create_gauge(self, name: str) -> Gauge:
        """
        Create a new gauge in the collection.

        Must be called before entering the context manager.

        Args:
            name: Name of the gauge

        Returns:
            The created Gauge instance
        """
        if self._started:
            raise RuntimeError("Cannot create gauges after collection has been started")
        if name in self._gauges:
            raise ValueError(f"Gauge '{name}' already exists")
        gauge = Gauge(name)
        self._gauges[name] = gauge
        return gauge

    def get_gauge(self, name: str) -> Optional[Gauge]:
        """
        Get a gauge by name.

        Args:
            name: Name of the gauge

        Returns:
            The Gauge instance or None if not found
        """
        return self._gauges.get(name)

    def _write_stats(self):
        """Write current stats to the stats file."""
        timestamp = time.time()
        elapsed = timestamp - self._start_time

        stats = {
            "timestamp": timestamp,
            "elapsed": elapsed,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "gauges": {},
        }

        for name, gauge in self._gauges.items():
            stats["gauges"][name] = gauge.get_stats()

        # Append to stats file
        with open(self._stats_file, "a") as f:
            f.write(json.dumps(stats) + "\n")

    def _stats_thread_func(self):
        """Background thread function that periodically writes stats."""
        while not self._stop_event.wait(self.interval):
            self._write_stats()

    def __enter__(self):
        """Enter context manager - start background thread and create stats file."""
        self._started = True
        self._start_time = time.time()
        self._stats_file = self._stats_file_path

        # Create initial stats file with metadata
        with open(self._stats_file, "w") as f:
            metadata = {
                "collection_name": self.name,
                "start_time": self._start_time,
                "start_datetime": datetime.fromtimestamp(self._start_time).isoformat(),
                "interval": self.interval,
            }
            f.write(json.dumps({"metadata": metadata}) + "\n")

        # Start background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stats_thread_func, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - stop background thread and write final stats."""
        # Signal thread to stop
        self._stop_event.set()

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)

        # Write final stats
        self._write_stats()

        return False
