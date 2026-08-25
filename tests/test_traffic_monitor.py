"""Unit tests for traffic-monitoring calculations."""

from __future__ import annotations

import unittest

import numpy as np

from src.traffic_monitor import LineCrossingCounter, calculate_occupancy, density_level


class TrafficMonitorTests(unittest.TestCase):
    def test_line_crossings_are_counted_by_direction_and_class(self) -> None:
        counter = LineCrossingCounter(line_y=50)
        self.assertIsNone(counter.update(7, "car", 40))
        self.assertEqual(counter.update(7, "car", 60), "down")
        self.assertEqual(counter.update(7, "car", 45), "up")
        self.assertEqual(counter.counts["down"]["car"], 1)
        self.assertEqual(counter.counts["up"]["car"], 1)

    def test_overlapping_boxes_are_not_double_counted(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        occupancy = calculate_occupancy([(0, 0, 50, 50), (25, 25, 75, 75)], frame.shape)
        self.assertAlmostEqual(occupancy, 0.4375)

    def test_density_levels(self) -> None:
        self.assertEqual(density_level(0.02, 0.08, 0.20), "LOW")
        self.assertEqual(density_level(0.12, 0.08, 0.20), "MODERATE")
        self.assertEqual(density_level(0.25, 0.08, 0.20), "HIGH")


if __name__ == "__main__":
    unittest.main()
