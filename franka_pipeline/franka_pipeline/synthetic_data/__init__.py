"""Synthetic data generation module for simulated sensor data."""

from .rgb_to_events import RGBToEventsConverter, EventArray, events_to_frame
from .synthetic_recorder import SyntheticDataRecorder

__all__ = [
    "RGBToEventsConverter",
    "EventArray", 
    "events_to_frame",
    "SyntheticDataRecorder",
]
