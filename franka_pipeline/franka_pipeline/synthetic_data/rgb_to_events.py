"""
RGB to Events converter using v2e-style brightness change detection.

This module converts consecutive RGB frames into synthetic DVS events by
detecting logarithmic brightness changes that exceed configurable thresholds.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EventArray:
    """Container for DVS events."""
    x: np.ndarray       # x coordinates (uint16)
    y: np.ndarray       # y coordinates (uint16)
    t: np.ndarray       # timestamps in microseconds (int64)
    p: np.ndarray       # polarity: 1 for ON, 0 for OFF (uint8)
    
    def __len__(self) -> int:
        return len(self.t)
    
    def to_structured_array(self) -> np.ndarray:
        """Convert to structured numpy array compatible with Metavision format."""
        dtype = np.dtype([
            ('x', '<u2'),
            ('y', '<u2'),
            ('p', '<i2'),
            ('t', '<i8')
        ])
        events = np.zeros(len(self), dtype=dtype)
        events['x'] = self.x.astype(np.uint16)
        events['y'] = self.y.astype(np.uint16)
        events['p'] = self.p.astype(np.int16)
        events['t'] = self.t.astype(np.int64)
        return events


class RGBToEventsConverter:
    """
    Converts RGB frames to synthetic DVS events using log intensity changes.
    
    Based on the v2e algorithm: events are generated when the log intensity
    change exceeds a threshold (positive for ON events, negative for OFF events).
    
    Args:
        threshold_pos: Positive contrast threshold (ON events), typically 0.2-0.5
        threshold_neg: Negative contrast threshold (OFF events), typically 0.2-0.5
        refractory_period_us: Minimum time between events at same pixel (microseconds)
        leak_rate: Rate at which the reference decays toward current intensity (0-1)
        shot_noise_rate: Rate of random noise events per pixel per second
    """
    
    def __init__(
        self,
        threshold_pos: float = 0.25,
        threshold_neg: float = 0.25,
        refractory_period_us: int = 1000,
        leak_rate: float = 0.0,
        shot_noise_rate: float = 0.0,
    ):
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.refractory_period_us = refractory_period_us
        self.leak_rate = leak_rate
        self.shot_noise_rate = shot_noise_rate
        
        # State
        self._reference_log_intensity: Optional[np.ndarray] = None
        self._last_event_time: Optional[np.ndarray] = None
        self._height: int = 0
        self._width: int = 0
        self._last_timestamp_us: int = 0
    
    def reset(self) -> None:
        """Reset converter state for a new sequence."""
        self._reference_log_intensity = None
        self._last_event_time = None
        self._last_timestamp_us = 0
    
    def _rgb_to_log_intensity(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame to log intensity."""
        # Convert to grayscale using standard weights
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = frame
        
        # Convert to float and add small epsilon to avoid log(0)
        gray = gray.astype(np.float32)
        gray = np.clip(gray, 0.1, 255.0)  # Clip to avoid log(0)
        
        return np.log(gray)
    
    def convert_frame(
        self,
        frame: np.ndarray,
        timestamp_us: int,
    ) -> EventArray:
        """
        Convert a single RGB frame to events by comparing with reference.
        
        Args:
            frame: RGB or grayscale frame (H, W, 3) or (H, W)
            timestamp_us: Frame timestamp in microseconds
            
        Returns:
            EventArray containing generated events
        """
        log_intensity = self._rgb_to_log_intensity(frame)
        height, width = log_intensity.shape
        
        # Initialize state on first frame
        if self._reference_log_intensity is None:
            self._reference_log_intensity = log_intensity.copy()
            self._last_event_time = np.full((height, width), -self.refractory_period_us - 1, dtype=np.int64)
            self._height = height
            self._width = width
            self._last_timestamp_us = timestamp_us
            return EventArray(
                x=np.array([], dtype=np.uint16),
                y=np.array([], dtype=np.uint16),
                t=np.array([], dtype=np.int64),
                p=np.array([], dtype=np.uint8),
            )
        
        # Apply leak (optional reference decay toward current value)
        if self.leak_rate > 0:
            self._reference_log_intensity = (
                (1 - self.leak_rate) * self._reference_log_intensity + 
                self.leak_rate * log_intensity
            )
        
        # Compute log intensity difference
        diff = log_intensity - self._reference_log_intensity
        
        # Time since last event at each pixel
        time_since_event = timestamp_us - self._last_event_time
        refractory_ok = time_since_event >= self.refractory_period_us
        
        # Find ON events (positive threshold crossing)
        on_mask = (diff >= self.threshold_pos) & refractory_ok
        on_y, on_x = np.where(on_mask)
        
        # Find OFF events (negative threshold crossing)
        off_mask = (diff <= -self.threshold_neg) & refractory_ok
        off_y, off_x = np.where(off_mask)
        
        # Combine events
        num_on = len(on_x)
        num_off = len(off_x)
        num_events = num_on + num_off
        
        if num_events > 0:
            # Concatenate coordinates
            x = np.concatenate([on_x, off_x]).astype(np.uint16)
            y = np.concatenate([on_y, off_y]).astype(np.uint16)
            
            # All events get the frame timestamp (could interpolate for more realism)
            t = np.full(num_events, timestamp_us, dtype=np.int64)
            
            # Polarity: 1 for ON, 0 for OFF
            p = np.concatenate([
                np.ones(num_on, dtype=np.uint8),
                np.zeros(num_off, dtype=np.uint8)
            ])
            
            # Add shot noise (optional)
            if self.shot_noise_rate > 0:
                dt_sec = (timestamp_us - self._last_timestamp_us) / 1e6
                expected_noise = self.shot_noise_rate * height * width * dt_sec
                num_noise = np.random.poisson(expected_noise)
                if num_noise > 0:
                    noise_x = np.random.randint(0, width, num_noise).astype(np.uint16)
                    noise_y = np.random.randint(0, height, num_noise).astype(np.uint16)
                    noise_t = np.random.randint(self._last_timestamp_us, timestamp_us + 1, num_noise).astype(np.int64)
                    noise_p = np.random.randint(0, 2, num_noise).astype(np.uint8)
                    
                    x = np.concatenate([x, noise_x])
                    y = np.concatenate([y, noise_y])
                    t = np.concatenate([t, noise_t])
                    p = np.concatenate([p, noise_p])
            
            # Sort by timestamp
            sort_idx = np.argsort(t)
            x, y, t, p = x[sort_idx], y[sort_idx], t[sort_idx], p[sort_idx]
            
            # Update reference where events occurred
            event_mask = on_mask | off_mask
            self._reference_log_intensity[event_mask] = log_intensity[event_mask]
            self._last_event_time[event_mask] = timestamp_us
        else:
            x = np.array([], dtype=np.uint16)
            y = np.array([], dtype=np.uint16)
            t = np.array([], dtype=np.int64)
            p = np.array([], dtype=np.uint8)
        
        self._last_timestamp_us = timestamp_us
        
        return EventArray(x=x, y=y, t=t, p=p)
    
    def convert_sequence(
        self,
        frames: np.ndarray,
        timestamps_us: np.ndarray,
    ) -> EventArray:
        """
        Convert a sequence of RGB frames to events.
        
        Args:
            frames: Array of frames (N, H, W, C) or (N, H, W)
            timestamps_us: Array of timestamps in microseconds (N,)
            
        Returns:
            EventArray containing all generated events
        """
        self.reset()
        
        all_x, all_y, all_t, all_p = [], [], [], []
        
        for i, (frame, ts) in enumerate(zip(frames, timestamps_us)):
            events = self.convert_frame(frame, ts)
            if len(events) > 0:
                all_x.append(events.x)
                all_y.append(events.y)
                all_t.append(events.t)
                all_p.append(events.p)
        
        if all_x:
            return EventArray(
                x=np.concatenate(all_x),
                y=np.concatenate(all_y),
                t=np.concatenate(all_t),
                p=np.concatenate(all_p),
            )
        else:
            return EventArray(
                x=np.array([], dtype=np.uint16),
                y=np.array([], dtype=np.uint16),
                t=np.array([], dtype=np.int64),
                p=np.array([], dtype=np.uint8),
            )


def events_to_frame(
    events: EventArray,
    height: int,
    width: int,
    accumulation_time_us: Optional[int] = None,
) -> np.ndarray:
    """
    Render events to a visualization frame.
    
    Args:
        events: EventArray to render
        height: Frame height
        width: Frame width
        accumulation_time_us: If provided, only render events within this time window
        
    Returns:
        RGB visualization frame (H, W, 3) with ON=red, OFF=blue
    """
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    
    if len(events) == 0:
        return frame
    
    if accumulation_time_us is not None and len(events) > 0:
        t_max = events.t.max()
        t_min = t_max - accumulation_time_us
        mask = events.t >= t_min
        x, y, p = events.x[mask], events.y[mask], events.p[mask]
    else:
        x, y, p = events.x, events.y, events.p
    
    # ON events = red, OFF events = blue
    on_mask = p == 1
    off_mask = p == 0
    
    # Set ON events (red)
    frame[y[on_mask], x[on_mask]] = [255, 0, 0]
    # Set OFF events (blue)  
    frame[y[off_mask], x[off_mask]] = [0, 0, 255]
    
    return frame
