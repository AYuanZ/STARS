import contextlib
import torch
import time

class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0, device: torch.device = None):
        """Initializes a profiling context for YOLOv5 with optional timing threshold and device specification."""
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))
        # print("-----------profile", self.cuda)

    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()

class Profile2(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0, device: torch.device = None):
        """Initializes a profiling context for YOLOv5 with optional timing threshold and device specification."""
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time

    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()
