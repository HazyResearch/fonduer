"""Fonduer tensorboard logger."""
from tensorboardX import SummaryWriter


class TensorBoardLogger(object):
    """A class for logging to Tensorboard during training process."""

    def __init__(self, log_dir: str):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar variable."""
        self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        """Close the tensorboard logger."""
        self.writer.close()
