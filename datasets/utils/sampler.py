import numpy as np

from .buffer import CompressedTrajectoryBuffer


class TrajectorySampler:
    """
    A class that samples sequences of observations and actions from a trajectory buffer.
    """

    def __init__(
        self,
        buffer: CompressedTrajectoryBuffer,
        seq_len: int,
        episode_mask: np.ndarray = None,
    ):
        """
        Initialize the trajectory sampler.

        Args:
            buffer: The trajectory buffer containing the data.
            seq_len: The length of the sequences to sample.
            episode_mask: A binary mask indicating valid episodes. If None, all episodes are valid.
        """
        self.buffer = buffer
        self.seq_len = seq_len
        self.keys = list(self.buffer.keys())

        # Compute all possible sample indices
        indices = []
        episode_start = 0
        for i, episode_end in enumerate(self.buffer.episode_ends):
            if episode_mask is None or episode_mask[i]:
                for j in range(episode_start, episode_end + 1 - seq_len):
                    indices.append([j, j + seq_len])
            episode_start = episode_end
        self.indices = np.array(indices, dtype=np.int64)
        print(f"Total number of valid sequences: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def sample_sequence(self, index: int) -> dict[str, np.ndarray]:
        start, end = self.indices[index]
        data = {}
        for key in self.keys:
            arr = self.buffer[key]
            value = arr[start:end]
            data[key] = value
        return data

    def sample_episode_final_step(self, index: int) -> dict[str, np.ndarray]:
        """
        Sample the final step from the episode that contains the current sequence index.

        Args:
            index: Index into `self.indices`, i.e., the same index used by `sample_sequence`.

        Returns:
            A dictionary containing one-step arrays for all buffer keys at the final
            timestep of the corresponding episode. Each value has shape (1, ...).
        """
        start, _ = self.indices[index]
        episode_idx = np.searchsorted(self.buffer.episode_ends, start, side="right")
        episode_end = int(self.buffer.episode_ends[episode_idx])
        final_step = episode_end - 1

        data = {}
        for key in self.keys:
            arr = self.buffer[key]
            data[key] = arr[final_step : final_step + 1]
        return data
