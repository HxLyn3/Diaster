from .replay_buffer import ReplayBuffer, EpisodicReplayBuffer

BUFFER = {
    "vanilla": ReplayBuffer,
    "episodic": EpisodicReplayBuffer
}