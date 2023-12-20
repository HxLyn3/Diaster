from .critic import QCritic, VCritic
from .reward import MLPReward, RNNReward
from .actor import ProbActor, DeterActor

ACTOR = {
    "prob": ProbActor,
    "deter": DeterActor,
}

CRITIC = {
    "q": QCritic,
    "v": VCritic
    
}

REWARD = {
    "mlp": MLPReward,
    "rnn": RNNReward
}
