from .agem import AGEMPlugin
from .cwr_star import CWRStarPlugin
from .evaluation import EvaluationPlugin
from .ewc import EWCPlugin
from .gdumb import GDumbPlugin
from .gem import GEMPlugin
from .lwf import LwFPlugin
from .replay import ReplayPlugin
from .momentum_encoder import MomentumPlugin
from .replay_similarity import SimilarityReplayPlugin
from .strategy_plugin import SupervisedPlugin
from .synaptic_intelligence import SynapticIntelligencePlugin
from .gss_greedy import GSS_greedyPlugin
from .cope import CoPEPlugin, PPPloss
from .lfl import LFLPlugin
from .early_stopping import EarlyStoppingPlugin,EarlyStoppingPluginForTrainMetric
from .lr_scheduling import LRSchedulerPlugin
from .generative_replay import (
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
)
from .rwalk import RWalkPlugin
from .mas import MASPlugin
from .bic import BiCPlugin
from .mir import MIRPlugin
from .from_scratch_training import FromScratchTrainingPlugin
from .one_hot_plugin import *
from avalanche.training.multi_label_plugins.multilabel_replay import *
from avalanche.training.multi_label_plugins.multilabel_replay_prs import *
from avalanche.training.multi_label_plugins.multilabel_replay_ocdm import *
from avalanche.training.multi_label_plugins.er_multilabel_replay import *
