"""hone-taste: stand-in scorers for human judgment.

>>> import hone_taste as tt
>>> tt.Score(0.5).value
0.5
"""

from hone_taste import errors, normalize, testing
from hone_taste._records import JsonlSpanSink, MemorySink, NullSink, SqliteSpanSink, recording
from hone_taste._tracing import current_trace
from hone_taste.agreement import AgreementReport, Spread, agreement, spread
from hone_taste.audience import AudiencePanel, audience
from hone_taste.bridge import SelectPairwise, SelectScorer, for_select, for_select_pairwise
from hone_taste.combine import combine
from hone_taste.fitting import Weights
from hone_taste.ports import (
    PORTS_VERSION,
    DecisionClient,
    GpuLease,
    NullGpuLease,
    RecordSink,
    TextClient,
    TextResult,
)
from hone_taste.profile import ConsoleIO, Profile, profile
from hone_taste.registry import ModelInfo, model_info, models
from hone_taste.scorers.audio_checks import audio_checks, audio_report
from hone_taste.scorers.audiobox import audiobox
from hone_taste.scorers.binoculars import binoculars
from hone_taste.scorers.character import character_consistency
from hone_taste.scorers.image_preference import image_preference
from hone_taste.scorers.patterns import patterns
from hone_taste.scorers.reward_model import reward_model
from hone_taste.scorers.slop import slop_score
from hone_taste.scorers.songeval import songeval
from hone_taste.scorers.stylometry import Fingerprint, fingerprint, stylometry
from hone_taste.style import StyleCalibration, calibrate_style, style_judge, style_match
from hone_taste.text_decisions import TextDecisionClient
from hone_taste.types import Score, Scorer

__version__ = "0.1.0"

__all__ = [
    "PORTS_VERSION",
    "AgreementReport",
    "AudiencePanel",
    "ConsoleIO",
    "DecisionClient",
    "Fingerprint",
    "GpuLease",
    "JsonlSpanSink",
    "MemorySink",
    "ModelInfo",
    "NullGpuLease",
    "NullSink",
    "Profile",
    "RecordSink",
    "Score",
    "Scorer",
    "SelectPairwise",
    "SelectScorer",
    "Spread",
    "SqliteSpanSink",
    "StyleCalibration",
    "TextClient",
    "TextDecisionClient",
    "TextResult",
    "Weights",
    "__version__",
    "agreement",
    "audience",
    "audio_checks",
    "audio_report",
    "audiobox",
    "binoculars",
    "calibrate_style",
    "character_consistency",
    "combine",
    "current_trace",
    "errors",
    "fingerprint",
    "for_select",
    "for_select_pairwise",
    "image_preference",
    "model_info",
    "models",
    "normalize",
    "patterns",
    "profile",
    "recording",
    "reward_model",
    "slop_score",
    "songeval",
    "spread",
    "style_judge",
    "style_match",
    "stylometry",
    "testing",
]
