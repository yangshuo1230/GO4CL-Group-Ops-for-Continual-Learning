"""Training config re-exports.

``TrainConfig`` lives in ``loop.py`` (used by the inner train segment).
Protocol names live next to protocol helpers.
"""

from go4cl.train.loop import TrainConfig
from go4cl.train.protocol_common import ProtocolName, ProtocolResult

__all__ = ["TrainConfig", "ProtocolName", "ProtocolResult"]
