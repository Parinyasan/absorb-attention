from .lm import LMData
from .mqar import MQAROverwrite
from .needle import Needle
from .variable_tracking import VariableTracking
from .vocab import Vocab


def make_task(name, seq_len, vocab=None, **kw):
    vocab = vocab or Vocab()
    if name == "mqar":
        return MQAROverwrite(vocab, seq_len=seq_len, **kw)
    if name == "vt":
        return VariableTracking(vocab, seq_len=seq_len, **kw)
    if name == "niah":
        return Needle(vocab, seq_len=seq_len, **kw)
    raise ValueError(name)
