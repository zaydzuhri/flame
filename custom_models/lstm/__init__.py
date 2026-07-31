from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_lstm import LSTMConfig
from .modeling_lstm import LSTMForCausalLM, LSTMModel

__all__ = ["LSTMConfig", "LSTMForCausalLM", "LSTMModel"]

AutoConfig.register("lstm", LSTMConfig)
AutoModel.register(LSTMConfig, LSTMModel)
AutoModelForCausalLM.register(LSTMConfig, LSTMForCausalLM)
