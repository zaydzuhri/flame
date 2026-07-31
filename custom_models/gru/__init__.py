from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_gru import GRUConfig
from .modeling_gru import GRUForCausalLM, GRUModel

__all__ = ["GRUConfig", "GRUForCausalLM", "GRUModel"]

AutoConfig.register("gru", GRUConfig)
AutoModel.register(GRUConfig, GRUModel)
AutoModelForCausalLM.register(GRUConfig, GRUForCausalLM)
