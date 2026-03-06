def get_model(cfg):
    if cfg.model_name == "llama":
        from .llama import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_dynamic_tanh":
        from .llama_dynamic_tanh import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_gelu_ffn":
        from .llama_gelu_ffn import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_headqk_output_normalization":
        from .llama_headqk_output_normalization import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_linear_ffn":
        from .llama_linear_ffn import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_qk_output_normalization":
        from .llama_qk_output_normalization import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_sandwich":
        from .llama_sandwich import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_attn_only":
        from .llama_attention_only import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_gated_attention":
        from .llama_gated_attention import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_unconditional_gated_attention":
        from .llama_unconditional_gated_attention import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_ffn_only":
        from .llama_ffn_only import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_duo_attention":
        from .llama_duo_attention import Llama, LlamaConfig, LlamaBlock
    elif cfg.model_name == "llama_embedding_gated_attention":
        from .llama_emb_gated_attention import Llama, LlamaConfig, LlamaBlock
    else:
        raise ValueError(f"Unsupported model name: {cfg.model_name}")

    llama_config = LlamaConfig(**cfg.model_config.to_dict())
    model = Llama(llama_config)
    return model, LlamaBlock
