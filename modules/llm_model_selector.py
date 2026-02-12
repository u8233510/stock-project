from typing import Any, Dict


def get_llm_model(cfg: Dict[str, Any], task: str, fallback: str = "meta/llama-3.3-70b-instruct") -> str:
    """Read task-specific model from config.

    Priority:
    1) llm.models.<task>
    2) llm.model (legacy single-model key)
    3) fallback
    """
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    models_cfg = llm_cfg.get("models", {}) if isinstance(llm_cfg, dict) else {}

    task_model = models_cfg.get(task) if isinstance(models_cfg, dict) else None
    if task_model:
        return str(task_model)

    legacy_model = llm_cfg.get("model") if isinstance(llm_cfg, dict) else None
    if legacy_model:
        return str(legacy_model)

    return fallback
