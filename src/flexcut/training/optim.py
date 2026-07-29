from __future__ import annotations

from typing import Any, Callable, Optional


def _build_param_groups(model):
    muon_params = []
    adam_params = []
    modules = dict(model.named_modules())

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(module_name, model)
        is_embedding = "embedding" in module.__class__.__name__.lower()

        if is_embedding:
            adam_params.append(parameter)
        elif parameter.ndim >= 2 and parameter.shape[0] > 1 and parameter.shape[1] > 1:
            muon_params.append(parameter)
        else:
            adam_params.append(parameter)

    return [
        {"params": muon_params, "use_muon": True},
        {"params": adam_params, "use_muon": False},
    ]


def build_optimizer_and_scheduler(
    model,
    *,
    optimizer_cls: Callable[..., Any],
    optimizer_kwargs: Optional[dict[str, Any]] = None,
    scheduler_cls: Optional[Callable[..., Any]] = None,
    scheduler_kwargs: Optional[dict[str, Any]] = None,
):
    optimizer_kwargs = optimizer_kwargs or {}
    scheduler_kwargs = scheduler_kwargs or {}

    if optimizer_cls.__name__ == "SingleDeviceMuonWrapper":
        optimizer = optimizer_cls(
            param_groups=_build_param_groups(model),
            **optimizer_kwargs,
        )
    else:
        optimizer = optimizer_cls(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            **optimizer_kwargs,
        )

    scheduler = None
    if scheduler_cls is not None:
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    return optimizer, scheduler
