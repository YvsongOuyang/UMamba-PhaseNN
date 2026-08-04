"""Backward compatibility shim for the local UMamba wrapper.

The actual UMamba wrapper lives in umamba_model.py. This module intentionally
re-exports the old names so existing commands that import model_tf_compatible
continue to run, while avoiding the previous bare `from UMambaEnc_3d import ...`
path ambiguity.
"""

try:
    from .umamba_model import (
        LOCAL_UMAMBA_FILE,
        TFCompatibleAutoPhaseNN,
        UMambaAutoPhaseNN,
        build_umamba_plans,
        count_parameters,
        extract_state_dict,
        load_weights,
        main,
    )
except ImportError:
    from umamba_model import (
        LOCAL_UMAMBA_FILE,
        TFCompatibleAutoPhaseNN,
        UMambaAutoPhaseNN,
        build_umamba_plans,
        count_parameters,
        extract_state_dict,
        load_weights,
        main,
    )


__all__ = [
    "LOCAL_UMAMBA_FILE",
    "TFCompatibleAutoPhaseNN",
    "UMambaAutoPhaseNN",
    "build_umamba_plans",
    "count_parameters",
    "extract_state_dict",
    "load_weights",
]


if __name__ == "__main__":
    main()
