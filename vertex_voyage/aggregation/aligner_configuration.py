import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Tuple

import torch

from vertex_voyage.aggregation.alignment_objective import AlignmentObjective
from vertex_voyage.aggregation.interface import aggregation_error

ITEM_SEPARATOR = ", "
ConfigurationItem = Tuple[str, str]


def text_of_setting(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    else:
        return repr(value)


def setting_items_of(settings: Any) -> List[ConfigurationItem]:
    if dataclasses.is_dataclass(settings) and not isinstance(settings, type):
        return [
            (field.name, text_of_setting(getattr(settings, field.name)))
            for field in dataclasses.fields(settings)
        ]
    else:
        raise aggregation_error(
            f"Aligner settings must be a dataclass instance, got {settings!r}"
        )


def runtime_items() -> List[ConfigurationItem]:
    return [
        ("torch_version", torch.__version__),
        ("torch_threads", str(torch.get_num_threads())),
    ]


@dataclass(frozen=True)
class AlignerConfiguration:
    objective: AlignmentObjective
    output_scale: str
    seed: int
    settings: Any
    epoch_loss_log_interval: int

    def items(self) -> List[ConfigurationItem]:
        return (
            [
                ("objective", type(self.objective).__name__),
                ("loss", self.objective.loss.__name__),
                ("margin", repr(self.objective.margin)),
                ("effective_output_scale", self.output_scale),
                ("seed", str(self.seed)),
                (
                    "epoch_loss_log_interval",
                    str(self.epoch_loss_log_interval),
                ),
            ]
            + setting_items_of(self.settings)
            + runtime_items()
        )

    def report(self) -> str:
        return ITEM_SEPARATOR.join(
            "%s=%s" % (name, value) for name, value in self.items()
        )
