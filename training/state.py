from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime


@dataclass
class State:
    step: int = field(default=0)
    epoch: int = field(default=0)
    logs: list = field(default_factory=list)
    amount_steps: int | None = field(default=None)
    verbose: bool = field(default=True)
    logging_steps: int = field(default=10)

    def add_event_log(self, type: Literal["info", "loss", "mc", "rouge"], content: dict | int | float | str):
        self.logs.append({
            "type": type,
            "content": content,
            "timestamp": str(datetime.now()),
            "step": self.step,
            "epoch": self.epoch
        })
        if self.verbose and self.step % self.logging_steps == 0:
            print(f"[{self.step}/{self.amount_steps}] {type}: {content} at {self.logs[-1]['timestamp']}")
