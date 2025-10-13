from dataclasses import dataclass


@dataclass
class Resolution:
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)
