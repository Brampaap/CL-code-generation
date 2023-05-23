class CLContinuousSchedulers:
    def __init__(self, min_fraction):
        super(CLContinuousSchedulers, self).__init__()
        self.min_fraction = min_fraction

    def validate_bounds(self, fraction: float) -> float:
        return min(
            max(fraction, self.min_fraction),
            1.0,
        )

    def linear(self, epoch: int, n_steps: int = 10) -> float:
        fraction = epoch / n_steps
        return self.validate_bounds(fraction)

    def baby_step(
        self, epoch: int, n_steps: int = 10, step_size: float = 0.2
    ) -> float:
        fraction = epoch // n_steps * step_size
        return self.validate_bounds(fraction)

    def root(self, epoch: int, n_steps: int = 10) -> float:
        fraction = (epoch / n_steps) ** (1 / 2)
        return self.validate_bounds(fraction)
