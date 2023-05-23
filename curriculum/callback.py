import lightning.pytorch as pl

from CL.curriculum.schedulers import CLContinuousSchedulers


class CLScheduler(pl.Callback):
    def __init__(self, n_steps: int, min_fraction: float, *args, **kwargs):
        super(CLScheduler, self).__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.min_fraction = min_fraction

    def _prepare_epoch(self, trainer, model, epoch):
        dataset_fraction = CLContinuousSchedulers(self.min_fraction).linear(
            epoch, self.n_steps
        )
        trainer.datamodule.make_curriculum_step(dataset_fraction)

    def on_train_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)
