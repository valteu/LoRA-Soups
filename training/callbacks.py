from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training import Trainer
from evaluation import evaluate_by_multiple_choice, evaluate_by_rouge


class TrainerCallback:

    def set_trainer(self, trainer: "Trainer"):
        self.trainer = trainer
        self.args = trainer.args
        self.state = trainer.state
        self.validation_dataset = trainer.validation_dataset
        self.tokenizer = trainer.tokenizer
        self.model = trainer.model

    @abstractmethod
    def on_step_begin(self):
        pass

    @abstractmethod
    def on_step_end(self):
        pass

    @abstractmethod
    def on_epoch_begin(self):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def on_training_begin(self):
        pass

    @abstractmethod
    def on_training_end(self):
        pass


class MCCallback(TrainerCallback):

    def on_step_end(self):
        if self.state.step % self.args.eval_steps == 0:
            pad_side = self.tokenizer.padding_side
            metrics = evaluate_by_multiple_choice(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.validation_dataset,
                batch_size=self.args.per_device_eval_batch_size,
            )
            self.state.add_event_log("mc", metrics)
            self.tokenizer.padding_side = pad_side

    def on_training_begin(self):
        pad_side = self.tokenizer.padding_side
        metrics = evaluate_by_multiple_choice(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.validation_dataset,
            batch_size=self.args.per_device_eval_batch_size,
        )
        self.state.add_event_log("mc", metrics)
        self.tokenizer.padding_side = pad_side

    def on_training_end(self):
        pad_side = self.tokenizer.padding_side
        metrics = evaluate_by_multiple_choice(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.validation_dataset,
            batch_size=self.args.per_device_eval_batch_size,
        )
        self.state.add_event_log("mc", metrics)
        self.tokenizer.padding_side = pad_side


class RougeCallback(TrainerCallback):

    def on_step_end(self):
        if self.state.step % self.args.eval_steps == 0:
            pad_side = self.tokenizer.padding_side
            metrics = evaluate_by_rouge(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.validation_dataset,
                batch_size=self.args.per_device_eval_batch_size,
            )
            self.state.add_event_log("rouge", metrics)
            self.tokenizer.padding_side = pad_side

    def on_training_begin(self):
        pass
        # pad_side = self.tokenizer.padding_side
        # metrics = evaluate_by_rouge(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     dataset=self.validation_dataset,
        #     batch_size=self.args.per_device_eval_batch_size,
        # )
        # self.state.add_event_log("rouge", metrics)
        # self.tokenizer.padding_side = pad_side

    def on_training_end(self):
        pad_side = self.tokenizer.padding_side
        metrics = evaluate_by_rouge(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.validation_dataset,
            batch_size=self.args.per_device_eval_batch_size,
        )
        self.state.add_event_log("rouge", metrics)
        self.tokenizer.padding_side = pad_side


class ModelCheckpointCallback(TrainerCallback):

    def on_step_end(self):
        if self.state.step % self.args.save_steps == 0:
            self.model.save_pretrained(self.trainer.output_directory_path / f"checkpoint-{self.state.step}")
            self.state.add_event_log("info", f"Checkpoint saved at step {self.state.step}")
