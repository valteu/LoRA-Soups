import os
from pathlib import Path
import math
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

from training import State, DataCollator, ModelCheckpointCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:

    def __init__(
        self,
        args: TrainingArguments,
        model: AutoModelForCausalLM,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        tokenizer: AutoTokenizer,
        collator: DataCollator,
        callbacks: list,
        verbose: bool = True
    ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.tokenizer = tokenizer
        self.collator = collator
        self.callbacks = callbacks
        self.verbose = verbose
        self.output_directory_path = Path(args.output_dir)
        self.state = State(
            amount_steps=args.num_train_epochs * math.ceil(len(train_dataset) / args.per_device_train_batch_size),
            verbose=self.verbose,
            logging_steps=args.logging_steps
        )
        if args.save_steps > 0:
            self.callbacks.append(ModelCheckpointCallback())

    def train(self):
        for callback in self.callbacks:
            callback.set_trainer(self)

        self.tokenizer.padding_side = "right"
        self.model.train()

        self.output_directory_path.mkdir(parents=False, exist_ok=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)

        self._call_callback("on_training_begin")

        for epoch in range(self.args.num_train_epochs):
            self.state.epoch += 1
            self._call_callback("on_epoch_begin")

            data_batches = DataLoader(
                dataset=self.train_dataset,
                collate_fn=self.collator,
                batch_size=self.args.per_device_train_batch_size,
                worker_init_fn=lambda id: np.random.seed(id + self.state.epoch)
            )

            batch_iterator = tqdm(
                data_batches,
                desc=f"Epoch {self.state.epoch}/{self.args.num_train_epochs}",
                dynamic_ncols=True,
                leave=False
            )

            for batch in batch_iterator:
                self._call_callback("on_step_begin")

                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / self.args.gradient_accumulation_steps
                loss.backward()

                self.state.step += 1

                if self.state.step % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    self.model.zero_grad()

                self.state.add_event_log("loss", loss.item())
                batch_iterator.set_postfix(step=self.state.step, loss=f"{loss.item():.4f}")
                self._call_callback("on_step_end")

            self._call_callback("on_epoch_end")

            if self.state.step % self.args.gradient_accumulation_steps != 0:
                optimizer.step()
                self.model.zero_grad()

        self._call_callback("on_training_end")

    def _call_callback(self, method_name: str):
        for callback in self.callbacks:
            getattr(callback, method_name, lambda x: x)()
