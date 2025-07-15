from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from evaluation import evaluate_by_rouge


class RougeCallback(TrainerCallback):

    def __init__(self, validation_dataset=None, tokenizer=None):
        self.validation_dataset = validation_dataset
        self.tokenizer = tokenizer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get('model')
        eval_dataset = self.validation_dataset or kwargs.get('eval_dataset')

        if model and eval_dataset:
            self._evaluate_and_log(model, eval_dataset, args, state, "train_begin")

        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        # Check if we should evaluate (equivalent to your eval_steps check)
        if args.eval_steps > 0 and state.global_step % args.eval_steps == 0:
            model = kwargs.get('model')
            eval_dataset = self.validation_dataset or kwargs.get('eval_dataset')

            if model and eval_dataset:
                self._evaluate_and_log(model, eval_dataset, args, state, "step_end")

        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        model = kwargs.get('model')
        eval_dataset = self.validation_dataset or kwargs.get('eval_dataset')

        if model and eval_dataset:
            self._evaluate_and_log(model, eval_dataset, args, state, "train_end")

        return control

    def _evaluate_and_log(self, model, dataset, args, state, event_type):
        """Helper method to perform evaluation and logging."""
        # Store original padding side
        original_padding_side = self.tokenizer.padding_side

        try:
            # Perform ROUGE evaluation
            metrics = evaluate_by_rouge(
                model=model,
                tokenizer=self.tokenizer,
                dataset=dataset,
                batch_size=args.per_device_eval_batch_size,
            )

            # Log metrics - you can customize this based on your logging needs
            print(f"ROUGE metrics at {event_type} (step {state.global_step}): {metrics}")

            # If you want to log to wandb, tensorboard, etc., you can access the trainer's log method
            # This would require passing the trainer or implementing custom logging

        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side


# Example usage:
"""
from transformers import Trainer, TrainingArguments

# Create your callback
rouge_callback = RougeCallback()

# Create trainer with callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[rouge_callback]
)

# Train
trainer.train()
"""
