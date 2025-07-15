import torch


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding_token = tokenizer.pad_token_id
        self.loss_mask_token = -100

    def _stack_and_pad(
        self,
        tensors,
        padding_side: str = "right",
        pad_token: int = None,
        pad_to_multiple_of: int = 16,
    ):
        if pad_token is None:
            pad_token = self.padding_token

        longest_sample_length = max(tensor.size(0) for tensor in tensors)

        if pad_to_multiple_of is not None:
            longest_sample_length = (longest_sample_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

        padded_tensors = []
        for tensor in tensors:
            padding = (0, longest_sample_length - tensor.size(0)) if padding_side == "right" else (longest_sample_length - tensor.size(0), 0)
            padded_tensors.append(torch.nn.functional.pad(tensor, padding, value=pad_token))

        return torch.stack(padded_tensors, dim=0)

    def _feature_to_tensor(self, feature):
        return torch.tensor(feature) if not isinstance(feature, torch.Tensor) else feature

    def __call__(self, features, padding_side: str = "right"):

        input_ids = [self._feature_to_tensor(f["input_ids"]) for f in features]
        labels = [
            (self._feature_to_tensor(f["labels"]) if "labels" in f else self._feature_to_tensor(f["input_ids"]))
            for f in features
        ]
        attention_masks = [torch.ones_like(inp) for inp in input_ids]
        batch = {
            "input_ids": self._stack_and_pad(input_ids, padding_side=padding_side),
            "attention_mask": self._stack_and_pad(attention_masks, padding_side=padding_side, pad_token=0),
            "labels": self._stack_and_pad(labels, pad_token=self.loss_mask_token, padding_side=padding_side),
        }
        return batch
