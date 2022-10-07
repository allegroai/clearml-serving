"""Hugginface preprocessing module for ClearML Serving."""
from typing import Any
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType


# Notice Preprocess class Must be named "Preprocess"
class Preprocess:
    """Processing class will be run by the ClearML inference services before and after each request."""

    def __init__(self):
        """Set internal state, this will be called only once. (i.e. not per request)."""
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("philschmid/MiniLM-L6-H384-uncased-sst2")

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        """Will be run when a request comes into the ClearML inference service."""
        tokens = self.tokenizer(
            text=body['text'],
            max_length=16,
            truncation=True,
            return_tensors=TensorType.NUMPY,
        )

        return [tokens["input_ids"].tolist(), tokens["token_type_ids"].tolist(), tokens["attention_mask"].tolist()]

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        """Will be run whan a request comes back from the Triton Engine."""
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return {'data': data.tolist()}
