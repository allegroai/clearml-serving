from typing import Any

import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        # we expect to get four valid numbers on the dict: x0, x1, x2, x3
        return np.array(
            [[body.get("x0", None), body.get("x1", None), body.get("x2", None), body.get("x3", None)], ],
            dtype=np.float32
        )

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        # we pick the most probably class and return the class index (argmax)
        return dict(y=int(np.argmax(data)) if isinstance(data, np.ndarray) else data)
