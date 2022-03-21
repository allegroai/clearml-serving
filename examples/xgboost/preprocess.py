from typing import Any

import numpy as np
import xgboost as xgb


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, collect_custom_statistics_fn=None) -> Any:
        # we expect to get four valid numbers on the dict: x0, x1, x2, x3
        return xgb.DMatrix(
            [[body.get("x0", None), body.get("x1", None), body.get("x2", None), body.get("x3", None)]])

    def postprocess(self, data: Any, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=data.tolist() if isinstance(data, np.ndarray) else data)
