from typing import Any, Optional

import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    serving_config = None
    # example: {
    #   'base_serving_url': 'http://127.0.0.1:8080/serve/',
    #   'triton_grpc_server': '127.0.0.1:9001',
    # }"

    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def load(self, local_file_name: str) -> Optional[Any]:
        """
        Optional, provide loading method for the model
        useful if we need to load a model in a specific way for the prediction engine to work
        :param local_file_name: file name / path to read load the model from
        :return: Object that will be called with .predict() method for inference
        """
        pass

    def preprocess(self, body: dict) -> Any:
        # do something with the request data, return any type of object.
        # The returned object will be passed as is to the inference engine
        return body

    def postprocess(self, data: Any) -> dict:
        # post process the data returned from the model inference engine
        # returned dict will be passed back as the request result as is.
        return data

    def process(self, data: Any) -> Any:
        # do something with the actual data, return any type of object.
        # The returned object will be passed as is to the postprocess function engine
        return data
