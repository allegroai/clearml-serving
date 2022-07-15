from typing import Any, Callable, Optional

import joblib
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    """
    Notice the execution flows is synchronous as follows:

    1. RestAPI(...) -> body: dict
    2. preprocess(body: dict, ...) -> data: Any
    3. process(data: Any, ...) -> data: Any
    4. postprocess(data: Any, ...) -> result: dict
    5. RestAPI(result: dict) -> returned request
    """
    def __init__(self):
        """
        Set any initial property on the Task (usually model object)
        Notice these properties will be accessed from multiple threads.
        If you need a stateful (per request) data, use the `state` dict argument passed to pre/post/process functions
        """
        # set internal state, this will be called only once. (i.e. not per request)
        self._model = None

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        """
        Optional: provide loading method for the model
        useful if we need to load a model in a specific way for the prediction engine to work
        :param local_file_name: file name / path to read load the model from
        :return: Object that will be called with .predict() method for inference
        """

        # Example now lets load the actual model

        self._model = joblib.load(local_file_name)

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        """
        Optional: do something with the request data, return any type of object.
        The returned object will be passed as is to the inference engine

        :param body: dictionary as recieved from the RestAPI
        :param state: Use state dict to store data passed to the post-processing function call.
            This is a per-request state dict (meaning a new empty dict will be passed per request)
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
            to the statictics collector servicd.
            None is passed if statiscs collector is not configured, or if the current request should not be
            collected

            Usage example:
            >>> print(body)
            {"x0": 1, "x1": 2}
            >>> if collect_custom_statistics_fn:
            >>>   collect_custom_statistics_fn({"x0": 1, "x1": 2})

        :return: Object to be passed directly to the model inference
        """

        # we expect to get a feature vector on the `feature` entry if the dict
        return np.array(body.get("features", []), dtype=np.float)

    def process(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:  # noqa
        """
        Optional: do something with the actual data, return any type of object.
        The returned object will be passed as is to the postprocess function engine

        :param data: object as recieved from the preprocessing function
        :param state: Use state dict to store data passed to the post-processing function call.
            This is a per-request state dict (meaning a dict instance per request)
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
            to the statictics collector servicd.
            None is passed if statiscs collector is not configured, or if the current request should not be collected

            Usage example:
            >>> if collect_custom_statistics_fn:
            >>>   collect_custom_statistics_fn({"type": "classification"})

        :return: Object to be passed tp the post-processing function
        """

        # this is where we do the heavy lifting, i.e. run our model.
        # notice we know data is a numpy array of type float, because this is what we prepared in preprocessing function
        data = self._model.predict(np.atleast_2d(data))
        # data is also a numpy array, as returned from our fit function
        return data

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        """
        Optional: post process the data returned from the model inference engine
        returned dict will be passed back as the request result as is.

        :param data: object as recieved from the inference model function
        :param state: Use state dict to store data passed to the post-processing function call.
            This is a per-request state dict (meaning a dict instance per request)
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
            to the statictics collector servicd.
            None is passed if statiscs collector is not configured, or if the current request should not be
            collected

            Usage example:
            >>> if collect_custom_statistics_fn:
            >>>   collect_custom_statistics_fn({"y": 1})

        :return: Dictionary passed directly as the returned result of the RestAPI
        """

        # Now we take the result numpy (predicted) and create a list of values to
        # send back as the restapi return value
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(predict=data.tolist())
