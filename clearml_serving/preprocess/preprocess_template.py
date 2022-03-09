from typing import Any, Optional


# Notice Preprocess class Must be named "Preprocess"
# Otherwise there are No limitations, No need to inherit or to implement all methods
class Preprocess(object):
    serving_config = None
    # example: {
    #   'base_serving_url': 'http://127.0.0.1:8080/serve/',
    #   'triton_grpc_server': '127.0.0.1:9001',
    # }"

    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        """
        Optional, provide loading method for the model
        useful if we need to load a model in a specific way for the prediction engine to work
        :param local_file_name: file name / path to read load the model from
        :return: Object that will be called with .predict() method for inference
        """
        pass

    def preprocess(self, body: dict) -> Any:  # noqa
        """
        do something with the request data, return any type of object.
        The returned object will be passed as is to the inference engine
        """
        return body

    def postprocess(self, data: Any) -> dict:  # noqa
        """
        post process the data returned from the model inference engine
        returned dict will be passed back as the request result as is.
        """
        return data

    def process(self, data: Any) -> Any:  # noqa
        """
        do something with the actual data, return any type of object.
        The returned object will be passed as is to the postprocess function engine
        """
        return data

    def send_request(  # noqa
            self,
            endpoint: str,
            version: Optional[str] = None,
            data: Optional[dict] = None
    ) -> Optional[dict]:
        """
        NOTICE: This method will be replaced in runtime, by the inference service

        Helper method to send model inference requests to the inference service itself.
        This is designed to help with model ensemble, model pipelines, etc.
        On request error return None, otherwise the request result data dictionary

        Usage example:

        >>> x0, x1 = 1, 2
        >>> result = self.send_request(endpoint="test_model_sklearn", version="1", data={"x0": x0, "x1": x1})
        >>> y = result["y"]
        """
        return None
