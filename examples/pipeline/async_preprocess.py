from typing import Any, List


# register with --engine custom_async
# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        pass

    async def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        # we expect to get two valid on the dict x0, and x1
        return body

    async def postprocess(self, data: List[dict], state: dict, collect_custom_statistics_fn=None) -> dict:
        # we will here average the results and return the new value
        # assume data is a list of dicts greater than 1

        # average result
        return dict(y=0.5 * data[0]['y'][0] + 0.5 * data[1]['y'][0])

    async def process(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> Any:
        """
        do something with the actual data, return any type of object.
        The returned object will be passed as is to the postprocess function engine
        """
        predict_a = self.send_request(endpoint="/test_model_sklearn_a/", version=None, data=data)
        predict_b = self.send_request(endpoint="/test_model_sklearn_b/", version=None, data=data)

        predict_a = await predict_a
        predict_b = await predict_b

        if not predict_b or not predict_a:
            raise ValueError("Error requesting inference endpoint test_model_sklearn a/b")

        return [predict_a, predict_b]

    async def send_request(self, endpoint, version, data) -> List[dict]:
        # Mock Function!
        # replaced by real send request function when constructed by the inference service
        pass
