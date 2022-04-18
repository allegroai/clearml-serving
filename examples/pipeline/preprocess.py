from concurrent.futures import ThreadPoolExecutor
from typing import Any, List


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        self.executor = ThreadPoolExecutor(max_workers=32)

    def postprocess(self, data: List[dict], collect_custom_statistics_fn=None) -> dict:
        # we will here average the results and return the new value
        # assume data is a list of dicts greater than 1

        # average result
        return dict(y=0.5 * data[0]['y'][0] + 0.5 * data[1]['y'][0])

    def process(self, data: Any, collect_custom_statistics_fn=None) -> Any:
        """
        do something with the actual data, return any type of object.
        The returned object will be passed as is to the postprocess function engine
        """
        predict_a = self.executor.submit(self.send_request, endpoint="/test_model_sklearn_a/", version=None, data=data)
        predict_b = self.executor.submit(self.send_request, endpoint="/test_model_sklearn_b/", version=None, data=data)

        predict_a = predict_a.result()
        predict_b = predict_b.result()

        if not predict_b or not predict_a:
            raise ValueError("Error requesting inference endpoint test_model_sklearn a/b")

        return [predict_a, predict_b]

    def send_request(self, endpoint, version, data) -> List[dict]:
        # Mock Function!
        # replaced by real send request function when constructed by the inference service
        pass
