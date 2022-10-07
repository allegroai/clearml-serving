import io
from typing import Any, Union

import numpy as np
from PIL import Image, ImageOps


from clearml import StorageManager


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        # we expect to get two valid on the dict x0, and x1
        if isinstance(body, bytes):
            # we expect to get a stream of encoded image bytes
            try:
                image = Image.open(io.BytesIO(body)).convert("RGB")
            except Exception:
                raise ValueError("Image could not be decoded")

        if isinstance(body, dict) and "url" in body.keys():
            # image is given as url, and is fetched
            url = body.get("url")
            local_file = StorageManager.get_local_copy(remote_url=url)
            image = Image.open(local_file)
    
        image = ImageOps.grayscale(image).resize((28, 28))
        return np.array([np.array(image).flatten()])

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        if not isinstance(data, np.ndarray):
            # this should not happen
            return dict(digit=-1)

        # data is returned as probability per class (10 class/digits)
        return dict(digit=int(data.flatten().argmax()))
