import numpy as np
from typing import Optional, Any, Callable, List

from attr import attrib, attrs, asdict

from clearml import Task, Model
from clearml.binding.artifacts import Artifacts
from clearml.storage.util import sha256sum


def _engine_validator(inst, attr, value):  # noqa
    if not BasePreprocessRequest.validate_engine_type(value):
        raise TypeError("{} not supported engine type".format(value))


def _matrix_type_validator(inst, attr, value):  # noqa
    if value and not np.dtype(value):
        raise TypeError("{} not supported matrix type".format(value))


@attrs
class ModelMonitoring(object):
    base_serving_url = attrib(type=str)  # serving point url prefix (example: "detect_cat")
    monitor_project = attrib(type=str)  # monitor model project (for model auto update)
    monitor_name = attrib(type=str)  # monitor model name (for model auto update, regexp selection)
    monitor_tags = attrib(type=list)  # monitor model tag (for model auto update)
    engine_type = attrib(type=str, validator=_engine_validator)  # engine type
    only_published = attrib(type=bool, default=False)  # only select published models
    max_versions = attrib(type=int, default=None)  # Maximum number of models to keep serving (latest X models)
    input_size = attrib(type=list, default=None)  # optional,  model matrix size
    input_type = attrib(type=str, default=None, validator=_matrix_type_validator)  # optional, model matrix type
    input_name = attrib(type=str, default=None)  # optional, layer name to push the input to
    output_size = attrib(type=list, default=None)  # optional, model matrix size
    output_type = attrib(type=str, default=None, validator=_matrix_type_validator)  # optional, model matrix type
    output_name = attrib(type=str, default=None)  # optional, layer name to pull the results from
    preprocess_artifact = attrib(
        type=str, default=None)  # optional artifact name storing the model preprocessing code
    auxiliary_cfg = attrib(type=dict, default=None)  # Auxiliary configuration (e.g. triton conf), Union[str, dict]

    def as_dict(self, remove_null_entries=False):
        if not remove_null_entries:
            return asdict(self)
        return {k: v for k, v in asdict(self).items() if v is not None}


@attrs
class ModelEndpoint(object):
    engine_type = attrib(type=str, validator=_engine_validator)  # engine type
    serving_url = attrib(type=str)  # full serving point url (including version) example: "detect_cat/v1"
    model_id = attrib(type=str)  # list of model IDs to serve (order implies versions first is v1)
    version = attrib(type=str, default="")  # key (version string), default no version
    preprocess_artifact = attrib(
        type=str, default=None)  # optional artifact name storing the model preprocessing code
    input_size = attrib(type=list, default=None)  # optional,  model matrix size
    input_type = attrib(type=str, default=None, validator=_matrix_type_validator)  # optional, model matrix type
    input_name = attrib(type=str, default=None)  # optional, layer name to push the input to
    output_size = attrib(type=list, default=None)  # optional, model matrix size
    output_type = attrib(type=str, default=None, validator=_matrix_type_validator)  # optional, model matrix type
    output_name = attrib(type=str, default=None)  # optional, layer name to pull the results from
    auxiliary_cfg = attrib(type=dict, default=None)  # Optional: Auxiliary configuration (e.g. triton conf), [str, dict]

    def as_dict(self, remove_null_entries=False):
        if not remove_null_entries:
            return asdict(self)
        return {k: v for k, v in asdict(self).items() if v is not None}


class BasePreprocessRequest(object):
    __preprocessing_lookup = {}
    __preprocessing_modules = set()

    def __init__(
            self,
            model_endpoint: ModelEndpoint,
            task: Task = None,
            server_config: dict = None,
    ):
        """
        Notice this object is not be created per request, but once per Process
        Make sure it is always thread-safe
        """
        self.model_endpoint = model_endpoint
        self._preprocess = None
        self._model = None
        self._server_config = server_config or {}
        # load preprocessing code here
        if self.model_endpoint.preprocess_artifact:
            if not task or self.model_endpoint.preprocess_artifact not in task.artifacts:
                print("Warning: could not find preprocessing artifact \'{}\' on Task id={}".format(
                    self.model_endpoint.preprocess_artifact, task.id))
            else:
                try:
                    path = task.artifacts[self.model_endpoint.preprocess_artifact].get_local_copy()
                    # check file content hash, should only happens once?!
                    # noinspection PyProtectedMember
                    file_hash, _ = sha256sum(path, block_size=Artifacts._hash_block_size)
                    if file_hash != task.artifacts[self.model_endpoint.preprocess_artifact].hash:
                        print("INFO: re-downloading artifact '{}' hash changed".format(
                            self.model_endpoint.preprocess_artifact))
                        path = task.artifacts[self.model_endpoint.preprocess_artifact].get_local_copy(
                            extract_archive=True,
                            force_download=True,
                        )
                    else:
                        # extract zip if we need to, otherwise it will be the same
                        path = task.artifacts[self.model_endpoint.preprocess_artifact].get_local_copy(
                            extract_archive=True,
                        )

                    import importlib.util
                    spec = importlib.util.spec_from_file_location("Preprocess", path)
                    _preprocess = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(_preprocess)
                    self._preprocess = _preprocess.Preprocess()  # noqa
                    self._preprocess.serving_config = server_config or {}
                    if callable(getattr(self._preprocess, 'load', None)):
                        self._model = self._preprocess.load(self._get_local_model_file())
                except Exception as ex:
                    print("Warning: Failed loading preprocess code for \'{}\': {}".format(
                        self.model_endpoint.preprocess_artifact, ex))

    def preprocess(self, request):
        # type: (dict) -> Optional[Any]
        """
        Raise exception to report an error
        Return value will be passed to serving engine
        """
        if self._preprocess is not None:
            return self._preprocess.preprocess(request)
        return request

    def postprocess(self, data):
        # type: (Any) -> Optional[dict]
        """
        Raise exception to report an error
        Return value will be passed to serving engine
        """
        if self._preprocess is not None:
            return self._preprocess.postprocess(data)
        return data

    def process(self, data: Any) -> Any:
        """
        The actual processing function. Can be send to external service
        """
        pass

    def _get_local_model_file(self):
        model_repo_object = Model(model_id=self.model_endpoint.model_id)
        return model_repo_object.get_local_copy()

    @classmethod
    def validate_engine_type(cls, engine: str) -> bool:
        return engine in cls.__preprocessing_lookup

    @classmethod
    def get_engine_cls(cls, engine: str) -> Callable:
        return cls.__preprocessing_lookup.get(engine)

    @staticmethod
    def register_engine(engine_name: str, modules: Optional[List[str]] = None) -> Callable:
        """
        A decorator to register an annotation type name for classes deriving from Annotation
        """
        def wrapper(cls):
            cls.__preprocessing_lookup[engine_name] = cls
            return cls

        if modules:
            BasePreprocessRequest.__preprocessing_modules |= set(modules)

        return wrapper

    @staticmethod
    def load_modules() -> None:
        for m in BasePreprocessRequest.__preprocessing_modules:
            try:
                # silently fail
                import importlib
                importlib.import_module(m)
            except (ImportError, TypeError):
                pass


@BasePreprocessRequest.register_engine("triton", modules=["grpc", "tritonclient"])
class TritonPreprocessRequest(BasePreprocessRequest):
    _content_lookup = {
        np.uint8: 'uint_contents',
        np.int8: 'int_contents',
        np.int64: 'int64_contents',
        np.uint64: 'uint64_contents',
        np.int: 'int_contents',
        np.uint: 'uint_contents',
        np.bool: 'bool_contents',
        np.float32: 'fp32_contents',
        np.float64: 'fp64_contents',
    }
    _ext_grpc = None
    _ext_np_to_triton_dtype = None
    _ext_service_pb2 = None
    _ext_service_pb2_grpc = None

    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None, server_config: dict = None):
        super(TritonPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task, server_config=server_config)

        # load Triton Module
        if self._ext_grpc is None:
            import grpc
            self._ext_grpc = grpc

        if self._ext_np_to_triton_dtype is None:
            from tritonclient.utils import np_to_triton_dtype
            self._ext_np_to_triton_dtype = np_to_triton_dtype

        if self._ext_service_pb2 is None:
            from tritonclient.grpc import service_pb2, service_pb2_grpc
            self._ext_service_pb2 = service_pb2
            self._ext_service_pb2_grpc = service_pb2_grpc

    def process(self, data: Any) -> Any:
        """
        The actual processing function.
        Detect gRPC server and send the request to it
        """
        # allow to override bt preprocessing class
        if self._preprocess is not None and getattr(self._preprocess, "process", None):
            return self._preprocess.process(data)

        # Create gRPC stub for communicating with the server
        triton_server_address = self._server_config.get("triton_grpc_server")
        if not triton_server_address:
            raise ValueError("External Triton gRPC server is not configured!")
        try:
            channel = self._ext_grpc.insecure_channel(triton_server_address)
            grpc_stub = self._ext_service_pb2_grpc.GRPCInferenceServiceStub(channel)
        except Exception as ex:
            raise ValueError("External Triton gRPC server misconfigured [{}]: {}".format(triton_server_address, ex))

        # Generate the request
        request = self._ext_service_pb2.ModelInferRequest()
        request.model_name = "{}/{}".format(self.model_endpoint.serving_url, self.model_endpoint.version).strip("/")
        # we do not use the Triton model versions, we just assume a single version per endpoint
        request.model_version = "1"

        # take the input data
        input_data = np.array(data, dtype=self.model_endpoint.input_type)

        # Populate the inputs in inference request
        input0 = request.InferInputTensor()
        input0.name = self.model_endpoint.input_name
        input_dtype = np.dtype(self.model_endpoint.input_type).type
        input0.datatype = self._ext_np_to_triton_dtype(input_dtype)
        input0.shape.extend(self.model_endpoint.input_size)

        # to be inferred
        input_func = self._content_lookup.get(input_dtype)
        if not input_func:
            raise ValueError("Input type nt supported {}".format(input_dtype))
        input_func = getattr(input0.contents, input_func)
        input_func[:] = input_data.flatten()

        # push into request
        request.inputs.extend([input0])

        # Populate the outputs in the inference request
        output0 = request.InferRequestedOutputTensor()
        output0.name = self.model_endpoint.output_name

        request.outputs.extend([output0])
        response = grpc_stub.ModelInfer(request, compression=self._ext_grpc.Compression.Gzip)

        output_results = []
        index = 0
        for output in response.outputs:
            shape = []
            for value in output.shape:
                shape.append(value)
            output_results.append(
                np.frombuffer(response.raw_output_contents[index], dtype=self.model_endpoint.output_type))
            output_results[-1] = np.resize(output_results[-1], shape)
            index += 1

        # if we have a single matrix, return it as is
        return output_results[0] if index == 1 else output_results


@BasePreprocessRequest.register_engine("sklearn", modules=["joblib", "sklearn"])
class SKLearnPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None, server_config: dict = None):
        super(SKLearnPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task, server_config=server_config)
        if self._model is None:
            # get model
            import joblib
            self._model = joblib.load(filename=self._get_local_model_file())

    def process(self, data: Any) -> Any:
        """
        The actual processing function.
        We run the model in this context
        """
        return self._model.predict(data)


@BasePreprocessRequest.register_engine("xgboost", modules=["xgboost"])
class XGBoostPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None, server_config: dict = None):
        super(XGBoostPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task, server_config=server_config)
        if self._model is None:
            # get model
            import xgboost
            self._model = xgboost.Booster()
            self._model.load_model(self._get_local_model_file())

    def process(self, data: Any) -> Any:
        """
        The actual processing function.
        We run the model in this context
        """
        return self._model.predict(data)


@BasePreprocessRequest.register_engine("lightgbm", modules=["lightgbm"])
class LightGBMPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None, server_config: dict = None):
        super(LightGBMPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task, server_config=server_config)
        if self._model is None:
            # get model
            import lightgbm
            self._model = lightgbm.Booster(model_file=self._get_local_model_file())

    def process(self, data: Any) -> Any:
        """
        The actual processing function.
        We run the model in this context
        """
        return self._model.predict(data)


@BasePreprocessRequest.register_engine("custom")
class CustomPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None, server_config: dict = None):
        super(CustomPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task, server_config=server_config)

    def process(self, data: Any) -> Any:
        """
        The actual processing function.
        We run the process in this context
        """
        if self._preprocess is not None:
            return self._preprocess.process(data)
        return None
