import numpy as np
from attr import attrib, attrs, asdict, validators


def _engine_validator(inst, attr, value):  # noqa
    from .preprocess_service import BasePreprocessRequest
    if not BasePreprocessRequest.validate_engine_type(value):
        raise TypeError("{} not supported engine type".format(value))


def _matrix_type_validator(inst, attr, value):  # noqa
    if isinstance(value, (tuple, list)):
        for v in value:
            if v and not np.dtype(v):
                raise TypeError("{} not supported matrix type".format(v))

    elif value and not np.dtype(value):
        raise TypeError("{} not supported matrix type".format(value))


def _list_type_convertor(inst):  # noqa
    if inst is None:
        return None
    return inst if isinstance(inst, (tuple, list)) else [inst]


def _nested_list_type_convertor(inst):  # noqa
    if inst is None:
        return None
    if isinstance(inst, (tuple, list)) and all(not isinstance(i, (tuple, list)) for i in inst):
        return [inst]
    inst = inst if isinstance(inst, (tuple, list)) else [inst]
    return inst


@attrs
class BaseStruct(object):
    def as_dict(self, remove_null_entries=False):
        if not remove_null_entries:
            return asdict(self)
        return {k: v for k, v in asdict(self).items() if v is not None}


@attrs
class ModelMonitoring(BaseStruct):
    base_serving_url = attrib(type=str)  # serving point url prefix (example: "detect_cat")
    engine_type = attrib(type=str, validator=_engine_validator)  # engine type
    monitor_project = attrib(type=str, default=None)  # monitor model project (for model auto update)
    monitor_name = attrib(type=str, default=None)  # monitor model name (for model auto update, regexp selection)
    monitor_tags = attrib(type=list, default=[])  # monitor model tag (for model auto update)
    only_published = attrib(type=bool, default=False)  # only select published models
    max_versions = attrib(type=int, default=None)  # Maximum number of models to keep serving (latest X models)
    input_size = attrib(type=list, default=None, converter=_nested_list_type_convertor)  # optional,  model matrix size
    input_type = attrib(type=list, default=None, validator=_matrix_type_validator, converter=_list_type_convertor)
    input_name = attrib(type=list, default=None, converter=_list_type_convertor)  # optional, input layer names
    output_size = attrib(type=list, default=None, converter=_nested_list_type_convertor)  # optional, model matrix size
    output_type = attrib(type=list, default=None, validator=_matrix_type_validator, converter=_list_type_convertor)
    output_name = attrib(type=list, default=None, converter=_list_type_convertor)  # optional, output layer names
    preprocess_artifact = attrib(
        type=str, default=None)  # optional artifact name storing the model preprocessing code
    auxiliary_cfg = attrib(type=dict, default=None)  # Auxiliary configuration (e.g. triton conf), Union[str, dict]


@attrs
class ModelEndpoint(BaseStruct):
    engine_type = attrib(type=str, validator=_engine_validator)  # engine type
    serving_url = attrib(type=str)  # full serving point url (including version) example: "detect_cat/v1"
    model_id = attrib(type=str, default=None)  # model ID to serve (and download)
    version = attrib(type=str, default="")  # key (version string), default no version
    preprocess_artifact = attrib(
        type=str, default=None)  # optional artifact name storing the model preprocessing code
    input_size = attrib(type=list, default=None, converter=_nested_list_type_convertor)  # optional,  model matrix size
    input_type = attrib(type=list, default=None, validator=_matrix_type_validator, converter=_list_type_convertor)
    input_name = attrib(type=list, default=None, converter=_list_type_convertor)  # optional, input layer names
    output_size = attrib(type=list, default=None, converter=_nested_list_type_convertor)  # optional, model matrix size
    output_type = attrib(type=list, default=None, validator=_matrix_type_validator, converter=_list_type_convertor)
    output_name = attrib(type=list, default=None, converter=_list_type_convertor)  # optional, output layer names
    auxiliary_cfg = attrib(type=dict, default=None)  # Optional: Auxiliary configuration (e.g. triton conf), [str, dict]


@attrs
class CanaryEP(BaseStruct):
    endpoint = attrib(type=str)  # load balancer endpoint
    weights = attrib(type=list)  # list of weights (order should be matching fixed_endpoints or prefix)
    load_endpoints = attrib(type=list, default=[])  # list of endpoints to balance and route
    load_endpoint_prefix = attrib(
        type=str, default=None)  # endpoint prefix to list
    # (any endpoint starting with this prefix will be listed, sorted lexicographically, or broken into /<int>)


@attrs
class EndpointMetricLogging(BaseStruct):
    @attrs
    class MetricType(BaseStruct):
        type = attrib(type=str, validator=validators.in_(("scalar", "enum", "value", "counter")))
        buckets = attrib(type=list, default=None)

    endpoint = attrib(type=str)  # Specific endpoint to log metrics w/ version (example: "model/1")
    # If endpoint name ends with a "*" any endpoint with a matching prefix will be selected

    log_frequency = attrib(type=float, default=None)  # Specific endpoint to log frequency
    # (0.0 to 1.0, where 1.0 is 100% of all requests are logged)

    metrics = attrib(
        type=dict, default={},
        converter=lambda x: {
            k: v if isinstance(v, EndpointMetricLogging.MetricType)
            else EndpointMetricLogging.MetricType(**v) for k, v in x.items()
        }
    )  # key=variable, value=MetricType

    # example:
    # {"x1": dict(type="scalar", buckets=[0,1,2,3]),
    #  "y": dict(type="enum", buckets=["cat", "dog"]).
    #  "latency": dict(type="value", buckets=[]).
    #  }

    def as_dict(self, remove_null_entries=False):
        if not remove_null_entries:
            return {k: v.as_dict(remove_null_entries) if isinstance(v, BaseStruct) else v
                    for k, v in asdict(self).items()}

        return {k: v.as_dict(remove_null_entries) if isinstance(v, BaseStruct) else v
                for k, v in asdict(self).items() if v is not None}
