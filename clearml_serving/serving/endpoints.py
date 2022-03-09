import numpy as np
from attr import attrib, attrs, asdict


def _engine_validator(inst, attr, value):  # noqa
    from .preprocess_service import BasePreprocessRequest
    if not BasePreprocessRequest.validate_engine_type(value):
        raise TypeError("{} not supported engine type".format(value))


def _matrix_type_validator(inst, attr, value):  # noqa
    if value and not np.dtype(value):
        raise TypeError("{} not supported matrix type".format(value))


@attrs
class ModelMonitoring(object):
    base_serving_url = attrib(type=str)  # serving point url prefix (example: "detect_cat")
    engine_type = attrib(type=str, validator=_engine_validator)  # engine type
    monitor_project = attrib(type=str, default=None)  # monitor model project (for model auto update)
    monitor_name = attrib(type=str, default=None)  # monitor model name (for model auto update, regexp selection)
    monitor_tags = attrib(type=list, default=[])  # monitor model tag (for model auto update)
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
    model_id = attrib(type=str, default=None)  # model ID to serve (and download)
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


@attrs
class CanaryEP(object):
    endpoint = attrib(type=str)  # load balancer endpoint
    weights = attrib(type=list)  # list of weights (order should be matching fixed_endpoints or prefix)
    load_endpoints = attrib(type=list, default=[])  # list of endpoints to balance and route
    load_endpoint_prefix = attrib(
        type=str, default=None)  # endpoint prefix to list
    # (any endpoint starting with this prefix will be listed, sorted lexicographically, or broken into /<int>)

    def as_dict(self, remove_null_entries=False):
        if not remove_null_entries:
            return asdict(self)
        return {k: v for k, v in asdict(self).items() if v is not None}
