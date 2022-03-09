import json
import os
from pathlib import Path
from time import sleep
from typing import Optional, Union, Dict, List
import itertools
import threading
from multiprocessing import Lock
from numpy.random import choice

from clearml import Task, Model
from clearml.storage.util import hash_dict
from .preprocess_service import BasePreprocessRequest
from .endpoints import ModelEndpoint, ModelMonitoring, CanaryEP


class FastWriteCounter(object):
    def __init__(self):
        self._counter_inc = itertools.count()
        self._counter_dec = itertools.count()

    def inc(self):
        next(self._counter_inc)

    def dec(self):
        next(self._counter_dec)

    def value(self):
        return next(self._counter_inc) - next(self._counter_dec)


class ModelRequestProcessor(object):
    _system_tag = 'serving-control-plane'

    def __init__(
            self,
            task_id: Optional[str] = None,
            update_lock_guard: Optional[Lock] = None,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None,
            force_create: bool = False,
    ) -> None:
        """
        :param task_id: Optional specify existing Task ID of the ServingService
        :param update_lock_guard: If provided use external (usually multi-process) lock guard for updates
        :param name: Optional name current serving service
        :param project: Optional select project for the current serving service
        :param tags: Optional add tags to the serving service
        :param force_create: force_create if provided, ignore task_id and create a new serving Task
        """
        self._task = self._create_task(name=name, project=project, tags=tags) \
            if force_create else self._get_control_plane_task(task_id=task_id, name=name, project=project, tags=tags)
        self._endpoints = dict()  # type: Dict[str, ModelEndpoint]
        self._model_monitoring = dict()  # type: Dict[str, ModelMonitoring]
        self._model_monitoring_versions = dict()  # type: Dict[str, Dict[int, str]]
        self._model_monitoring_endpoints = dict()  # type: Dict[str, ModelEndpoint]
        self._model_monitoring_update_request = True
        # Dict[base_serve_url, Dict[version, model_id]]
        self._canary_endpoints = dict()  # type: Dict[str, CanaryEP]
        self._canary_route = dict()  # type: Dict[str, dict]
        self._engine_processor_lookup = dict()  # type: Dict[str, BasePreprocessRequest]
        self._last_update_hash = None
        self._sync_daemon_thread = None
        # this is used for Fast locking mechanisms (so we do not actually need to use Locks)
        self._update_lock_flag = False
        self._request_processing_state = FastWriteCounter()
        self._update_lock_guard = update_lock_guard or threading.Lock()
        # serving server config
        self._configuration = {}
        self._instance_task = None

    def process_request(self, base_url: str, version: str, request_body: dict) -> dict:
        """
        Process request coming in,
        Raise Value error if url does not match existing endpoints
        """
        self._request_processing_state.inc()
        # check if we need to stall
        if self._update_lock_flag:
            self._request_processing_state.dec()
            while self._update_lock_flag:
                sleep(1)
            # retry to process
            return self.process_request(base_url=base_url, version=version, request_body=request_body)

        try:
            # normalize url and version
            url = self._normalize_endpoint_url(base_url, version)

            # check canary
            canary_url = self._process_canary(base_url=url)
            if canary_url:
                url = canary_url

            ep = self._endpoints.get(url, None) or self._model_monitoring_endpoints.get(url, None)
            if not ep:
                raise ValueError("Model inference endpoint '{}' not found".format(url))

            processor = self._engine_processor_lookup.get(url)
            if not processor:
                processor_cls = BasePreprocessRequest.get_engine_cls(ep.engine_type)
                processor = processor_cls(
                    model_endpoint=ep, task=self._task, server_config=dict(**self._configuration)
                )
                self._engine_processor_lookup[url] = processor

            return_value = self._process_request(processor=processor, url=url, body=request_body)
        finally:
            self._request_processing_state.dec()

        return return_value

    def _process_canary(self, base_url: str) -> Optional[dict]:
        canary = self._canary_route.get(base_url)
        if not canary:
            return None
        # random choice
        draw = choice(canary['endpoints'], 1, p=canary['weights'])
        # the new endpoint to use
        return draw[0]

    def configure(
            self,
            external_serving_base_url: Optional[str] = None,
            external_triton_grpc_server: Optional[str] = None,
    ):
        """
        Set ModelRequestProcessor configuration arguments.

        :param external_serving_base_url: Set the external base http endpoint for the serving service
            This URL will be passed to user custom preprocess class,
            allowing it to concatenate and combine multiple model requests into one
        :param external_triton_grpc_server: set the external grpc tcp port of the Nvidia Triton clearml container.
            Used by the clearml triton engine class to send inference requests
        """
        if external_serving_base_url is not None:
            self._task.set_parameter(
                name="General/serving_base_url",
                value=str(external_serving_base_url),
                value_type="str",
                description="external base http endpoint for the serving service"
            )
        if external_triton_grpc_server is not None:
            self._task.set_parameter(
                name="General/triton_grpc_server",
                value=str(external_triton_grpc_server),
                value_type="str",
                description="external grpc tcp port of the Nvidia Triton ClearML container running"
            )

    def add_endpoint(
            self,
            endpoint: Union[ModelEndpoint, dict],
            preprocess_code: Optional[str] = None,
            model_name: Optional[str] = None,
            model_project: Optional[str] = None,
            model_tags: Optional[List[str]] = None,
            model_published: Optional[bool] = None,
    ) -> str:
        """
        Return the unique name of the endpoint (endpoint + version)
        Overwrite existing endpoint if already exists  (outputs a warning)

        :param endpoint: New endpoint to register (overwrite existing endpoint if exists)
        :param preprocess_code: If provided upload local code as artifact
        :param model_name: If model-id not provided on, search based on model name
        :param model_project: If model-id not provided on, search based on model project
        :param model_tags: If model-id not provided on, search based on model tags
        :param model_published: If model-id not provided on, search based on model published state
        """
        if not isinstance(endpoint, ModelEndpoint):
            endpoint = ModelEndpoint(**endpoint)

        # make sure we have everything configured
        self._validate_model(endpoint)

        url = self._normalize_endpoint_url(endpoint.serving_url, endpoint.version)
        if url in self._endpoints:
            print("Warning: Model endpoint \'{}\' overwritten".format(url))

        if not endpoint.model_id and any([model_project, model_name, model_tags]):
            model_query = dict(
                project_name=model_project,
                model_name=model_name,
                tags=model_tags,
                only_published=bool(model_published),
                include_archived=False,
            )
            models = Model.query_models(max_results=2, **model_query)
            if not models:
                raise ValueError("Could not fine any Model to serve {}".format(model_query))
            if len(models) > 1:
                print("Warning: Found multiple Models for \'{}\', selecting id={}".format(model_query, models[0].id))
            endpoint.model_id = models[0].id
        elif not endpoint.model_id:
            print("Warning: No Model provided for \'{}\'".format(url))

        # upload as new artifact
        if preprocess_code:
            if not Path(preprocess_code).exists():
                raise ValueError("Preprocessing code \'{}\' could not be found".format(preprocess_code))
            preprocess_artifact_name = "py_code_{}".format(url.replace("/", "_"))
            self._task.upload_artifact(
                name=preprocess_artifact_name, artifact_object=Path(preprocess_code), wait_on_upload=True)
            endpoint.preprocess_artifact = preprocess_artifact_name

        self._endpoints[url] = endpoint
        return url

    def add_model_monitoring(
            self,
            monitoring: Union[ModelMonitoring, dict],
            preprocess_code: Optional[str] = None,
    ) -> str:
        """
        Return the unique name of the endpoint (endpoint + version)
        Overwrite existing endpoint if already exists  (outputs a warning)

        :param monitoring: Model endpoint monitor (overwrite existing endpoint if exists)
        :param preprocess_code: If provided upload local code as artifact
        :return: Unique model monitoring ID (base_model_url)
        """
        if not isinstance(monitoring, ModelMonitoring):
            monitoring = ModelMonitoring(**monitoring)

        # make sure we actually have something to monitor
        if not any([monitoring.monitor_project, monitoring.monitor_name, monitoring.monitor_tags]):
            raise ValueError("Model monitoring requires at least a "
                             "project / name / tag to monitor, none were provided.")

        # make sure we have everything configured
        self._validate_model(monitoring)

        name = monitoring.base_serving_url
        if name in self._model_monitoring:
            print("Warning: Model monitoring \'{}\' overwritten".format(name))

        # upload as new artifact
        if preprocess_code:
            if not Path(preprocess_code).exists():
                raise ValueError("Preprocessing code \'{}\' could not be found".format(preprocess_code))
            preprocess_artifact_name = "py_code_{}".format(name.replace("/", "_"))
            self._task.upload_artifact(
                name=preprocess_artifact_name, artifact_object=Path(preprocess_code), wait_on_upload=True)
            monitoring.preprocess_artifact = preprocess_artifact_name

        self._model_monitoring[name] = monitoring
        return name

    def remove_model_monitoring(self, model_base_url: str) -> bool:
        """
        Remove model monitoring, use base_model_url as unique identifier
        """
        if model_base_url not in self._model_monitoring:
            return False
        self._model_monitoring.pop(model_base_url, None)
        return True

    def remove_endpoint(self, endpoint_url: str, version: Optional[str] = None) -> bool:
        """
        Remove specific model endpoint, use base_model_url as unique identifier
        """
        endpoint_url = self._normalize_endpoint_url(endpoint_url, version)
        if endpoint_url not in self._endpoints:
            return False
        self._endpoints.pop(endpoint_url, None)
        return True

    def add_canary_endpoint(
            self,
            canary: Union[CanaryEP, dict],
    ) -> str:
        """
        Return the unique name of the endpoint (endpoint + version)
        Overwrite existing endpoint if already exists  (outputs a warning)

        :param canary: Canary endpoint router (overwrite existing endpoint if exists)
        :return: Unique canary ID (base_model_url)
        """
        if not isinstance(canary, CanaryEP):
            canary = CanaryEP(**canary)
        if canary.load_endpoints and canary.load_endpoint_prefix:
            raise ValueError(
                "Could not add canary endpoint with both "
                "prefix ({}) and fixed set of endpoints ({})".format(
                    canary.load_endpoints, canary.load_endpoint_prefix))
        name = canary.endpoint
        if name in self._canary_endpoints:
            print("Warning: Model monitoring \'{}\' overwritten".format(name))

        self._canary_endpoints[name] = canary
        return name

    def remove_canary_endpoint(self, endpoint_url: str) -> bool:
        """
        Remove specific canary model endpoint, use base_model_url as unique identifier
        """
        if endpoint_url not in self._canary_endpoints:
            return False
        self._canary_endpoints.pop(endpoint_url, None)
        return True

    def deserialize(self, task: Task = None, prefetch_artifacts=False, skip_sync=False) -> bool:
        """
        Restore ModelRequestProcessor state from Task
        return True if actually needed serialization, False nothing changed
        :param task: Load data from Task
        :param prefetch_artifacts: If True prefetch artifacts requested by the endpoints
        :param skip_sync: If True do not update the canary/monitoring state
        """
        if not task:
            task = self._task
        configuration = task.get_parameters_as_dict().get("General") or {}
        endpoints = task.get_configuration_object_as_dict(name='endpoints') or {}
        canary_ep = task.get_configuration_object_as_dict(name='canary') or {}
        model_monitoring = task.get_configuration_object_as_dict(name='model_monitoring') or {}
        hashed_conf = hash_dict(dict(endpoints=endpoints, canary_ep=canary_ep, model_monitoring=model_monitoring))
        if self._last_update_hash == hashed_conf and not self._model_monitoring_update_request:
            return False
        print("Info: syncing model endpoint configuration, state hash={}".format(hashed_conf))
        self._last_update_hash = hashed_conf

        endpoints = {
            k: ModelEndpoint(**{i: j for i, j in v.items() if hasattr(ModelEndpoint.__attrs_attrs__, i)})
            for k, v in endpoints.items()
        }
        model_monitoring = {
            k: ModelMonitoring(**{i: j for i, j in v.items() if hasattr(ModelMonitoring.__attrs_attrs__, i)})
            for k, v in model_monitoring.items()
        }
        canary_endpoints = {
            k: CanaryEP(**{i: j for i, j in v.items() if hasattr(CanaryEP.__attrs_attrs__, i)})
            for k, v in canary_ep.items()
        }

        # if there is no need to sync Canary and Models we can just leave
        if skip_sync:
            self._endpoints = endpoints
            self._model_monitoring = model_monitoring
            self._canary_endpoints = canary_endpoints
            self._configuration = configuration
            return True

        # make sure we only have one stall request at any given moment
        with self._update_lock_guard:
            # download artifacts
            # todo: separate into two, download before lock, and overwrite inside lock
            if prefetch_artifacts:
                for item in list(endpoints.values()) + list(model_monitoring.values()):
                    if item.preprocess_artifact:
                        # noinspection PyBroadException
                        try:
                            self._task.artifacts[item.preprocess_artifact].get_local_copy(
                                extract_archive=True,
                            )
                        except Exception:
                            pass

            # stall all requests
            self._update_lock_flag = True
            # wait until we have no request processed
            while self._request_processing_state.value() != 0:
                sleep(1)

            self._endpoints = endpoints
            self._model_monitoring = model_monitoring
            self._canary_endpoints = canary_endpoints
            self._configuration = configuration

            # if we have models we need to sync, now is the time
            self._sync_monitored_models()

            self._update_canary_lookup()

            # release stall lock
            self._update_lock_flag = False

            # update the state on the inference task
            if Task.current_task() and Task.current_task().id != self._task.id:
                self.serialize(task=Task.current_task())

        return True

    def serialize(self, task: Optional[Task] = None) -> None:
        """
        Store ModelRequestProcessor state into Task
        """
        if not task:
            task = self._task
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._endpoints.items()}
        task.set_configuration_object(name='endpoints', config_dict=config_dict)
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._canary_endpoints.items()}
        task.set_configuration_object(name='canary', config_dict=config_dict)
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._model_monitoring.items()}
        task.set_configuration_object(name='model_monitoring', config_dict=config_dict)

    def _update_canary_lookup(self):
        canary_route = {}
        for k, v in self._canary_endpoints.items():
            if v.load_endpoint_prefix and v.load_endpoints:
                print("Warning: Canary has both prefix and fixed endpoints, ignoring canary endpoint")
                continue
            if v.load_endpoints:
                if len(v.load_endpoints) != len(v.weights):
                    print("Warning: Canary \'{}\' weights [{}] do not match number of endpoints [{}], skipping!".format(
                        k, v.weights, v.load_endpoints))
                    continue
                endpoints = []
                weights = []
                for w, ep in zip(v.weights, v.load_endpoints):
                    if ep not in self._endpoints and ep not in self._model_monitoring_endpoints:
                        print("Warning: Canary \'{}\' endpoint \'{}\' could not be found, skipping".format(k, ep))
                        continue
                    endpoints.append(ep)
                    weights.append(float(w))
                # normalize weights
                sum_weights = sum(weights)
                weights = [w/sum_weights for w in weights]
                canary_route[k] = dict(endpoints=endpoints, weights=weights)
            elif v.load_endpoint_prefix:
                endpoints = [ep for ep in list(self._endpoints.keys()) + list(self._model_monitoring_endpoints.keys())
                             if str(ep).startswith(v.load_endpoint_prefix)]
                endpoints = sorted(
                    endpoints,
                    reverse=True,
                    key=lambda x: '{}/{:0>9}'.format('/'.join(x.split('/')[:-1]), x.split('/')[-1]) if '/' in x else x
                )
                endpoints = endpoints[:len(v.weights)]
                weights = v.weights[:len(endpoints)]
                # normalize weights
                sum_weights = sum(weights)
                weights = [w/sum_weights for w in weights]
                canary_route[k] = dict(endpoints=endpoints, weights=weights)
                self._report_text(
                    "Info: Canary endpoint \'{}\' selected [{}]".format(k, canary_route[k])
                )

        # update back
        self._canary_route = canary_route

    def _sync_monitored_models(self, force: bool = False) -> bool:
        if not force and not self._model_monitoring_update_request:
            return False
        dirty = False

        for serving_base_url, versions_model_id_dict in self._model_monitoring_versions.items():
            # find existing endpoint versions
            for ep_base_url in list(self._model_monitoring_endpoints.keys()):
                # skip over endpoints that are not our own
                if not ep_base_url.startswith(serving_base_url+"/"):
                    continue
                # find endpoint version
                _, version = ep_base_url.split("/", 1)
                if int(version) not in versions_model_id_dict:
                    # remove old endpoint
                    self._model_monitoring_endpoints.pop(ep_base_url, None)
                    dirty = True
                    continue

            # add new endpoint
            for version, model_id in versions_model_id_dict.items():
                url = "{}/{}".format(serving_base_url, version)
                if url in self._model_monitoring_endpoints:
                    continue
                model = self._model_monitoring.get(serving_base_url)
                if not model:
                    # this should never happen
                    continue
                ep = ModelEndpoint(
                    engine_type=model.engine_type,
                    serving_url=serving_base_url,
                    model_id=model_id,
                    version=str(version),
                    preprocess_artifact=model.preprocess_artifact,
                    input_size=model.input_size,
                    input_type=model.input_type,
                    output_size=model.output_size,
                    output_type=model.output_type
                )
                self._model_monitoring_endpoints[url] = ep
                dirty = True

        # filter out old model monitoring endpoints
        for ep_url in list(self._model_monitoring_endpoints.keys()):
            if not any(True for url in self._model_monitoring_versions if ep_url.startswith(url+"/")):
                self._model_monitoring_endpoints.pop(ep_url, None)
                dirty = True

        # reset flag
        self._model_monitoring_update_request = False

        if dirty:
            config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._model_monitoring_endpoints.items()}
            self._task.set_configuration_object(name='model_monitoring_eps', config_dict=config_dict)

        return dirty

    def _update_monitored_models(self):
        for model in self._model_monitoring.values():
            current_served_models = self._model_monitoring_versions.get(model.base_serving_url, {})
            # To Do: sort by updated time ?
            models = Model.query_models(
                project_name=model.monitor_project or None,
                model_name=model.monitor_name or None,
                tags=model.monitor_tags or None,
                only_published=model.only_published,
                max_results=model.max_versions,
                include_archived=False,
            )

            # check what we already have:
            current_model_id_version_lookup = dict(
                zip(list(current_served_models.values()), list(current_served_models.keys()))
            )
            versions = sorted(current_served_models.keys(), reverse=True)

            # notice, most updated model first
            # first select only the new models
            model_ids = [m.id for m in models]

            # we want last updated model to be last (so it gets the highest version number)
            max_v = 1 + (versions[0] if versions else 0)
            versions_model_ids = []
            for m_id in reversed(model_ids):
                v = current_model_id_version_lookup.get(m_id)
                if v is None:
                    v = max_v
                    max_v += 1
                versions_model_ids.append((v, m_id))

            # remove extra entries (old models)
            versions_model_ids_dict = dict(versions_model_ids[:model.max_versions])

            # mark dirty if something changed:
            if versions_model_ids_dict != current_served_models:
                self._model_monitoring_update_request = True

            # update model serving state
            self._model_monitoring_versions[model.base_serving_url] = versions_model_ids_dict

        if not self._model_monitoring_update_request:
            return False

        self._report_text("INFO: Monitored Models updated: {}".format(
            json.dumps(self._model_monitoring_versions, indent=2))
        )
        return True

    def launch(self, poll_frequency_sec=300):
        """
        Launch the background synchronization thread and monitoring thread
        (updating runtime process based on changes on the Task, and monitoring model changes in the system)
        :param poll_frequency_sec: Sync every X seconds (default 300 seconds)
        """
        if self._sync_daemon_thread:
            return

        # read state
        self.deserialize(self._task, prefetch_artifacts=True)
        # model monitoring sync
        if self._update_monitored_models():
            # update endpoints
            self.deserialize(self._task, prefetch_artifacts=True)

        # get the serving instance (for visibility and monitoring)
        self._instance_task = Task.current_task()

        # start the background thread
        with self._update_lock_guard:
            if self._sync_daemon_thread:
                return
            self._sync_daemon_thread = threading.Thread(
                target=self._sync_daemon, args=(poll_frequency_sec, ), daemon=True)
            self._sync_daemon_thread.start()
        # we return immediately

    def _sync_daemon(self, poll_frequency_sec=300):
        """
        Background thread, syncing model changes into request service.
        """
        poll_frequency_sec = float(poll_frequency_sec)
        # force mark started on the main serving service task
        self._task.mark_started(force=True)
        self._report_text("Launching - configuration sync every {} sec".format(poll_frequency_sec))
        cleanup = False
        self._update_serving_plot()
        while True:
            try:
                # this should be the only place where we call deserialize
                self._task.reload()
                if self.deserialize(self._task):
                    self._report_text("New configuration updated")
                    # mark clean up for next round
                    cleanup = True
                # model monitoring sync
                if self._update_monitored_models():
                    self._report_text("Model monitoring synced")
                    # update endpoints
                    self.deserialize(self._task)
                    # mark clean up for next round
                    cleanup = True
                # update serving layout plot
                if cleanup:
                    self._update_serving_plot()
            except Exception as ex:
                print("Exception occurred in monitoring thread: {}".format(ex))
            sleep(poll_frequency_sec)
            try:
                # we assume that by now all old deleted endpoints requests already returned
                if cleanup:
                    cleanup = False
                    for k in list(self._engine_processor_lookup.keys()):
                        if k not in self._endpoints:
                            # atomic
                            self._engine_processor_lookup.pop(k, None)
            except Exception as ex:
                print("Exception occurred in monitoring thread: {}".format(ex))

    def get_id(self) -> str:
        return self._task.id

    def get_endpoints(self) -> Dict[str, ModelEndpoint]:
        endpoints = dict(**self._endpoints)
        endpoints.update(**self._model_monitoring_endpoints)
        return endpoints

    def get_synced_endpoints(self) -> Dict[str, ModelEndpoint]:
        self._task.reload()
        _endpoints = self._task.get_configuration_object_as_dict(name='endpoints') or {}
        _monitor_endpoints = self._task.get_configuration_object_as_dict(name='model_monitoring_eps') or {}
        endpoints = {
            k: ModelEndpoint(**{i: j for i, j in v.items() if hasattr(ModelEndpoint.__attrs_attrs__, i)})
            for k, v in _endpoints.items()}
        endpoints.update({
            k: ModelEndpoint(**{i: j for i, j in v.items() if hasattr(ModelEndpoint.__attrs_attrs__, i)})
            for k, v in _monitor_endpoints.items()
        })
        return endpoints

    def get_canary_endpoints(self) -> dict:
        return self._canary_endpoints

    def get_model_monitoring(self) -> dict:
        return self._model_monitoring

    def _get_instance_id(self) -> Optional[str]:
        return self._instance_task.id if self._instance_task else None

    def _report_text(self, text) -> Optional[str]:
        return self._task.get_logger().report_text("Instance [{}, pid={}]: {}".format(
            self._get_instance_id(), os.getpid(), text))

    def _update_serving_plot(self) -> None:
        """
        Update the endpoint serving graph on the serving instance Task
        """
        if not self._instance_task:
            return

        # Generate configuration table and details
        endpoints = list(self._endpoints.values()) + list(self._model_monitoring_endpoints.values())
        if not endpoints:
            # clear plot if we had any
            return

        endpoints = [e.as_dict() for e in endpoints]
        table_values = [list(endpoints[0].keys())]
        table_values += [[e[c] or "" for c in table_values[0]] for e in endpoints]
        self._instance_task.get_logger().report_table(
            title='Serving Endpoint Configuration', series='Details', iteration=0, table_plot=table_values,
            extra_layout={"title": "Model Endpoints Details"})

        # generate current endpoint view
        sankey_node = dict(
            label=[],
            color=[],
            customdata=[],
            hovertemplate='%{customdata}<extra></extra>',
            hoverlabel={"align": "left"},
        )
        sankey_link = dict(
            source=[],
            target=[],
            value=[],
            hovertemplate='<extra></extra>',
        )
        # root
        sankey_node['color'].append("mediumpurple")
        sankey_node['label'].append('{}'.format('external'))
        sankey_node['customdata'].append("")

        sankey_node_idx = {}

        # base_url = self._task._get_app_server() + '/projects/*/models/{model_id}/general'

        # draw all static endpoints
        # noinspection PyProtectedMember
        for i, ep in enumerate(endpoints):
            serve_url = ep['serving_url']
            full_url = '{}/{}'.format(serve_url, ep['version'] or "")
            sankey_node['color'].append("blue")
            sankey_node['label'].append("/{}/".format(full_url.strip("/")))
            sankey_node['customdata'].append(
                "model id: {}".format(ep['model_id'])
            )
            sankey_link['source'].append(0)
            sankey_link['target'].append(i + 1)
            sankey_link['value'].append(1. / len(self._endpoints))
            sankey_node_idx[full_url] = i + 1

        # draw all model monitoring
        sankey_node['color'].append("mediumpurple")
        sankey_node['label'].append('{}'.format('monitoring models'))
        sankey_node['customdata'].append("")
        monitoring_root_idx = len(sankey_node['customdata']) - 1

        for i, m in enumerate(self._model_monitoring.values()):
            serve_url = m.base_serving_url
            sankey_node['color'].append("purple")
            sankey_node['label'].append('{}'.format(serve_url))
            sankey_node['customdata'].append(
                "project: {}<br />name: {}<br />tags: {}".format(
                    m.monitor_project or '', m.monitor_name or '', m.monitor_tags or '')
            )
            sankey_link['source'].append(monitoring_root_idx)
            sankey_link['target'].append(monitoring_root_idx + i + 1)
            sankey_link['value'].append(1. / len(self._model_monitoring))

            # add links to the current models
            serve_url = serve_url.rstrip("/") + "/"
            for k in sankey_node_idx:
                if k.startswith(serve_url):
                    sankey_link['source'].append(monitoring_root_idx + i + 1)
                    sankey_link['target'].append(sankey_node_idx[k])
                    sankey_link['value'].append(1.0 / m.max_versions)

        # add canary endpoints
        # sankey_node['color'].append("mediumpurple")
        # sankey_node['label'].append('{}'.format('Canary endpoints'))
        # sankey_node['customdata'].append("")
        canary_root_idx = len(sankey_node['customdata']) - 1

        # sankey_link['source'].append(0)
        # sankey_link['target'].append(canary_root_idx)
        # sankey_link['value'].append(1.)

        for i, c in enumerate(self._canary_endpoints.values()):
            serve_url = c.endpoint
            sankey_node['color'].append("green")
            sankey_node['label'].append('CANARY: /{}/'.format(serve_url.strip("/")))
            sankey_node['customdata'].append(
                "outputs: {}".format(
                    c.load_endpoints or c.load_endpoint_prefix)
            )
            sankey_link['source'].append(0)
            sankey_link['target'].append(canary_root_idx + i + 1)
            sankey_link['value'].append(1. / len(self._canary_endpoints))

            # add links to the current models
            if serve_url not in self._canary_route:
                continue
            for ep, w in zip(self._canary_route[serve_url]['endpoints'], self._canary_route[serve_url]['weights']):
                idx = sankey_node_idx.get(ep)
                if idx is None:
                    continue
                sankey_link['source'].append(canary_root_idx + i + 1)
                sankey_link['target'].append(idx)
                sankey_link['value'].append(w)

        # create the sankey graph
        dag_flow = dict(
            link=sankey_link,
            node=sankey_node,
            textfont=dict(color='rgba(0,0,0,255)', size=10),
            type='sankey',
            orientation='h'
        )
        fig = dict(data=[dag_flow], layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

        self._instance_task.get_logger().report_plotly(
            title='Serving Endpoints Layout', series='', iteration=0, figure=fig)

    @staticmethod
    def _process_request(processor: BasePreprocessRequest, url: str, body: dict) -> dict:
        # todo: add some statistics
        preprocessed = processor.preprocess(body)
        processed = processor.process(data=preprocessed)
        return processor.postprocess(data=processed)

    @classmethod
    def list_control_plane_tasks(
            cls,
            task_id: Optional[str] = None,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None
    ) -> List[dict]:

        # noinspection PyProtectedMember
        tasks = Task.query_tasks(
            task_name=name or None,
            project_name=project or None,
            tags=tags or None,
            additional_return_fields=["id", "name", "project", "tags"],
            task_filter={'type': ['service'],
                         'status': ["created", "in_progress"],
                         'system_tags': [cls._system_tag]}
        )  # type: List[dict]
        if not tasks:
            return []

        for t in tasks:
            # noinspection PyProtectedMember
            t['project'] = Task._get_project_name(t['project'])

        return tasks

    @classmethod
    def _get_control_plane_task(
            cls,
            task_id: Optional[str] = None,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None,
            disable_change_state: bool = False,
    ) -> Task:
        if task_id:
            task = Task.get_task(task_id=task_id)
            if not task:
                raise ValueError("Could not find Control Task ID={}".format(task_id))
            task_status = task.status
            if task_status not in ("created", "in_progress",):
                if disable_change_state:
                    raise ValueError(
                        "Could Control Task ID={} status [{}] "
                        "is not valid (only 'draft', 'running' are supported)".format(task_id, task_status))
                else:
                    task.mark_started(force=True)
            return task

        # noinspection PyProtectedMember
        tasks = Task.query_tasks(
            task_name=name or None,
            project_name=project or None,
            tags=tags or None,
            task_filter={'type': ['service'],
                         'status': ["created", "in_progress"],
                         'system_tags': [cls._system_tag]}
        )
        if not tasks:
            raise ValueError("Could not find any valid Control Tasks")

        if len(tasks) > 1:
            print("Warning: more than one valid Controller Tasks found, using Task ID={}".format(tasks[0]))

        return Task.get_task(task_id=tasks[0])

    @classmethod
    def _create_task(
            cls,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None
    ) -> Task:
        task = Task.create(
            project_name=project or "DevOps",
            task_name=name or "Serving Service",
            task_type="service",
        )
        task.set_system_tags([cls._system_tag])
        if tags:
            task.set_tags(tags)
        return task

    @classmethod
    def _normalize_endpoint_url(cls, endpoint: str, version: Optional[str] = None) -> str:
        return "{}/{}".format(endpoint.rstrip("/"), version or "").rstrip("/")

    @classmethod
    def _validate_model(cls, endpoint: Union[ModelEndpoint, ModelMonitoring]) -> bool:
        """
        Raise exception if validation fails, otherwise return True
        """
        if endpoint.engine_type in ("triton", ):
            # verify we have all the info we need
            d = endpoint.as_dict()
            missing = [
                k for k in [
                    'input_type', 'input_size', 'input_name',
                    'output_type', 'output_size', 'output_name',
                ] if not d.get(k)
            ]
            if not endpoint.auxiliary_cfg and missing:
                raise ValueError("Triton engine requires input description - missing values in {}".format(missing))
        return True

