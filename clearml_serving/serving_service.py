import json
import shutil
from logging import getLogger
from pathlib import Path as Path3
from time import time
from typing import Optional, Union, Dict, Sequence

from attr import attrib, attrs, asdict
from pathlib2 import Path

from clearml import Task, Model, InputModel


class ServingService(object):
    _config_pbtxt_section = 'config.pbtxt'
    _supported_serving_engines = ('triton', 'ovms', 'kfserving')

    @attrs
    class EndPoint(object):
        serving_url = attrib(type=str)
        model_ids = attrib(type=list)
        model_project = attrib(type=str)
        model_name = attrib(type=str)
        model_tags = attrib(type=list)
        model_config_blob = attrib(type=str, default=None)
        max_num_revisions = attrib(type=int, default=None)
        versions = attrib(type=dict, default={})

        def as_dict(self):
            return asdict(self)

    def __init__(self, task_id=None, task_project=None, task_name=None, task=None, engine_type='triton'):
        # type: (Optional[str], Optional[str], Optional[str], Optional[Task], Optional[str]) -> None
        """
        :param task_id: Optional specify existing Task ID of the ServingService
        :param task_project: Select the project where the new ServingService task will be created
        :param task_name: Specify the Task name for the newly created ServingService
        :param task: Optional pass existing ServingService Task object
        :param engine_type: Specify the serving engine Type. Examples: triton, ovms, kfserving
        """
        assert engine_type in self._supported_serving_engines

        if task:
            self._task = task
        elif task_id:
            self._task = Task.get_task(task_id=task_id)
        else:
            # noinspection PyProtectedMember
            if Task._query_tasks(project_name=task_project, task_name=task_name):
                self._task = Task.get_task(project_name=task_project, task_name=task_name)
            else:
                self._task = Task.create(
                    project_name=task_project, task_name=task_name, task_type=Task.TaskTypes.service,
                    repo="https://github.com/allegroai/clearml-serving.git",
                    branch="main",
                    commit="ad049c51c146e9b7852f87e2f040e97d88848a1f",
                    script="clearml_serving/service.py",
                    working_directory=".",
                    add_task_init_call=False,
                )
                self._task.set_system_tags(list(self._task.get_system_tags()) + ['serving'])

        # self._current_serving_endpoints = {'an_enpoint_url': {1: 'model_id'}}
        self._current_serving_endpoints = {}  # type: Dict[str, Dict[int, str]]
        # self._endpoints = {'an_enpoint_url': ServingService.EndPoint()}
        self._endpoints = {}  # type: Dict[str, ServingService.EndPoint]
        self._engine_type = engine_type
        self._dirty = False
        self._last_update_step = None
        # try to deserialize from Task
        # noinspection PyBroadException
        try:
            self._deserialize()
        except Exception:
            pass

    def add_model_serving(
            self,
            serving_url,  # type: str
            model_ids=None,  # type: Optional[Sequence[str]]
            model_project=None,  # type: Optional[str]
            model_name=None,  # type: Optional[str]
            model_tags=None,  # type: Optional[Sequence[str]]
            config_file=None,  # type: Optional[Union[Path, Path3, str]]
            max_versions=1,  # type: Optional[int]
    ):
        """
        Add new model serving endpoint, automatically published

        :param serving_url:
        :param model_ids:
        :param model_project:
        :param model_name:
        :param model_tags:
        :param config_file:
        :param max_versions:
        :return:
        """
        if not serving_url:
            raise ValueError("serving_url is required")

        if model_tags and not isinstance(model_tags, (list, tuple)):
            raise ValueError("model_tags must be a list of strings")

        # normalize endpoint url
        serving_url = str(serving_url).strip('/')

        endpoint = self.EndPoint(
            serving_url=serving_url,
            model_ids=list(model_ids) if model_ids else None,
            model_name=model_name,
            model_project=model_project,
            model_tags=model_tags,
            max_num_revisions=max_versions or None,
            versions={},
            model_config_blob='',
        )
        # load config file
        if config_file:
            with open(str(config_file), 'rt') as f:
                endpoint.model_config_blob = f.read()
        else:
            # Look for the config on the Model generated Task
            found_models = Model.query_models(project_name=model_project, model_name=model_name, tags=model_tags) or []

            selected_model = None
            # find the first model with config.pbtxt configuration
            # prefer published models
            found_models = [m for m in found_models if m.published] + [m for m in found_models if not m.published]
            for m in found_models:
                task_id = m.task
                task = Task.get_task(task_id=task_id)
                config_pbtxt = task.get_configuration_object(self._config_pbtxt_section)
                if config_pbtxt and str(config_pbtxt).strip():
                    endpoint.model_config_blob = config_pbtxt
                    selected_model = m
                    break

            if not selected_model:
                raise ValueError(
                    "Requested Model project={} name={} tags={} not found. 'config.pbtxt' could not be inferred. "
                    "please provide specific config.pbtxt definition.".format(model_project, model_name, model_tags))
            elif len(found_models) > 1:
                getLogger('clearml-serving').warning(
                    "Found more than one Model, using model id={}".format(selected_model.id))

        self._endpoints[serving_url] = endpoint
        self._dirty = True

    def launch(self,  queue_name='services', queue_id=None, force=False, verbose=True):
        # type: (Optional[str], Optional[str], bool, bool) -> None
        """
        Launch serving service on a remote machine using the specified queue

        :param queue_name: Queue name to launch the serving service control plane
        :param queue_id: specify queue id (unique stand stable) instead of queue_name
        :param force: if False check if service Task is already running before enqueuing
        :param verbose: If True print progress to console
        """
        # check if we are not already running
        if not force and ((self._task.data.execution.queue and self._task.status == 'in_progress')
                          or self._task.status == 'queued'):
            if verbose:
                print('Serving service already running')
        else:
            if verbose:
                print('Launching Serving service on {} queue'.format(queue_id or queue_name))
            self.update_endpoint_graph(force=True)
            self.update_model_endpoint_state()
            self.serialize()
            self._task.flush(wait_for_uploads=True)
            self._task.reset()
            self._task.enqueue(task=self._task, queue_name=queue_name, queue_id=queue_id)

    def launch_engine(self, queue_name, queue_id=None, verbose=True):
        # type: (Optional[str], Optional[str], bool) -> None
        """
        Launch serving engine on a specific queue

        :param queue_name: Queue name to launch the engine service running the inference on.
        :param queue_id: specify queue id (unique stand stable) instead of queue_name
        :param verbose: If True print progress to console
        """

        # todo: add more engines
        if self._engine_type == 'triton':
            # create the serving engine Task
            engine_task = Task.create(
                project_name=self._task.get_project_name(),
                task_name="triton serving engine",
                task_type=Task.TaskTypes.inference,
                repo="https://github.com/allegroai/clearml-serving.git",
                branch="main",
                commit="ad049c51c146e9b7852f87e2f040e97d88848a1f",
                script="clearml_serving/triton_helper.py",
                working_directory=".",
                docker="nvcr.io/nvidia/tritonserver:21.03-py3 --ipc=host -p 8000:8000 -p 8001:8001 -p 8002:8002",
                argparse_args=[('serving_id', self._task.id), ],
                add_task_init_call=False,
            )
            if verbose:
                print('Launching engine {} on queue {}'.format(self._engine_type, queue_id or queue_name))
            engine_task.enqueue(task=engine_task, queue_name=queue_name, queue_id=queue_id)

    def update_endpoint_graph(self, force=False):
        # type: (bool) -> None
        """
        Update the endpoint serving graph

        :param force: If True always update, otherwise skip if service was not changed since lat time
        """
        if not force and not self._dirty:
            return

        # Generate configuration table and details
        table_values = [["Endpoint", "Model ID", "Model Project", "Model Name", "Model Tags", "Max Versions"]]
        for endpoint in sorted(self._endpoints.keys()):
            n = self._endpoints[endpoint]
            table_values.append([
                str(n.serving_url or ''),
                str(n.model_ids or ''),
                str(n.model_project or ''),
                str(n.model_name or ''),
                str(n.model_tags or ''),
                str(n.max_num_revisions or '')
            ])
        self._task.get_logger().report_table(
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
        sankey_node['label'].append('{}'.format('serving'))
        sankey_node['customdata'].append("")

        # Generate table and details
        table_values = [["Endpoint", "Version", "Model ID"]]
        # noinspection PyProtectedMember
        base_url = self._task._get_app_server() + '/projects/*/models/{model_id}/general'
        for i, serve_url in enumerate(sorted(self._endpoints.keys())):
            ep = self._endpoints[serve_url]
            sankey_node['color'].append("blue")
            sankey_node['label'].append('{}'.format(serve_url))
            sankey_node['customdata'].append(
                "project: {}<br />name: {}<br />tags: {}".format(
                    ep.model_project or '', ep.model_name or '', ep.model_tags or '')
            )
            sankey_link['source'].append(0)
            sankey_link['target'].append(i + 1)
            sankey_link['value'].append(1. / len(self._endpoints))

            for v in sorted(self._current_serving_endpoints.get(serve_url, [])):
                model_id = self._current_serving_endpoints[serve_url][v]
                href = '<a href="{}"> {} </a>'.format(base_url.format(model_id=model_id), model_id)
                table_values.append([str(serve_url), str(v), href])
                sankey_node['color'].append("lightblue")
                sankey_node['label'].append('{}'.format(v))
                sankey_node['customdata'].append(model_id)

                sankey_link['source'].append(i + 1)
                sankey_link['target'].append(len(sankey_node['color']) - 1)
                sankey_link['value'].append(1. / len(self._current_serving_endpoints[serve_url]))

        # create the sankey graph
        dag_flow = dict(
            link=sankey_link,
            node=sankey_node,
            textfont=dict(color='rgba(0,0,0,255)', size=10),
            type='sankey',
            orientation='h'
        )
        fig = dict(data=[dag_flow], layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

        self._task.get_logger().report_plotly(
            title='Model Serving Endpoints', series='', iteration=0, figure=fig)

        # report detailed table
        self._task.get_logger().report_table(
            title='Serving Endpoint', series='Details', iteration=0, table_plot=table_values,
            extra_layout={"title": "Model Endpoints Details"})

        self._dirty = False

    def update_model_endpoint_state(self):
        # type: () -> bool
        """
        Update model endpoint state from the model repository

        :return: True if endpoints were updated
        """

        for endpoint, node in self._endpoints.items():
            # model ID supersedes everything
            if node.model_ids:
                model_ids = node.model_ids
            else:
                # get list of models sorted by descending update time
                models = Model.query_models(
                    project_name=node.model_project,
                    model_name=node.model_name,
                    tags=node.model_tags
                )
                # prefer published models
                model_ids = [m.id for m in models if m.published] + [m.id for m in models if not m.published]

            cur_endpoint = self._current_serving_endpoints.get(node.serving_url, {})
            cur_endpoint = {int(k): v for k, v in cur_endpoint.items() if v in model_ids}
            cur_endpoint_m_ids = list(cur_endpoint.values())
            max_v = max(list(cur_endpoint.keys()) or [0])
            for i, m_id in enumerate(model_ids):
                # only pick the latest in the history
                if node.max_num_revisions and max_v >= node.max_num_revisions:
                    break

                if m_id in cur_endpoint_m_ids:
                    continue
                max_v += 1
                cur_endpoint[max_v] = m_id

            # check if we need to update,
            if self._current_serving_endpoints.get(node.serving_url) != cur_endpoint:
                # set dirty flag
                self._dirty = True
                # store updated results
                self._current_serving_endpoints[node.serving_url] = cur_endpoint

        return self._dirty

    def stats(self):
        pass

    def get_endpoints(self):
        # type: () -> Dict[str, ServingService.EndPoint]
        """
        return the internal endpoints configuration

        :return: dict where the keys is the endpoint url and the value is the endpoint configuration
        """
        return self._endpoints

    def get_endpoint_version_model_id(self, serving_url):
        # type: (str) -> Dict[int, str]
        """
        Return dict with model versions and model id for the specific serving url
        If serving url is not found, return None

        :param serving_url: sering url string

        :return: dictionary keys are the versions (integers) and values are the model IDs (str)
        """
        return self._current_serving_endpoints.get(serving_url) or {}

    def _serialize(self):
        configuration = dict()
        for name, ep in self._endpoints.items():
            # noinspection PyProtectedMember
            self._task.set_configuration_object(
                name="model.{}".format(name),
                description='Model Serving Configuration',
                config_type='pbtxt',
                config_text=ep.model_config_blob)
            ep_conf = ep.as_dict()
            ep_conf.pop('model_config_blob', None)
            configuration['"{}"'.format(name)] = ep_conf
        # noinspection PyProtectedMember
        self._task._set_configuration(
            config_dict=configuration, name='endpoints',
            config_type='hocon', description='Serving Endpoints Configuration')
        # set configuration of current served endpoints
        # noinspection PyProtectedMember
        self._task._set_configuration(
            config_dict=self._current_serving_endpoints, name='serving_state',
            config_type='hocon', description='Current Serving Endpoints State',
        )
        serving = dict(engine=self._engine_type)
        self._task.connect(serving, name='serving')

    def _deserialize(self):
        # type: () -> bool
        """
        deserialize internal state from Task backend

        :return: return True if new state a was updated.
        """
        # update if the task was updated
        if self._endpoints:
            last_update = self._task.data.last_update
            try:
                # noinspection PyProtectedMember
                if last_update == self._task._get_last_update():
                    return True
            except AttributeError:
                # support old clearml packages
                pass

        self._task.reload()

        # noinspection PyProtectedMember
        configuration = self._task._get_configuration_dict(name='endpoints')
        if not configuration:
            return False

        self._endpoints = {}
        self._current_serving_endpoints = {}
        serving = dict(engine='')
        task_parameters = self._task.get_parameters_as_dict()
        serving.update(task_parameters.get('serving', {}))
        self._engine_type = serving['engine']

        for name, endpoint in configuration.items():
            ep = self.EndPoint(model_config_blob='', **endpoint)
            ep.model_config_blob = self._task.get_configuration_object(
                name="model.{}".format(ep.serving_url))
            self._endpoints[ep.serving_url] = ep

        # get configuration of current served endpoints
        # noinspection PyProtectedMember
        self._current_serving_endpoints = self._task._get_configuration_dict(name='serving_state')

        self._dirty = True
        return True

    def update(self, force=False):
        # type: (bool) -> bool
        """
        Update internal endpoint state based on Task configuration and model repository

        :param force: if True force update

        :return: True if internal state updated.
        """
        if not self._task:
            return False

        # store current internal state
        state_hash = self.__state_hash()

        if not self._deserialize():
            return False

        # check if current internal state changed
        if not force and state_hash == self.__state_hash():
            print("Skipping update, nothing changed")
            return False

        return self.update_model_endpoint_state()

    def get_id(self):
        # type: () -> str
        """
        Return the Serving Service Task ID
        :return: Unique Task ID (str)
        """
        return self._task.id

    def get_engine_type(self):
        # type: () -> str
        """
        return the engine type used ib the serving service
        :return: engine type (str). example: triton, ovms, kfserving
        """
        return self._engine_type

    def serialize(self, force=False):
        # type: (bool) -> None
        """
        Serialize current service state to the Task

        :param force: If True synchronize an aborted/completed Task
        """
        if force and self._task.status not in (Task.TaskStatusEnum.created, Task.TaskStatusEnum.in_progress):
            self._task.mark_started(force=True)

        self._serialize()

    def triton_model_service_update_step(self, model_repository_folder=None, verbose=True):
        # type: (Optional[str], bool) -> None

        # check if something changed since last time
        if not self.update(force=self._last_update_step is None):
            return

        self._last_update_step = time()

        if not model_repository_folder:
            model_repository_folder = '/models/'

        if verbose:
            print('Updating local model folder: {}'.format(model_repository_folder))

        for url, endpoint in self.get_endpoints().items():
            folder = Path(model_repository_folder) / url
            folder.mkdir(parents=True, exist_ok=True)
            with open((folder / 'config.pbtxt').as_posix(), 'wt') as f:
                f.write(endpoint.model_config_blob)

            # download model versions
            for version, model_id in self.get_endpoint_version_model_id(serving_url=url).items():
                model_folder = folder / str(version)

                model_folder.mkdir(parents=True, exist_ok=True)
                model = None
                # noinspection PyBroadException
                try:
                    model = InputModel(model_id)
                    local_path = model.get_local_copy()
                except Exception:
                    local_path = None
                if not local_path:
                    print("Error retrieving model ID {} []".format(model_id, model.url if model else ''))
                    continue

                local_path = Path(local_path)

                if verbose:
                    print('Update model v{} in {}'.format(version, model_folder))

                # if this is a folder copy every and delete the temp folder
                if local_path.is_dir():
                    # we assume we have a `tensorflow.savedmodel` folder
                    model_folder /= 'model.savedmodel'
                    model_folder.mkdir(parents=True, exist_ok=True)
                    # rename to old
                    old_folder = None
                    if model_folder.exists():
                        old_folder = model_folder.parent / '.old.{}'.format(model_folder.name)
                        model_folder.replace(old_folder)
                    if verbose:
                        print('copy model into {}'.format(model_folder))
                    shutil.copytree(
                        local_path.as_posix(), model_folder.as_posix(), symlinks=False,
                    )
                    if old_folder:
                        shutil.rmtree(path=old_folder.as_posix())
                    # delete temp folder
                    shutil.rmtree(local_path.as_posix())
                else:
                    # single file should be moved
                    target_path = model_folder / local_path.name
                    old_file = None
                    if target_path.exists():
                        old_file = target_path.parent / '.old.{}'.format(target_path.name)
                        target_path.replace(old_file)
                    shutil.move(local_path.as_posix(), target_path.as_posix())
                    if old_file:
                        old_file.unlink()

    def __state_hash(self):
        # type: () -> int
        """
        Return Hash of the internal state (use only for in process comparison
        :return: hash int
        """
        return hash(json.dumps(
            [self._current_serving_endpoints, {k: v.as_dict() for k, v in self._endpoints.items()}],
            sort_keys=True))
