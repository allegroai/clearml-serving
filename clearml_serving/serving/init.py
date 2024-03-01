import os
from clearml import Task
from clearml_serving.serving.model_request_processor import ModelRequestProcessor
from clearml_serving.serving.preprocess_service import BasePreprocessRequest


def setup_task(force_threaded_logging=None):
    serving_service_task_id = os.environ.get("CLEARML_SERVING_TASK_ID", None)

    # always use background thread, it requires less memory
    if force_threaded_logging or os.environ.get("CLEARML_BKG_THREAD_REPORT") in ("1", "Y", "y", "true"):
        os.environ["CLEARML_BKG_THREAD_REPORT"] = "1"
        Task._report_subprocess_enabled = False

    # get the serving controller task
    # noinspection PyProtectedMember
    serving_task = ModelRequestProcessor._get_control_plane_task(task_id=serving_service_task_id)
    # set to running (because we are here)
    if serving_task.status != "in_progress":
        serving_task.started(force=True)

    # create a new serving instance (for visibility and monitoring)
    instance_task = Task.init(
        project_name=serving_task.get_project_name(),
        task_name="{} - serve instance".format(serving_task.name),
        task_type="inference",  # noqa
    )
    instance_task.set_system_tags(["service"])
    # make sure we start logging thread/process
    instance_logger = instance_task.get_logger()  # noqa
    # this will use the main thread/process
    session_logger = serving_task.get_logger()

    # preload modules into memory before forking
    BasePreprocessRequest.load_modules()

    return serving_service_task_id, session_logger, instance_task.id
