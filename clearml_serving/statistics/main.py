import os

import prometheus_client
from clearml import Task

from clearml_serving.serving.model_request_processor import ModelRequestProcessor
from clearml_serving.statistics.metrics import StatisticsController


def main():
    serving_service_task_id = os.environ.get("CLEARML_SERVING_TASK_ID", None)
    model_sync_frequency_secs = 5
    try:
        model_sync_frequency_secs = float(os.environ.get("CLEARML_SERVING_POLL_FREQ", model_sync_frequency_secs))
    except (ValueError, TypeError):
        pass

    # noinspection PyProtectedMember
    serving_task = ModelRequestProcessor._get_control_plane_task(task_id=serving_service_task_id)
    # create a new serving instance (for visibility and monitoring)
    instance_task = Task.init(
        project_name=serving_task.get_project_name(),
        task_name="{} - statistics controller".format(serving_task.name),
        task_type="monitor",
    )
    instance_task.set_system_tags(["service"])
    # noinspection PyProtectedMember
    kafka_server_url = os.environ.get("CLEARML_DEFAULT_KAFKA_SERVE_URL", "localhost:9092")
    stats_controller = StatisticsController(
        task=instance_task,
        kafka_server_url=kafka_server_url,
        serving_id=serving_service_task_id,
        poll_frequency_min=model_sync_frequency_secs
    )
    prometheus_client.start_http_server(int(os.environ.get("CLEARML_SERVING_PORT", 9999)))
    # we will never leave here
    stats_controller.start()


if __name__ == '__main__':
    main()
