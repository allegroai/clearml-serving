import re
import subprocess
from argparse import ArgumentParser
from time import time
from typing import Optional

from pathlib2 import Path

from clearml import Task, Logger
from clearml.backend_api.utils import get_http_session_with_retry
from clearml_serving.serving_service import ServingService


class TritonHelper(object):
    _metric_line_parsing = r"(\w+){(gpu_uuid=\"[\w\W]*\",)?model=\"(\w+)\",\s*version=\"(\d+)\"}\s*([0-9.]*)"
    _default_metrics_port = 8002

    def __init__(
            self,
            args,  # Any
            task,  # type: Task
            serving_id,  # type: str
            metric_host=None,  # type: Optional[str]
            metric_port=None,  # type: int
    ):
        # type: (...) -> None
        self._http_session = get_http_session_with_retry()
        self.args = dict(**args.__dict__) if args else {}
        self.task = task
        self.serving_id = serving_id
        self.metric_host = metric_host or '0.0.0.0'
        self.metric_port = metric_port or self._default_metrics_port
        self._parse_metric = re.compile(self._metric_line_parsing)
        self._timestamp = time()
        print('String Triton Helper service\n{}\n'.format(self.args))

    def report_metrics(self, remote_logger):
        # type: (Optional[Logger]) -> bool
        # iterations are seconds from start
        iteration = int(time() - self._timestamp)

        report_msg = "reporting metrics: relative time {} sec".format(iteration)
        self.task.get_logger().report_text(report_msg)
        if remote_logger:
            remote_logger.report_text(report_msg)

        # noinspection PyBroadException
        try:
            request = self._http_session.get('http://{}:{}/metrics'.format(
                self.metric_host, self.metric_port))
            if not request.ok:
                return False
            content = request.content.decode().split('\n')
        except Exception:
            return False

        for line in content:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # noinspection PyBroadException
            try:
                metric, gpu_uuid, variant, version, value = self._parse_metric.match(line).groups()
                value = float(value)
            except Exception:
                continue
            self.task.get_logger().report_scalar(
                title=metric,
                series='{}.v{}'.format(variant, version),
                iteration=iteration,
                value=value
            )
            # on the remote logger we add our own Task ID (unique ID),
            # to support multiple servers reporting to the same service controller
            if remote_logger:
                remote_logger.report_scalar(
                    title=metric,
                    series='{}.v{}.{}'.format(variant, version, self.task.id),
                    iteration=iteration,
                    value=value
                )

    def maintenance_daemon(
            self,
            local_model_repo='/models',  # type: str
            update_frequency_sec=60.0,  # type: float
            metric_frequency_sec=60.0  # type: float
    ):
        # type: (...) -> None

        Path(local_model_repo).mkdir(parents=True, exist_ok=True)

        a_service = ServingService(task_id=self.serving_id)
        a_service.triton_model_service_update_step(model_repository_folder=local_model_repo)

        # noinspection PyProtectedMember
        remote_logger = a_service._task.get_logger()

        # todo: log triton server outputs when running locally

        # we assume we can run the triton server
        cmd = [
            'tritonserver',
            '--model-control-mode=poll',
            '--model-repository={}'.format(local_model_repo),
            '--repository-poll-secs={}'.format(update_frequency_sec),
            '--metrics-port={}'.format(self._default_metrics_port),
            '--allow-metrics=true',
            '--allow-gpu-metrics=true',
        ]
        for k, v in self.args.items():
            if not v or not str(k).startswith('t_'):
                continue
            cmd.append('--{}={}'.format(k, v))

        print('Starting server: {}'.format(cmd))
        try:
            proc = subprocess.Popen(cmd)
        except FileNotFoundError:
            raise ValueError(
                "Triton Server Engine (tritonserver) could not be found!\n"
                "Verify you running inside the `nvcr.io/nvidia/tritonserver` docker container")
        base_freq = min(update_frequency_sec, metric_frequency_sec)
        metric_tic = update_tic = time()
        while True:
            try:
                error_code = proc.wait(timeout=base_freq)
                if error_code == 0:
                    print("triton-server process ended with error code {}".format(error_code))
                    return
                raise ValueError("triton-server process ended with error code {}".format(error_code))
            except subprocess.TimeoutExpired:
                pass
            pass

            # update models
            if time() - update_tic > update_frequency_sec:
                a_service.triton_model_service_update_step(model_repository_folder=local_model_repo)
                update_tic = time()

            # update stats
            if time() - metric_tic > metric_frequency_sec:
                metric_tic = time()
                self.report_metrics(remote_logger)


def main():
    title = 'clearml-serving - Nvidia Triton Engine Helper'
    print(title)
    parser = ArgumentParser(prog='clearml-serving', description=title)
    parser.add_argument(
        '--serving-id', default=None, type=str, required=True,
        help='Specify main serving service Task ID')
    parser.add_argument(
        '--project', default='serving', type=str,
        help='Optional specify project for the serving engine Task')
    parser.add_argument(
        '--name', default='nvidia-triton', type=str,
        help='Optional specify task name for the serving engine Task')
    parser.add_argument(
        '--update-frequency', default=10, type=float,
        help='Model update frequency in minutes')
    parser.add_argument(
        '--metric-frequency', default=1, type=float,
        help='Metric reporting update frequency in minutes')
    parser.add_argument(
        '--t-http-port', type=str, help='<integer> The port for the server to listen on for HTTP requests')
    parser.add_argument(
        '--t-http-thread-count', type=str, help='<integer> Number of threads handling HTTP requests')
    parser.add_argument(
        '--t-allow-grpc', type=str, help='<integer> Allow the server to listen for GRPC requests')
    parser.add_argument(
        '--t-grpc-port', type=str, help='<integer> The port for the server to listen on for GRPC requests')
    parser.add_argument(
        '--t-grpc-infer-allocation-pool-size', type=str,
        help='<integer> The maximum number of inference request/response objects that remain '
             'allocated for reuse. As long as the number of in-flight requests doesn\'t exceed '
             'this value there will be no allocation/deallocation of request/response objects')
    parser.add_argument(
        '--t-pinned-memory-pool-byte-size', type=str,
        help='<integer> The total byte size that can be allocated as pinned system '
             'memory. If GPU support is enabled, the server will allocate pinned '
             'system memory to accelerate data transfer between host and devices '
             'until it exceeds the specified byte size. This option will not affect '
             'the allocation conducted by the backend frameworks. Default is 256 MB')
    parser.add_argument(
        '--t-cuda-memory-pool-byte-size', type=str,
        help='<<integer>:<integer>> The total byte size that can be allocated as CUDA memory for '
             'the GPU device. If GPU support is enabled, the server will allocate '
             'CUDA memory to minimize data transfer between host and devices '
             'until it exceeds the specified byte size. This option will not affect '
             'the allocation conducted by the backend frameworks. The argument '
             'should be 2 integers separated by colons in the format <GPU device'
             'ID>:<pool byte size>. This option can be used multiple times, but only '
             'once per GPU device. Subsequent uses will overwrite previous uses for '
             'the same GPU device. Default is 64 MB')
    parser.add_argument(
        '--t-min-supported-compute-capability', type=str,
        help='<float> The minimum supported CUDA compute capability. GPUs that '
             'don\'t support this compute capability will not be used by the server')
    parser.add_argument(
        '--t-buffer-manager-thread-count', type=str,
        help='<integer> The number of threads used to accelerate copies and other'
             'operations required to manage input and output tensor contents.'
             'Default is 0')

    args = parser.parse_args()
    task = Task.init(project_name=args.project, task_name=args.name, task_type=Task.TaskTypes.inference)
    helper = TritonHelper(args, task, serving_id=args.serving_id)
    # this function will never end
    helper.maintenance_daemon(
        local_model_repo='/models',
        update_frequency_sec=args.update_frequency*60.0,
        metric_frequency_sec=args.metric_frequency*60.0,
    )


if __name__ == '__main__':
    main()
