import json
import os
import re
from copy import deepcopy
from functools import partial
from threading import Event, Thread
from time import time, sleep

from clearml import Task
from typing import Optional, Dict, Any, Iterable, Set

from prometheus_client import Histogram, Enum, Gauge, Counter, values
from kafka import KafkaConsumer
from prometheus_client.metrics import MetricWrapperBase, _validate_exemplar
from prometheus_client.registry import REGISTRY
from prometheus_client.samples import Exemplar, Sample
from prometheus_client.context_managers import Timer
from prometheus_client.utils import floatToGoString

from ..serving.endpoints import EndpointMetricLogging
from ..serving.model_request_processor import ModelRequestProcessor


class ScalarHistogram(Histogram):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def observe(self, amount, exemplar=None):
        """Observe the given amount.

        The amount is usually positive or zero. Negative values are
        accepted but prevent current versions of Prometheus from
        properly detecting counter resets in the sum of
        observations. See
        https://prometheus.io/docs/practices/histograms/#count-and-sum-of-observations
        for details.
        """
        self._raise_if_not_observable()
        if not isinstance(amount, (list, tuple)):
            amount = [amount]
        self._sum.inc(len(amount))
        for v in amount:
            for i, bound in enumerate(self._upper_bounds):
                if v <= bound:
                    self._buckets[i].inc(1)
                    if exemplar:
                        _validate_exemplar(exemplar)
                        self._buckets[i].set_exemplar(Exemplar(exemplar, v, time()))
                    break

    def _child_samples(self) -> Iterable[Sample]:
        samples = []
        for i, bound in enumerate(self._upper_bounds):
            acc = self._buckets[i].get()
            samples.append(
                Sample('_bucket', {'le': floatToGoString(bound)}, acc, None, self._buckets[i].get_exemplar())
            )
            samples.append(Sample('_sum', {'le': floatToGoString(bound)}, self._sum.get(), None, None))

        return tuple(samples)


class EnumHistogram(MetricWrapperBase):
    """A Histogram tracks the size and number of events in buckets.

    You can use Histograms for aggregatable calculation of quantiles.

    Example use cases:
    - Response latency
    - Request size

    Example for a Histogram:

        from prometheus_client import Histogram

        h = Histogram('request_size_bytes', 'Request size (bytes)')
        h.observe(512)  # Observe 512 (bytes)

    Example for a Histogram using time:

        from prometheus_client import Histogram

        REQUEST_TIME = Histogram('response_latency_seconds', 'Response latency (seconds)')

        @REQUEST_TIME.time()
        def create_response(request):
          '''A dummy function'''
          time.sleep(1)

    Example of using the same Histogram object as a context manager:

        with REQUEST_TIME.time():
            pass  # Logic to be timed

    The default buckets are intended to cover a typical web/rpc request from milliseconds to seconds.
    They can be overridden by passing `buckets` keyword argument to `Histogram`.
    """
    _type = 'histogram'

    def __init__(self,
                 name,
                 documentation,
                 buckets,
                 labelnames=(),
                 namespace='',
                 subsystem='',
                 unit='',
                 registry=REGISTRY,
                 _labelvalues=None,
                 ):
        self._prepare_buckets(buckets)
        super().__init__(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=registry,
            _labelvalues=_labelvalues,
        )
        self._kwargs['buckets'] = buckets

    def _prepare_buckets(self, buckets):
        buckets = [str(b) for b in buckets]
        if buckets != sorted(buckets):
            # This is probably an error on the part of the user,
            # so raise rather than sorting for them.
            raise ValueError('Buckets not in sorted order')

        if len(buckets) < 2:
            raise ValueError('Must have at least two buckets')
        self._upper_bounds = buckets

    def _metric_init(self):
        self._buckets = {}
        self._created = time()
        bucket_labelnames = self._upper_bounds
        self._sum = values.ValueClass(
            self._type, self._name, self._name + '_sum', self._labelnames, self._labelvalues)
        for b in self._upper_bounds:
            self._buckets[b] = values.ValueClass(
                self._type,
                self._name,
                self._name + '_bucket',
                bucket_labelnames,
                self._labelvalues + (b,))

    def observe(self, amount, exemplar=None):
        """Observe the given amount.

        The amount is usually positive or zero. Negative values are
        accepted but prevent current versions of Prometheus from
        properly detecting counter resets in the sum of
        observations. See
        https://prometheus.io/docs/practices/histograms/#count-and-sum-of-observations
        for details.
        """
        self._raise_if_not_observable()
        if not isinstance(amount, (list, tuple)):
            amount = [amount]
        self._sum.inc(len(amount))
        for v in amount:
            self._buckets[v].inc(1)
            if exemplar:
                _validate_exemplar(exemplar)
                self._buckets[v].set_exemplar(Exemplar(exemplar, 1, time()))

    def time(self):
        """Time a block of code or function, and observe the duration in seconds.

        Can be used as a function decorator or context manager.
        """
        return Timer(self, 'observe')

    def _child_samples(self) -> Iterable[Sample]:
        samples = []
        for i in self._buckets:
            acc = self._buckets[i].get()
            samples.append(Sample(
                '_bucket', {'enum': i}, acc, None, self._buckets[i].get_exemplar()))
            samples.append(Sample('_sum', {'enum': i}, self._sum.get(), None, None))

        return tuple(samples)


class StatisticsController(object):
    _reserved = {
        '_latency': partial(ScalarHistogram, buckets=(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0)),
        '_count': Counter
    }
    _metric_type_class = {"scalar": ScalarHistogram, "enum": EnumHistogram, "value": Gauge, "counter": Counter}

    def __init__(
            self,
            task: Task,
            kafka_server_url: str,
            serving_id: Optional[str],
            poll_frequency_min: float = 5
    ):
        self.task = task
        self._serving_service_task_id = serving_id
        self._poll_frequency_min = float(poll_frequency_min)
        self._serving_service = None  # type: Optional[ModelRequestProcessor]
        self._current_endpoints = {}  # type: Optional[Dict[str, EndpointMetricLogging]]
        self._auto_added_endpoints = set()  # type: Set[str]
        self._prometheus_metrics = {}  # type: Optional[Dict[str, Dict[str, MetricWrapperBase]]]
        self._timestamp = time()
        self._sync_thread = None
        self._last_sync_time = time()
        self._dirty = False
        self._sync_event = Event()
        self._sync_threshold_sec = 30
        self._kafka_server = kafka_server_url
        # noinspection PyProtectedMember
        self._kafka_topic = ModelRequestProcessor._kafka_topic

    def start(self):
        self._serving_service = ModelRequestProcessor(task_id=self._serving_service_task_id)

        if not self._sync_thread:
            self._sync_thread = Thread(target=self._sync_daemon, daemon=True)
            self._sync_thread.start()

        # noinspection PyProtectedMember
        kafka_server = \
            self._serving_service.get_configuration().get(ModelRequestProcessor._config_key_kafka_stats) or \
            self._kafka_server

        print("Starting Kafka Statistics processing: {}".format(kafka_server))

        while True:
            try:
                consumer = KafkaConsumer(self._kafka_topic, bootstrap_servers=kafka_server)
                break
            except Exception as ex:
                print("Error: failed opening Kafka consumer [{}]: {}".format(kafka_server, ex))
                print("Retrying in 30 seconds")
                sleep(30)

        # we will never leave this loop
        for message in consumer:
            # noinspection PyBroadException
            try:
                list_data = json.loads(message.value.decode("utf-8"))
            except Exception:
                print("Warning: failed to decode kafka stats message")
                continue
            for data in list_data:
                try:
                    url = data.pop("_url", None)
                    if not url:
                        # should not happen
                        continue
                    endpoint_metric = self._current_endpoints.get(url)
                    if not endpoint_metric:
                        # add default one, we will just log the reserved valued:
                        endpoint_metric = dict()
                        self._current_endpoints[url] = EndpointMetricLogging(endpoint=url)
                        self._auto_added_endpoints.add(url)
                        # we should sync,
                        if time()-self._last_sync_time > self._sync_threshold_sec:
                            self._last_sync_time = time()
                            self._sync_event.set()

                    metric_url_log = self._prometheus_metrics.get(url)
                    if not metric_url_log:
                        # create a new one
                        metric_url_log = dict()
                        self._prometheus_metrics[url] = metric_url_log

                    # check if we have the prometheus_logger
                    for k, v in data.items():
                        prometheus_logger = metric_url_log.get(k)
                        if not prometheus_logger:
                            prometheus_logger = self._create_prometheus_logger_class(url, k, endpoint_metric)
                            if not prometheus_logger:
                                continue
                            metric_url_log[k] = prometheus_logger

                        self._report_value(prometheus_logger, v)

                except Exception as ex:
                    print("Warning: failed to report stat to Prometheus: {}".format(ex))
                    continue

    @staticmethod
    def _report_value(prometheus_logger: Optional[MetricWrapperBase], v: Any) -> bool:
        if not prometheus_logger:
            # this means no one configured the variable to log
            return False
        elif isinstance(prometheus_logger, (Histogram, EnumHistogram)):
            prometheus_logger.observe(amount=v)
        elif isinstance(prometheus_logger, Gauge):
            prometheus_logger.set(value=v)
        elif isinstance(prometheus_logger, Counter):
            prometheus_logger.inc(amount=v)
        elif isinstance(prometheus_logger, Enum):
            prometheus_logger.state(state=v)
        else:
            # we should not get here
            return False

        return True

    def _create_prometheus_logger_class(
            self,
            url: str,
            variable_name: str,
            endpoint_config: EndpointMetricLogging
    ) -> Optional[MetricWrapperBase]:
        reserved_cls = self._reserved.get(variable_name)
        name = "{}:{}".format(url, variable_name)
        name = re.sub(r"[^(a-zA-Z0-9_:)]", "_", name)
        if reserved_cls:
            return reserved_cls(name=name, documentation="Built in {}".format(variable_name))

        if not endpoint_config:
            # we should not end up here
            return None

        metric_ = endpoint_config.metrics.get(variable_name)
        if not metric_:
            return None
        metric_cls = self._metric_type_class.get(metric_.type)
        if not metric_cls:
            return None
        if metric_cls in (ScalarHistogram, EnumHistogram):
            return metric_cls(
                name=name,
                documentation="User defined metric {}".format(metric_.type),
                buckets=metric_.buckets
            )
        return metric_cls(name=name, documentation="User defined metric {}".format(metric_.type))

    def _sync_daemon(self):
        self._last_sync_time = time()
        poll_freq_sec = self._poll_frequency_min*60
        print("Instance [{}, pid={}]: Launching - configuration sync every {} sec".format(
            self.task.id, os.getpid(), poll_freq_sec))
        while True:
            try:
                self._serving_service.reload()
                endpoint_metrics = self._serving_service.list_endpoint_logging()
                self._last_sync_time = time()
                # we might have added new urls (auto metric logging), we need to compare only configured keys
                current_endpoints = {
                    k: v for k, v in self._current_endpoints.items()
                    if k not in self._auto_added_endpoints}
                if current_endpoints == endpoint_metrics:
                    self._sync_event.wait(timeout=poll_freq_sec)
                    self._sync_event.clear()
                    continue

                # update metrics:
                self._dirty = True
                self._auto_added_endpoints -= set(endpoint_metrics.keys())
                # merge top level configuration (we might have auto logged url endpoints)
                self._current_endpoints.update(deepcopy(endpoint_metrics))
                print("New configuration synced")
            except Exception as ex:
                print("Warning: failed to sync state from serving service Task: {}".format(ex))
                continue
