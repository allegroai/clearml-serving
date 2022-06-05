#!/bin/bash

# print configuration
echo CLEARML_SERVING_TASK_ID="$CLEARML_SERVING_TASK_ID"
echo CLEARML_TRITON_POLL_FREQ="$CLEARML_TRITON_POLL_FREQ"
echo CLEARML_TRITON_METRIC_FREQ="$CLEARML_TRITON_METRIC_FREQ"
echo CLEARML_TRITON_HELPER_ARGS="$CLEARML_TRITON_HELPER_ARGS"
echo CLEARML_EXTRA_PYTHON_PACKAGES="$CLEARML_EXTRA_PYTHON_PACKAGES"

# we should also have clearml-server configurations

if [ ! -z "$CLEARML_EXTRA_PYTHON_PACKAGES" ]
then
      python3 -m pip install $CLEARML_EXTRA_PYTHON_PACKAGES
fi

# start service
PYTHONPATH=$(pwd) python3 clearml_serving/engines/triton/triton_helper.py $CLEARML_TRITON_HELPER_ARGS $@
