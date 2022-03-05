#!/bin/bash

# print configuration
echo CLEARML_SERVING_TASK_ID="$CLEARML_SERVING_TASK_ID"
echo CLEARML_SERVING_PORT="$CLEARML_SERVING_PORT"
echo EXTRA_PYTHON_PACKAGES="$EXTRA_PYTHON_PACKAGES"
echo CLEARML_SERVING_NUM_PROCESS="$CLEARML_SERVING_NUM_PROCESS"
echo CLEARML_SERVING_POLL_FREQ="$CLEARML_SERVING_POLL_FREQ"

GUNICORN_NUM_PROCESS="${CLEARML_SERVING_NUM_PROCESS:-4}"
GUNICORN_SERVING_PORT="${CLEARML_SERVING_PORT:-8080}"

echo GUNICORN_NUM_PROCESS="$GUNICORN_NUM_PROCESS"
echo GUNICORN_SERVING_PORT="$GUNICORN_SERVING_PORT"

# we should also have clearml-server configurations

if [ ! -z "$EXTRA_PYTHON_PACKAGES" ]
then
      python3 -m pip install $EXTRA_PYTHON_PACKAGES
fi

# start service
PYTHONPATH=$(pwd) python3 -m gunicorn \
    --preload clearml_serving.serving.main:app \
    --workers $GUNICORN_NUM_PROCESS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$GUNICORN_SERVING_PORT
