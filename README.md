
<div align="center">

<a href="https://app.community.clear.ml"><img src="https://github.com/allegroai/clearml/blob/master/docs/clearml-logo.svg?raw=true" width="250px"></a>

**ClearML Serving - Model deployment made easy**

## **`clearml-serving v2.0` </br> :sparkles: Model Serving (ML/DL) Made Easy :tada:**


[![GitHub license](https://img.shields.io/github/license/allegroai/clearml-serving.svg)](https://img.shields.io/github/license/allegroai/clearml-serving.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/clearml-serving.svg)](https://img.shields.io/pypi/pyversions/clearml-serving.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/clearml-serving.svg)](https://img.shields.io/pypi/v/clearml-serving.svg)
[![PyPI status](https://img.shields.io/pypi/status/clearml-serving.svg)](https://pypi.python.org/pypi/clearml-serving/)
[![Slack Channel](https://img.shields.io/badge/slack-%23clearml--community-blueviolet?logo=slack)](https://join.slack.com/t/allegroai-trains/shared_invite/zt-c0t13pty-aVUZZW1TSSSg2vyIGVPBhg)


</div>


**`clearml-serving`** is a command line utility for model deployment and orchestration.  
It enables model deployment including serving and preprocessing code to a Kubernetes cluster or custom container based solution.

Features:
* Easy to deploy & configure
  * Support Machine Learning Models (Scikit Learn, XGBoost, LightGBM)
  * Support Deep Learning Models (Tensorflow, PyTorch, ONNX)
  * Customizable RestAPI for serving (i.e. allow per model pre/post-processing for easy integration)
* Flexible  
  * On-line model deployment 
  * On-line endpoint model/version deployment (i.e. no need to take the service down)
  * Per model standalone preprocessing and postprocessing python code 
* Scalable
  * Multi model per container
  * Multi models per serving service
  * Multi-service support (fully seperated multiple serving service running independently)
  * Multi cluster support
  * Out-of-the-box node auto-scaling based on load/usage
* Efficient
  * multi-container resource utilization
  * Support for CPU & GPU nodes
  * Auto-batching for DL models
* Automatic deployment
  * Automatic model upgrades w/ canary support 
  * Programmable API for model deployment
* Canary A/B deployment
  * Online Canary updates
* Model Monitoring
  * Usage Metric reporting
  * Metric Dashboard
  * Model performance metric
  * Model performance Dashboard

## ClearML Serving Design 

### ClearML Serving Design Principles 

**Modular** , **Scalable** , **Flexible** , **Customizable** , **Open Source**

<a href="https://excalidraw.com/#json=v0ip945hun2SnO4HVLe0h,QKHfB04TFQLds3_4aqeBjQ"><img src="https://github.com/allegroai/clearml-serving/blob/dev/docs/design_diagram.png?raw=true" width="100%"></a>

## Installation

### :information_desk_person: Concepts

**CLI** - Secure configuration interface for on-line model upgrade/deployment on running Serving Services

**Serving Service Task** - Control plane object storing configuration on all the endpoints. Support multiple separated instance, deployed on multiple clusters.

**Inference Services** - Inference containers, performing model serving pre/post processing. Also support CPU model inferencing.

**Serving Engine Services** - Inference engine containers (e.g. Nvidia Triton, TorchServe etc.) used by the Inference Services for heavier model inference.

**Statistics Service** - Single instance per Serving Service  collecting and broadcasting model serving & performance statistics

**Time-series DB** - Statistics collection service used by the Statistics Service, e.g. Prometheus

**Dashboards** - Customizable dashboard-ing solution on top of the collected statistics, e.g. Grafana

### prerequisites

* ClearML-Server : Model repository, Service Health, Control plane
* Kubernetes / Single-instance VM : Deploying containers 
* CLI : Configuration & model deployment interface


### :nail_care: Initial Setup

1. Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.community.clear.ml)
2. Install the CLI on your  laptop `clearml` and `clearml-serving`
   - `pip3 install https://github.com/allegroai/clearml-serving.git@dev`
   - Make sure to configure your machine to connect to your `clearml-server` see [clearml-init](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml) for details
3. Create the Serving Service Controller
  - `clearml-serving create --name "serving example"`
  - The new serving service UID should be printed `"New Serving Service created: id=aa11bb22aa11bb22`
4. Write down the Serving Service UID

### :point_right: Toy model (scikit learn) deployment example 

1. Train toy scikit-learn model
  - create new python virtual environment
  - `pip3 install -r examples/sklearn/requirements.txt`
  - `python3 examples/sklearn/train_model.py`
  - Model was automatically registered and uploaded into the model repository. For Manual model registration see [here](#registering--deploying-new-models-manually) 
2. Register the new Model on the Serving Service
  - `clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model" --project "serving examples"`
  - **Notice** the preprocessing python code is packaged and uploaded to the "Serving Service", to be used by any inference container, and downloaded in realtime when updated
3. Spin the Inference Container
  - Customize container [Dockerfile](clearml_serving/serving/Dockerfile) if needed
  - Build container `docker build --tag clearml-serving-inference:latest -f clearml_serving/serving/Dockerfile .`
  - Spin the inference container: `docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> -e CLEARML_SERVING_POLL_FREQ=5 clearml-serving-inference:latest` 
4. Test new model inference endpoint
  - `curl -X POST "http://127.0.0.1:8080/serve/test_model_sklearn" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x0": 1, "x1": 2}'`
  
**Notice**, now that we have an inference container running, we can add new model inference endpoints directly with the CLI. The inference container will automatically sync once every 5 minutes.

**Notice** On the first few requests the inference container needs to download the model file and preprocessing python code, this means the request might take a little longer, once everything is cached, it will return almost immediately.

**Notes:**
> To review the model repository in the ClearML web UI, under the "serving examples" Project on your ClearML account/server ([free hosted](https://app.clear.ml) or [self-deployed](https://github.com/allegroai/clearml-server)).

> Inference services status, console outputs and machine metrics are available in the ClearML UI in the Serving Service project (default: "DevOps" project)

> To learn more on training models and the ClearML model repository, see the [ClearML documentation](https://clear.ml/docs)


### :muscle: Nvidia Triton serving engine setup

Nvidia Triton Serving Engine is used by clearml-serving to do the heavy lifting of deep-learning models on both GPU & CPU nodes. 
Inside the Triton container a clearml controller is spinning and monitoring the Triton server.
All the triton models are automatically downloaded into the triton container in real-time, configured, and served.
A single Triton serving container is serving multiple models, based on the registered models on the Serving Service 
Communication from the Inference container to the Triton container is done transparently over compressed gRPC channel.

#### setup

Optional: build the Triton container 
  - Customize container [Dockerfile](clearml_serving/engines/triton/Dockerfile)
  - Build container `docker build --tag clearml-serving-triton:latest -f clearml_serving/engines/triton/Dockerfile .`

Spin the triton engine container: `docker run -v ~/clearml.conf:/root/clearml.conf -p 8001:8001 -e CLEARML_SERVING_TASK_ID=<service_id> -e CLEARML_TRITON_POLL_FREQ=5 -e CLEARML_TRITON_METRIC_FREQ=1 clearml-serving-triton:latest`

Configure the "Serving Service" with the new Triton Engine gRPC IP:Port. Notice that when deploying on a Kubernetes cluster this should be a TCP ingest endpoint, to allow for transparent auto-scaling of the Triton Engine Containers

`clearml-serving --id <service_id> config --triton-grpc-server <local_ip_here>:8001`

Spin the inference service (this is the external RestAPI interface)
`docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> -e CLEARML_SERVING_POLL_FREQ=5 clearml-serving-inference:latest`

Now eny model that will register with "Triton" engine, will run the pre/post processing code on the Inference service container, and the model inference itself will be executed on the Triton Engine container.
See Tensorflow [example](examples/keras/readme.md) and Pytorch [example](examples/pytorch/readme.md) for further details.


### :ocean: Container Configuration Variables 

When spinning the Inference container or the Triton Engine container, 
we need to specify the `clearml-server` address and access credentials
One way of achieving that is by mounting the `clearml.conf` file into the container's HOME folder (i.e. `-v ~/clearml.conf:/root/clearml.conf`) 
We can also pass environment variables instead (see [details](https://clear.ml/docs/latest/docs/configs/env_vars#server-connection):
```bash
CLEARML_API_HOST="https://api.clear.ml"
CLEARML_WEB_HOST="https://app.clear.ml"
CLEARML_FILES_HOST="https://files.clear.ml"
CLEARML_API_ACCESS_KEY="access_key_here"
CLEARML_API_SECRET_KEY="secret_key_here"
```

To access models stored on an S3 buckets, Google Storage or Azure blob storage (notice that with GS you also need to make sure the access json is available inside the containers). See further details on configuring the storage access [here](https://clear.ml/docs/latest/docs/integrations/storage#configuring-storage)

```bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION

GOOGLE_APPLICATION_CREDENTIALS

AZURE_STORAGE_ACCOUNT
AZURE_STORAGE_KEY
```

### :turtle: Registering & Deploying new models manually 

Uploading an existing model file into the model repository can be done via the `clearml` RestAPI, the python interface, or with the `clearml-serving` CLI 

> To learn more on training models and the ClearML model repository, see the [ClearML documentation](https://clear.ml/docs)

- local model file on our laptop: 'examples/sklearn/sklearn-model.pkl'
- Upload the model file to the `clearml-server` file storage and register it
`clearml-serving --id <service_id> model upload --name "manual sklearn model" --project "serving examples" --framework "scikit-learn" --path examples/sklearn/sklearn-model.pkl`
- We now have a new Model in the "serving examples" project, by the name of "manual sklearn model". The CLI output prints the UID of the newly created model, we will use it to register a new endpoint 
- In the `clearml` web UI we can see the new model listed under the `Models` tab in the associated project. we can also download the model file itself directly from the web UI 
- Register a new endpoint with the new model
`clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn" --preprocess "examples/sklearn/preprocess.py" --model-id <newly_created_model_id_here>`

**Notice** we can also provide a differnt storage destination for the model, such as S3/GS/Azure, by passing
`--destination="s3://bucket/folder"`, `gs://bucket/folder`, `azure://bucket/folder`. Yhere is no need to provide a unique path tp the destination argument, the location of the model will be a unique path based on the serving service ID and the model name


### :rabbit: Automatic model deployment

The clearml Serving Service support automatic model deployment and upgrades, directly connected with the model repository and API. When the model auto-deploy is configured, a new model versions will be automatically deployed when you "publish" or "tag" a new model in the `clearml` model repository. This automation interface allows for simpler CI/CD model deployment process, as a single API automatically deploy (or remove) a model from the Serving Service.

#### automatic model deployment example

1. Configure the model auto-update on the Serving Service
- `clearml-serving --id <service_id> model auto-update --engine sklearn --endpoint "test_model_sklearn_auto" --preprocess "preprocess.py" --name "train sklearn model" --project "serving examples" --max-versions 2`
2. Deploy the Inference container (if not already deployed)
3. Publish a new model the model repository
- Go to the "serving examples" project in the ClearML web UI, click on the Models Tab, search for "train sklearn model" right click and select "Publish"
- Use the RestAPI [details](https://clear.ml/docs/latest/docs/references/api/models#post-modelspublish_many)
- Use Python interface: 
```python
from clearml import Model
Model(model_id="unique_model_id_here").publish()
```
4. The new model is available on a new endpoint version (1), test with: 
`curl -X POST "http://127.0.0.1:8080/serve/test_model_sklearn_auto/1" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x0": 1, "x1": 2}'`

### :bird: Canary endpoint setup

Canary endpoint deployment add a new endpoint where the actual request is sent to a preconfigured set of endpoints with pre-provided distribution. For example, let's create a new endpoint "test_model_sklearn_canary", we can provide a list of endpoints and probabilities (weights).

```bash
clearml-serving --id <service_id> model canary --endpoint "test_model_sklearn_canary" --weights 0.1 0.9 --input-endpoints test_model_sklearn/2 test_model_sklearn/1
```
This means that any request coming to `/test_model_sklearn_canary/` will be routed with probability of 90% to
`/test_model_sklearn/1/` and with probability of 10% to `/test_model_sklearn/2/` 

**Note:**
> As with any other Serving Service configuration, we can configure the Canary endpoint while the Inference containers are already running and deployed, they will get updated in their next update cycle (default: once every 5 minutes)

We Can also prepare a "fixed" canary endpoint, always splitting the load between the last two deployed models:
```bash
clearml-serving --id <service_id> model canary --endpoint "test_model_sklearn_canary" --weights 0.1 0.9 --input-endpoints-prefix test_model_sklearn/
```

This means that is we have two model inference endpoints: `/test_model_sklearn/1/`, `/test_model_sklearn/2/`  
the 10% probability (weight 0.1) will match the last (order by version number) endpoint, i.e. `/test_model_sklearn/2/` and the 90% will match `/test_model_sklearn/2/`
When we add a new model endpoint version, e.g. `/test_model_sklearn/3/`, the canary distribution will automatically match the 90% probability to `/test_model_sklearn/2/` and the 10% to the new endpoint `/test_model_sklearn/3/`  

Example:
1. Add two endpoints:
  - `clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model" --version 1 --project "serving examples"`
  -  `clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model" --version 2 --project "serving examples"`
2. Add Canary endpoint:
  - `clearml-serving --id <service_id> model canary --endpoint "test_model_sklearn_canary" --weights 0.1 0.9 --input-endpoints test_model_sklearn/2 test_model_sklearn/1`
3. Test Canary endpoint:
  - `curl -X POST "http://127.0.0.1:8080/serve/test_model" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x0": 1, "x1": 2}'` 


### :fire: Model Serving Examples

- Scikit-Learn [example](examples/sklearn/readme.md) - random data 
- XGBoost [example](examples/xgboost/readme.md) - iris dataset
- LightGBM [example](examples/lightgbm/readme.md) - iris dataset
- PyTorch [example](examples/pytorch/readme.md) - mnist dataset
- TensorFlow/Keras [example](examples/keras/readme.md) - mnist dataset

### :pray: Status

  - [x] FastAPI integration for inference service
  - [x] multi-process Gunicorn for inference service
  - [x] Dynamic preprocess python code loading (no need for container/process restart)
  - [x] Model files download/caching (http/s3/gs/azure)
  - [x] Scikit-learn. XGBoost, LightGBM integration
  - [x] Custom inference, including dynamic code loading
  - [x] Manual model upload/registration to model repository (http/s3/gs/azure)
  - [x] Canary load balancing
  - [x] Auto model endpoint deployment based on model repository state
  - [x] Machine/Node health metrics
  - [x] Dynamic online configuration
  - [x] CLI configuration tool
  - [x] Nvidia Triton integration
  - [x] GZip request compression
  - [ ] TorchServe engine integration
  - [ ] Prebuilt Docker containers (dockerhub)
  - [x] Scikit-Learn example
  - [x] XGBoost example
  - [x] LightGBM example
  - [x] PyTorch example
  - [x] TensorFlow/Keras example
  - [ ] Model ensemble example
  - [ ] Model pipeline example
  - [ ] Statistics Service
  - [ ] Kafka install instructions
  - [ ] Prometheus install instructions
  - [ ] Grafana install instructions
  - [ ] Kubernetes Helm Chart

## Contributing

**PRs are always welcomed** :heart: See more details in the ClearML [Guidelines for Contributing](https://github.com/allegroai/clearml/blob/master/docs/contributing.md).


