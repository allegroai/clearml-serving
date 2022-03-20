FROM python:3.9-bullseye


ENV LC_ALL=C.UTF-8

# install base package
RUN pip3 install clearml-serving

# get latest execution code from the git repository
# RUN cd $HOME && git clone https://github.com/allegroai/clearml-serving.git
COPY clearml_serving /root/clearml/clearml_serving

RUN pip3 install -r /root/clearml/clearml_serving/statistics/requirements.txt

# default serving port
EXPOSE 9999

# environement variable to load Task from CLEARML_SERVING_TASK_ID, CLEARML_SERVING_PORT

WORKDIR /root/clearml/
ENTRYPOINT ["clearml_serving/statistics/entrypoint.sh"]
