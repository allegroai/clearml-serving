from time import sleep
from clearml import Task
from clearml_serving.serving_service import ServingService


def main():
    # we should only be running in remotely by an agent
    task = Task.init()
    serving = ServingService(task=task)
    while True:
        serving.update()
        serving.stats()
        sleep(60.)


if __name__ == '__main__':
    main()
