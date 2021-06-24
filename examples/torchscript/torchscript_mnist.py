# ClearML - Torchscript example code, automatic logging traced model
# Then store the torchscript model to be served by clearml-serving
import argparse
from pathlib import Path

import requests
import torch
from clearml import OutputModel, Task
from PIL import Image
from torch import nn
from torchvision import transforms


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_config_pbtxt(config_pbtxt_file):
    platform = "pytorch_libtorch"
    input_name = 'INPUT__0'
    output_name = 'OUTPUT__0'
    input_data_type = "TYPE_FP32"
    output_data_type = "TYPE_FP32"
    input_dims = str([-1, 3, 224, 224])
    output_dims = str([-1, 1000])

    config_pbtxt = """
        platform: "%s"
        input [
            {
                name: "%s"
                data_type: %s
                dims: %s
            }
        ]
        output [
            {
                name: "%s"
                data_type: %s
                dims: %s
            }
        ]
    """ % (
        platform,
        input_name, input_data_type, input_dims,
        output_name, output_data_type, output_dims
    )

    with open(config_pbtxt_file, "w") as config_file:
        config_file.write(config_pbtxt)


def preprocess(url):
    response = requests.get(url)
    filename = "sample.jpg"
    with open(filename, 'wb') as f:
        f.write(response.content)

    input_image = Image.open(filename)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def postprocess(output):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    return top5_prob, top5_catid


def get_mnist_labels():
    data = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    return data.text.split("\n")


def get_alexnet_state_dict():
    filename = "alexnet_weights.pt"
    if not Path(filename).exists():
        response = requests.get("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth")
        with open(filename, 'wb') as f:
            f.write(response.content)
    return torch.load(filename)


def main():
    parser = argparse.ArgumentParser(description='Torchscript MNIST Example - serving torchscript model')
    args = parser.parse_args()

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name='examples', task_name='Torchscript MNIST serve example', output_uri=True)

    # This could work, but the github api limits the number of downloads
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)

    # Instead we use hardcoded AlexNet model
    model = AlexNet()
    state_dict = get_alexnet_state_dict()
    model.load_state_dict(state_dict)

    model.eval()

    # Advanced: setting model class enumeration
    mnist_labels_list = get_mnist_labels()
    labels = {label: i for i, label in enumerate(mnist_labels_list)}
    task.set_model_label_enumeration(labels)

    # Get a input image for the model
    url = 'https://github.com/pytorch/hub/raw/master/images/dog.jpg'
    input = preprocess(url)

    # Trace and save the model in a format that can be served
    jit_model = torch.jit.trace(model, input)
    jit_model.save('serving_model')

    # Predict class using traced model on input
    output = jit_model(input)
    top5_prob, top5_catid = postprocess(output)

    for i in range(top5_prob.size(0)):
        print(mnist_labels_list[top5_catid[i]], top5_prob[i].item())

    # create the config.pbtxt for triton to be able to serve the model
    create_config_pbtxt(config_pbtxt_file='config.pbtxt')

    task.update_output_model(model_path='serving_model')

    # store the configuration on the creating Task,
    # this will allow us to skip over manually setting the config.pbtxt for `clearml-serving`
    task.connect_configuration(configuration=Path('config.pbtxt'), name='config.pbtxt')


if __name__ == '__main__':
    main()
