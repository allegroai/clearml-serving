#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from PIL import Image
import numpy as np

from http_triton import InferenceServerClient, InferInput

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default localhost:8000')

    FLAGS = parser.parse_args()

    model_name = "keras_mnist"
    model_version = "1"

    input_name = "dense_input"
    shape = (1, 784)
    datatype = 'FP32'

    output_name = 'activation_2'

    # Path of an image
    image_path = '68747470733a2f2f646174616d61646e6573732e6769746875622e696f2f6173736574732f696d616765732f74665f66696c655f666565642f4d4e4953545f64696769742e706e67.png'

    # The image is opened using Pillow, then converted to grayscale since the model deployed is trained on grayscale images
    image = Image.open(image_path).convert('L')

    # The image is resized to 28x28 pixels
    image = image.resize(shape, Image.ANTIALIAS)

    # The image is converted to a numpy array and data type is converted to float32 since the model is trained on float32
    np_image = np.array(image).astype(np.float32)

    # The image is reshaped to fit the model
    np_image = np_image.reshape(-1, 784)

    # Create an InferInput object with the input name, its data type and its shape defined
    inferInput = InferInput(name=input_name, datatype=datatype, shape=shape)

    # Set the data inside the InferInput object to the image in numpy format
    inferInput.set_data_from_numpy(np_image)

    # Create an InferenceServerClient and pass to it the url of the server
    client = InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)

    # Call client.infer(), pass the model name, version and the InferInput object inside a list since there can be multiple inputs
    inferResult = client.infer(model_name=model_name, inputs=[inferInput], model_version=model_version)

    # Print the output of the model in numpy format, pass in the name of the output layer
    print(inferResult.as_numpy(output_name))