import argparse
import ctypes
import os

import numpy as np
import torch
import torch.nn as nn
import poptorch

# This simple example demonstrates compiling a model to add
# two tensors together using the IPU.


def load_custom_ops_lib():
    so_path = os.path.join(os.path.dirname(__file__), "build/nms_custom_op.so")

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


class NMS(nn.Module):
    def forward(self, scores, boxes, classes):
        return poptorch.custom_op(
            inputs=[scores, boxes, classes],
            name="Nms",
            domain="ai.graphcore",
            domain_version=1,
            attributes={"threshold": 0.5, "numDetections": 3},
            example_outputs=[torch.zeros(dtype=torch.int, size=[3])],
        )


model = NMS()
inference_model = poptorch.inferenceModel(model)

t1 = torch.tensor([1.0])
t2 = torch.tensor([2.0])


load_custom_ops_lib()
input = torch.arange(1, 5, dtype=torch.float32, requires_grad=True).view(1, 1, 2, 2)
scores = np.array([.9, .75, .6, .95, .5, .3, .9, .75, .6, .95, .5, .3]).astype(np.float32).reshape(2, 6)
boxes = np.array([0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11, 0, 10.1, 1, 11.1,
                  0, 100, 1, 101, 0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11,
                  0, 10.1, 1, 11.1, 0, 100,  1, 101]).astype(np.float32).reshape(2, 6, 4)
classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.int32).reshape(2, 6)

# assert inference_model(t1, t2) == 3.0
res = inference_model(torch.from_numpy(scores), torch.from_numpy(boxes), torch.from_numpy(classes))
print(res)
print("Success")
