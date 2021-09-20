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


numDetections = 6
numClasses = 2
class NMS(nn.Module):
    def forward(self, scores, boxes):
        batch = scores.shape[0]
        return poptorch.custom_op(
            inputs=[scores, boxes],
            name="Nms",
            domain="ai.graphcore",
            domain_version=1,
            attributes={"threshold": 0.5, "scoreThreshold": 0.0, "numDetections": numDetections, "useGather": 1},
            example_outputs=[torch.zeros(dtype=torch.int, size=[batch, numDetections]),
                             torch.zeros(dtype=scores.dtype, size=[batch, numDetections]),
                             torch.zeros(dtype=boxes.dtype, size=[batch, numDetections, 4]),
                             torch.zeros(dtype=torch.int, size=[batch, numDetections]),
                             torch.zeros(dtype=torch.int, size=[batch])]
        )


model = NMS()
inference_model = poptorch.inferenceModel(model)

t1 = torch.tensor([1.0])
t2 = torch.tensor([2.0])


load_custom_ops_lib()
scores = np.array([.9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7,
                   .9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7]).astype(np.float32).reshape(2, 6, 2)
boxes = np.array([0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11, 0, 10.1, 1, 11.1,
                      0, 100, 1, 101, 0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11,
                      0, 10.1, 1, 11.1, 0, 100,  1, 101]).astype(np.float32).reshape(2, 6, 4)

# assert inference_model(t1, t2) == 3.0
res = inference_model(torch.from_numpy(scores), torch.from_numpy(boxes))
print(res)
print("Success")
