# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import ctypes
import os

import numpy as np
import popart


# Define a function to build and run the leaky relu graph with
# specified input tensor data and alpha value
def build_and_run_graph(scores_data, boxes_data, threshold, num_detections, run_on_ipu):
    builder = popart.Builder()

    scores_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", scores_data.shape))
    boxes_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", boxes_data.shape))

    output_tensor = builder.customOp(
        opName="Nms",
        opVersion=1,
        domain="ai.graphcore",
        inputs=[scores_tensor, boxes_tensor],
        attributes={"threshold": threshold, "numDetections": num_detections, "scoreThreshold": 0.0, "useGather": 1,
                    "inPlace": 0},
        numOutputs=5,
    )[0]

    builder.addOutputTensor(output_tensor)

    proto = builder.getModelProto()

    anchors = {output_tensor: popart.AnchorReturnType("FINAL")}
    dataFlow = popart.DataFlow(1, anchors)

    if run_on_ipu:
        device = popart.DeviceManager().acquireAvailableDevice(1)
        print("IPU hardware device acquired")
    else:
        device = popart.DeviceManager().createIpuModelDevice({})
        print("Running on IPU Model")

    session = popart.InferenceSession(proto, dataFlow, device)

    session.prepareDevice()
    result = session.initAnchorArrays()

    S = (np.array(scores_data)).astype(np.float32)
    B = (np.array(boxes_data)).astype(np.float32)

    stepio = popart.PyStepIO({scores_tensor: S, boxes_tensor: B}, result)
    session.run(stepio, "Nms")

    return result


def load_custom_ops_lib():
    so_path = os.path.join(os.path.dirname(__file__), "build/nms_custom_op.so")

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


if __name__ == "__main__":
    load_custom_ops_lib()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        help="sets the upsample scaling_factor attribute",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ipu", help="run on available IPU hardware device", action="store_true"
    )

    args = parser.parse_args()
    scores = np.array([.9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7,
                       .9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7]).astype(np.float32).reshape(2, 6, 2)
    boxes = np.array([0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11, 0, 10.1, 1, 11.1,
                      0, 100, 1, 101, 0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11,
                      0, 10.1, 1, 11.1, 0, 100,  1, 101]).astype(np.float32).reshape(2, 6, 4)

    result = build_and_run_graph(scores, boxes, args.threshold, 6, args.ipu)

    print("RESULT X")
    print(result)
