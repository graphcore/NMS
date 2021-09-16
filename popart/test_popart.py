# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import ctypes
import os

import numpy as np
import popart


# Define a function to build and run the leaky relu graph with
# specified input tensor data and alpha value
def build_and_run_graph(input_data, scaling_factor, run_on_ipu):
    builder = popart.Builder()
    print("Debug", input_data.shape)
    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", input_data.shape))
    inter = builder.aiOnnx.add([input_tensor, input_tensor])
    label_tensor = builder.addInputTensor(popart.TensorInfo("INT32", [1, 1, 4]))
    output_tensor = builder.customOp(
        opName="Upsample",
        opVersion=1,
        domain="com.acme",
        inputs=[inter],
        attributes={"scaling_factor": scaling_factor},
        numOutputs=1,
    )[0]

    output_probs = builder.aiOnnx.softmax([output_tensor], axis=2)
    builder.addOutputTensor(output_probs)
    loss = builder.aiGraphcore.nllloss([output_probs, label_tensor], popart.ReductionType.Sum, debugPrefix="loss")
    proto = builder.getModelProto()

    anchors = {output_tensor: popart.AnchorReturnType("FINAL")}
    dataFlow = popart.DataFlow(1, anchors)

    if run_on_ipu:
        device = popart.DeviceManager().acquireAvailableDevice(1)
        print("IPU hardware device acquired")
    else:
        device = popart.DeviceManager().createIpuModelDevice({})
        print("Running on IPU Model")

    print("scaling_factor={}".format(scaling_factor))

    optimizer = popart.ConstSGD(0.001)
    options = popart.SessionOptions()
    options.enableFloatingPointChecks = False
    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        deviceInfo=device,
        userOptions=options,
    )
    session.prepareDevice()
    result = session.initAnchorArrays()
    session.weightsFromHost()

    X = (np.array(input_data)).astype(np.float32)
    Y = np.array([1, 1, 1, 1]).reshape([1, 1, 4]).astype(np.int32)
    print("X={}".format(X))

    stepio = popart.PyStepIO({input_tensor: X, label_tensor: Y}, result)
    session.run(stepio, "Upsample")

    return result


def load_custom_ops_lib():
    so_path = os.path.join(os.path.dirname(__file__), "build/upsample_custom_op.so")

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


if __name__ == "__main__":
    load_custom_ops_lib()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaling_factor",
        help="sets the upsample scaling_factor attribute",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--ipu", help="run on available IPU hardware device", action="store_true"
    )

    args = parser.parse_args()
    input = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    print(input)

    result = build_and_run_graph(input, args.scaling_factor, args.ipu)

    print("RESULT X")
    print(result)
