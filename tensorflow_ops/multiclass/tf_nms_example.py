# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python import ipu

threshold = 0.5
num_detections = 5

def nms(scores, boxes):
    attributes = {
        "threshold": threshold,
        "scoreThreshold": 0.5,
        "numDetections": num_detections,
        "useGather" : False
    }
    attributes_json = json.dumps(attributes)
    output_shape = [scores.shape[0], num_detections]
    outputs = {
        "output_types": [tf.int32, tf.float32, tf.float32, tf.int32, tf.int32],
        "output_shapes": [tf.TensorShape(output_shape), tf.TensorShape(output_shape),
                          tf.TensorShape([scores.shape[0], num_detections, 4]),
                          tf.TensorShape(output_shape), tf.TensorShape([scores.shape[0]])]
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "build/nms_custom_op.so")
    gp_path = os.path.join(base_path, "codelet.cpp")
    return ipu.custom_ops.precompiled_user_op([scores, boxes],
                                              lib_path,
                                              gp_path,
                                              attributes=attributes_json,
                                              outs=outputs)


if __name__ == '__main__':
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    bs = 2
    N = 6
    with tf.device("cpu"):
        scores_data = tf.placeholder(np.float32, [bs, N, 2])
        boxes_data = tf.placeholder(np.float32, [bs, N, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(nms, [scores_data, boxes_data])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores = np.array([.9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7,
                           .9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7]).astype(np.float32).reshape(2, 6, 2)
        boxes = np.array([0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11, 0, 10.1, 1, 11.1,
                          0, 100, 1, 101, 0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11,
                          0, 10.1, 1, 11.1, 0, 100,  1, 101]).astype(np.float32).reshape(2, 6, 4)


        result = sess.run(xla_result, feed_dict={scores_data: scores, boxes_data: boxes})
        print(result)
