# Non-Maximum Suppression (NMS) for IPU

This repository contains the source code for the Graphcore implementation of Non-Maximum Suppression.

Two variants are included - "best-class" and "multi-class". Multi-class NMS can assign multiple classes
per bounding-box.

In the multi-class version, or when classes are provided in the single class
version, only the bounding boxes with the same class as the best box would be
considered for overlapping.

# Quick-Start

Download and install the Poplar SDK following the Getting Started guide for your IPU system.
Source the `enable.sh` script to install Poplar.

To build the code:

```
$ cmake .
$ make
```

Now run the tests:

```
$ ./tests
```

# Including the ops in your application

There are three frameworks supported in this repository, TensorFlow 2 and PopTorch and PopART.

To include the operator in your preferred framework, you can use the custom operator
interface for both TensorFlow and PopTorch. To understand how to do this, see the
examples in the [tensorflow_ops](tensorflow_ops) and [poptorch_ops](poptorch_ops) directories. In each, you'll find an
example of single or "best" class NMS, and "multi-class" NMS.

To run the examples, you'll need to install the framework of your choice by `pip install`ing
the wheel from the Poplar SDK directory.

For PopTorch:

```
$ pip install <poplar_sdk>/poptorch-2.6.0+<...>-linux_x86_64.whl
```

For TensorFlow:
```
$ pip install <poplar_sdk>/tensorflow-2.6.3<...>.whl
```

Then for each example, you can install requirements (if necessary):

```
$ pip install -r requirements.txt
```

Finally, run the example (replacing `<framework>` with one of `tf`, `popart` or
`poptorch`, depending on the example you'd like to run.

```
$ python <framework>_nms_example.py
```

# Algorithm

## Introduction

The NMS algorithm is used in object detection systems to select the best
bounding box for an object and remove all overlapping bounding boxes, or
decrease their scores in soft-nms.

The inputs are:

* a set of N bounding boxes (usually overlapping)
* either of set of NxC scores for the multi class NMS, where C is
  the number of classes recognized by the system, or N scores in the single
  class NMS. The latter is usually obtained after processing the NxC
  scores, either by slicing the scores of 1 class, or by applying Argmax.
* optionally, a set of N classes.

The outputs are:

* the set of K bounding boxes corresponding to the top K scores, without
  overlap. Usually, only the indices are returned. In the case of the multi
  class NMS, the same box can appear more than once with different classes and
  scores.

## Pseudo-code

The algorithm is quite simple. Given:

* A set of `N` candidates boxes with their associated score, `Cand`.
* A number of detections `D`, i.e. how many bounding boxes should be
  returned.
* A overlapping threshold `T`.

The algorithm is then:

1. Select the box bwith the highest score from Cand and put it the answer set
   `A`.
2. Compute the overlapping score between `b` and all the other boxes in `Cand`
3. Remove from `Cand` all the boxes such that their overlapping score is higher
   than `T`
4. Repeat from step 1 until there is no box in `Cand` or cardinality of `A` is
   equal to `D`

More formally:

```
1. do
2.   b = max(Cand)
3.   A = A U b
4.   Cand = Cand \ b
5.   I = IoU(b, Cand)
6.   Cand = Cand \ [I > T]
7. while |A| < D and Cand not empty
```

## Soft-NMS

Soft-NMS is a variant of NMS where the scores of the overlapping boxes with
the best candidates are decayed instead of removed from the candidates list.

There are 2 variations:

* linear rescoring: score = (1 - IoU) x score

* Gaussian rescoring: score = exp(-IoU^2/sigma) x score

Only the Gaussian rescoring is implemented in Tensorflow version 5 of NMS, so
our implementation is also restricted to the Gaussian version, and is
triggered when the parameter `sigma` is different from 0.

In both versions, when the IoU is 0, the score remains the same.

Soft-NMS is supposed to improve the accuracy of detection system.


# Licensing

This repository is licensed under the MIT license - see the LICENSE file in this directory.

This repository includes derived work from the following, used within the test harness:
* [Catch2](https://github.com/catchorg/Catch2) (Boost Software License)
* [Lyra](https://www.bfgroup.xyz/Lyra/) (Boost Software License)
