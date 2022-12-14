// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include "utils.hpp"

NDArray<uint32_t> Nms(const NDArray<float> &scores, const NDArray<float> &boxes,
                      const NDArray<uint32_t> &classes, uint32_t numDetections,
                      float threshold);
NDArray<uint32_t> NmsMulti(const NDArray<float> &scores,
                           const NDArray<float> &boxes, uint32_t numDetections,
                           float threshold);
NDArray<uint32_t> sort_indices(const NDArray<float> &scores);
NDArray<float> compute_area(const NDArray<float> &boxes);
NDArray<uint32_t> argmax(const NDArray<float> &scores);
