#pragma once

#include "ipu_utils.hpp"

poplar::Tensor build_nms(poplar::Graph &graph, poplar::program::Sequence &prog,
                         const poplar::Tensor &scores,
                         const poplar::Tensor &boxes, uint32_t numDetections,
                         float threshold);
