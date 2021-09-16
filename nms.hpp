#pragma once

#include "ipu_utils.hpp"
#include <poputil/DebugInfo.hpp>
poplar::Tensor nms(poplar::Graph &graph, program::Sequence &prog,
                   const poplar::Tensor &scores, const poplar::Tensor &boxes,
                   const poplar::Tensor &classes, float threshold,
                   int num_detections, const poplar::DebugContext &dc = {});
