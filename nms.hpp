// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "ipu_utils.hpp"
#include <poputil/DebugInfo.hpp>
#include <utility>

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, const poplar::Tensor &classes, float threshold,
    int num_detections, float score_threshold = 0.0, float sigma = 0.0f,
    bool useGather = false, bool inPlace = false,
    const poplar::DebugContext &dc = {});

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, float threshold, int num_detections,
    float score_threshold = 0.0, float sigma = 0.0f, bool useGather = false,
    bool inPlace = false, const poplar::DebugContext &dc = {});

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nmsMulti(poplar::Graph &graph, program::Sequence &prog,
         const poplar::Tensor &scores, const poplar::Tensor &boxes,
         float threshold, int num_detections, float score_threshold = 0.0f,
         float sigma = 0.0f, bool useGather = false, bool inPlace = false,
         const poplar::DebugContext &dc = {});
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nmsMultiLarge(poplar::Graph &graph, program::Sequence &prog,
              const poplar::Tensor &scores, const poplar::Tensor &boxes,
              float threshold, int num_detections, float score_threshold,
              bool useGather = false, uint32_t topk = 1,
              const poplar::DebugContext &dc = {}, uint32_t minPerTile = 1);

std::pair<poplar::Tensor, poplar::Tensor>
localMaxAndArgMax(poplar::Graph &graph, const poplar::Tensor &input,
                  const poplar::Tensor &indices, poplar::Type,
                  program::Sequence &prog,
                  const poplar::DebugContext &debugContext);
