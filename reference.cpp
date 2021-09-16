// Copyright (c) 2021, Graphcore Ltd, All rights reserved.

#include "reference.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

NDArray<uint32_t> sort_indices(const NDArray<float> &scores) {
  const auto dims = scores.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  NDArray<uint32_t> index{{batch, N}};
  for (size_t b = 0; b < batch; ++b) {
    std::iota(index({b, 0}), index({b, N - 1}) + 1, 0);
    std::stable_sort(index({b, 0}), index({b, N - 1}) + 1,
                     [&](size_t i1, size_t i2) {
                       return scores[{b, i1}] > scores[{b, i2}];
                     });
  }
  return index;
}

NDArray<uint32_t> argmax(const NDArray<float> &scores,
                         const NDArray<uint8_t> &keep) {
  const auto dims = scores.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  NDArray<uint32_t> index{{batch}};
  for (size_t b = 0; b < batch; ++b) {
    float best = std::numeric_limits<float>::min();
    uint32_t best_index = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < N; ++i) {
      if (keep[{b, i}] == 0 && scores[{b, i}] > best) {
        best = scores[{b, i}];
        best_index = uint32_t(i);
      }
    }
    index[{b}] = best_index;
  }
  return index;
}

NDArray<float> compute_area(const NDArray<float> &boxes) {
  const auto dims = boxes.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  NDArray<float> areas{{batch, N}};
  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < N; ++i) {
      float x1 = boxes[{b, i, 0}];
      float y1 = boxes[{b, i, 1}];
      float x2 = boxes[{b, i, 2}];
      float y2 = boxes[{b, i, 3}];
      areas[{b, i}] = (x2 - x1) * (y2 - y1);
    }
  }
  return areas;
}

float compute_iou(const NDArray<float> &boxes, const NDArray<float> &areas,
                  size_t batch, size_t current, size_t best) {
  float xx1 = std::max(boxes[{batch, current, 0}], boxes[{batch, best, 0}]);
  float yy1 = std::max(boxes[{batch, current, 1}], boxes[{batch, best, 1}]);
  float xx2 = std::min(boxes[{batch, current, 2}], boxes[{batch, best, 2}]);
  float yy2 = std::min(boxes[{batch, current, 3}], boxes[{batch, best, 3}]);
  float w = std::max(0.0f, xx2 - xx1);
  float h = std::max(0.0f, yy2 - yy1);
  float inter = w * h;

  float iou = inter / (areas[{batch, current}] + areas[{batch, best}] - inter);
  return iou;
}

void update_scores(NDArray<float> &scores, const NDArray<float> &boxes,
                   const NDArray<float> &areas, const NDArray<uint8_t> &keep,
                   const NDArray<uint32_t> &classes,
                   const NDArray<uint32_t> &best, float threshold) {
  const auto dims = scores.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  for (size_t b = 0; b < batch; ++b) {
    uint32_t best_b = best[{b}];
    for (size_t i = 0; i < N; ++i) {
      if (keep[{b, i}] == 0 && classes[{b, i}] == classes[{b, best_b}]) {
        float iou = compute_iou(boxes, areas, b, i, best_b);
        if (iou > threshold) { // soft-nms here !!
          scores[{b, i}] = 0.0;
        }
      }
    }
  }
}

NDArray<uint32_t> Nms(const NDArray<float> &scores, const NDArray<float> &boxes,
                      const NDArray<uint32_t> &classes, uint32_t numDetections,
                      float threshold) {
  // scores: [batch, N]
  // boxes: [batch, N, 4]
  // classes: [batch, N]
  // return indices: [batch, k]
  assert(scores.shape().rank() == 2);
  const auto dims = scores.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  assert(boxes.shape().rank() == 3);
  assert(classes.shape().rank() == 2);
  assert(numDetections <= N);
  assert(boxes.shape().same_dims({batch, N, 4}));
  assert(classes.shape().same_dims({batch, N}));
  NDArray<uint32_t> indices{{batch, numDetections}};
  NDArray<float> areas = compute_area(boxes);
  // areas.print(std::cerr);
  NDArray<uint8_t> keep{{batch, N}, 0};
  NDArray<float> scores_copy{scores.shape(), scores.copy_data()};
  NDArray<float> last_scores{indices.shape()};
  for (size_t i = 0; i < numDetections; ++i) {
    NDArray<uint32_t> best = argmax(scores_copy, keep);
    for (size_t b = 0; b < batch; ++b) {
      keep[{b, size_t(best[{b}])}] = 1; // mark the best indices
      indices[{b, i}] = best[{b}];
      last_scores[{b, i}] = scores_copy[{b, best[{b}]}];
      scores_copy[{b, best[{b}]}] = std::numeric_limits<float>::min();
    }
    update_scores(scores_copy, boxes, areas, keep, classes, best, threshold);
  }
  return indices;
}
