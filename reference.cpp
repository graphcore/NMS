// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

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

NDArray<uint32_t> argmax(const NDArray<float> &scores) {
  const auto dims = scores.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  NDArray<uint32_t> index{{batch}};
  for (size_t b = 0; b < batch; ++b) {
    float best = -std::numeric_limits<float>::max();
    uint32_t best_index = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < N; ++i) {
      if (scores[{b, i}] >= best) {
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
                   const NDArray<float> &areas,
                   const NDArray<uint32_t> &classes,
                   const NDArray<uint32_t> &best, float threshold) {
  const auto dims = scores.dims();
  size_t batch = dims[0];
  size_t N = dims[1];
  for (size_t b = 0; b < batch; ++b) {
    uint32_t best_b = best[{b}];
    for (size_t i = 0; i < N; ++i) {
      if (scores[{b, i}] > -std::numeric_limits<float>::max() &&
          classes[{b, i}] == classes[{b, best_b}]) {
        float iou = compute_iou(boxes, areas, b, i, best_b);
        if (iou > threshold) { // soft-nms here !!
          scores[{b, i}] = -std::numeric_limits<float>::max();
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
  NDArray<uint32_t> indices{{batch, numDetections},
                            std::numeric_limits<uint32_t>::max()};
  NDArray<float> areas = compute_area(boxes);
  NDArray<float> scores_copy{scores.shape(), scores.copy_data()};
  for (size_t i = 0; i < numDetections; ++i) {
    NDArray<uint32_t> best = argmax(scores_copy);
    for (size_t b = 0; b < batch; ++b) {
      if (scores_copy[{b, best[{b}]}] > -std::numeric_limits<float>::max()) {
        indices[{b, i}] = best[{b}];
        scores_copy[{b, best[{b}]}] = -std::numeric_limits<float>::max();
      }
    }
    update_scores(scores_copy, boxes, areas, classes, best, threshold);
  }
  return indices;
}

NDArray<uint32_t> multiArgmax(const NDArray<float> &scores) {
  const auto dims = scores.dims();
  const size_t batch = dims[0];
  const size_t N = dims[1];
  const size_t C = dims[2];
  NDArray<uint32_t> index{{batch, 2}};
  for (size_t b = 0; b < batch; ++b) {
    float best = -std::numeric_limits<float>::max();
    uint32_t best_index = std::numeric_limits<uint32_t>::max();
    uint32_t best_class = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < C; ++j) {
        if (scores[{b, i, j}] >= best) {
          best = scores[{b, i, j}];
          best_index = uint32_t(i);
          best_class = uint32_t(j);
        }
      }
    }
    index[{b, 0}] = best_index;
    index[{b, 1}] = best_class;
  }
  return index;
}

void updateMultiScores(NDArray<float> &scores, const NDArray<float> &boxes,
                       const NDArray<float> &areas,
                       const NDArray<uint32_t> &best, float threshold) {
  const auto dims = scores.dims();
  const size_t batch = dims[0];
  const size_t N = dims[1];
  for (size_t b = 0; b < batch; ++b) {
    uint32_t best_b = best[{b, 0}];
    uint32_t best_c = best[{b, 1}];
    for (size_t i = 0; i < N; ++i) {
      if (scores[{b, i, best_c}] > -std::numeric_limits<float>::max()) {
        float iou = compute_iou(boxes, areas, b, i, best_b);
        if (iou > threshold) { // soft-nms here !!
          scores[{b, i, best_c}] = -std::numeric_limits<float>::max();
        }
      }
    }
  }
}

NDArray<uint32_t> NmsMulti(const NDArray<float> &scores,
                           const NDArray<float> &boxes, uint32_t numDetections,
                           float threshold) {
  // scores: [batch, N, C]
  // boxes: [batch, N, 4]
  // return indices: [batch, K, 2] // [idx_boxes, idx_classes] N,C

  assert(scores.shape().rank() == 3);
  const auto dims = scores.dims();
  const size_t batch = dims[0];
  const size_t N = dims[1];
  assert(boxes.shape().rank() == 3);
  assert(numDetections <= N);
  assert(boxes.shape().same_dims({batch, N, 4}));
  NDArray<uint32_t> indices{{batch, numDetections, 2},
                            std::numeric_limits<uint32_t>::max()};
  NDArray<float> areas = compute_area(boxes);
  NDArray<float> scores_copy{scores.shape(), scores.copy_data()};
  for (size_t i = 0; i < numDetections; ++i) {
    NDArray<uint32_t> best = multiArgmax(scores_copy);
    for (size_t b = 0; b < batch; ++b) {
      if (scores_copy[{b, best[{b, 0}], best[{b, 1}]}] >
          std::numeric_limits<float>::min()) {
        indices[{b, i, 0}] = best[{b, 0}];
        indices[{b, i, 1}] = best[{b, 1}];
        scores_copy[{b, best[{b, 0}], best[{b, 1}]}] =
            -std::numeric_limits<float>::max();
      }
    }
    updateMultiScores(scores_copy, boxes, areas, best, threshold);
  }
  return indices;
}
