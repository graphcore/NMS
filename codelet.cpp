// Copyright (c) 2021, Graphcore Ltd, All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef __IPU__
#include <ipu_vector_math>
#endif

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <typename T> class UpdateBestVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint32_t>> best; // [bs]
  poplar::Input<poplar::Vector<uint32_t>> batches;
  poplar::Input<poplar::Vector<uint32_t>> indices;

  poplar::InOut<poplar::Vector<T>> scores;
  poplar::InOut<poplar::Vector<bool>> keep;
  bool compute() {
    for (int i = 0; i < batches.size(); ++i) {
      if (best[batches[i]] == indices[i]) { // we found the best
        scores[i] = std::numeric_limits<T>::min();
        keep[i] = true;
      }
    }
    return true;
  }
};

template class UpdateBestVertex<float>;
template class UpdateBestVertex<half>;

template <typename T, typename C> class NmsVertex : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<T>> scores; // [bs, N]
  poplar::Input<poplar::Vector<T>> areas;  // [bs, N] useful ?
  poplar::Input<poplar::Vector<bool>> keep;
  poplar::Input<poplar::Vector<uint32_t>> batches;
  poplar::Input<poplar::Vector<C>> classes;
  poplar::Input<poplar::Vector<T>> boxes;     // [bs, N, 4]
  poplar::Input<poplar::Vector<T>> bestBox;   // [bs, 4]
  poplar::Input<poplar::Vector<C>> bestClass; // [bs]
  poplar::Input<poplar::Vector<T>> bestArea;  // [bs]
  poplar::Input<float> threshold;

  bool compute() {
    assert(scores.size() == areas.size());
    assert(scores.size() == keep.size());
    assert(scores.size() * 4 == boxes.size());
    assert((bestBox.size() % 4) == 0);
    const size_t batch = bestBox.size() / 4;
    for (int i = 0; i < scores.size(); ++i) {
      const size_t bs_i = batches[i];
      if (keep[i] == false && classes[i] == bestClass[bs_i]) {
        T xx1 = std::max(boxes[i * 4], bestBox[bs_i * 4]);
        T yy1 = std::max(boxes[i * 4 + 1], bestBox[bs_i * 4 + 1]);
        T xx2 = std::min(boxes[i * 4 + 2], bestBox[bs_i * 4 + 2]);
        T yy2 = std::min(boxes[i * 4 + 3], bestBox[bs_i * 4 + 3]);
        float w = std::max(0.0f, float(xx2 - xx1));
        float h = std::max(0.0f, float(yy2 - yy1));
        float inter = w * h;

        float iou =
            float(inter) / (float(areas[i]) + float(bestArea[bs_i]) - inter);
        if (iou > threshold) {
          scores[i] = 0.0; // soft nms ?
        }
      }
    }
    return true;
  }
};

template class NmsVertex<float, int>;
template class NmsVertex<float, unsigned int>;
template class NmsVertex<half, int>;
template class NmsVertex<half, unsigned int>;
