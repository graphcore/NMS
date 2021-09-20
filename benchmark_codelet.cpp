// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef __IPU__
#include <ipu_vector_math>
#endif

#include <algorithm>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <typename T> class RoundRobinVertex : public poplar::Vertex {
public:
  RoundRobinVertex();
  poplar::Input<T> input;
  poplar::Output<T> output;

  const uint32_t repeat;
  bool compute() {
    float sum = float(*input);
    for (uint32_t i = 0; i < repeat; ++i) {
      sum += 1.0;
    }
    *output = T(sum);
    return true;
  }
};

template class RoundRobinVertex<float>;
template class RoundRobinVertex<half>;

template <typename T> class LinearMultiVertex : public poplar::MultiVertex {
public:
  LinearMultiVertex();
  poplar::Input<poplar::Vector<T>> input;
  poplar::Output<poplar::Vector<T, ONE_PTR>> output;

  const uint32_t repeat;
  bool compute(unsigned int workerId) {
    const size_t slice = input.size() / MultiVertex::numWorkers();
    const size_t offset = workerId * slice;
    const size_t end = std::min<size_t>(offset + slice, input.size());
    for (size_t i = offset; i < end; ++i) {
      float sum = float(input[i]);
      for (uint32_t j = 0; j < repeat; ++j) {
        sum += 1.0;
      }
      output[i] = T(sum);
    }
    return true;
  }
};

template class LinearMultiVertex<float>;
template class LinearMultiVertex<half>;

template <typename T> class LinearVertex : public poplar::Vertex {
public:
  LinearVertex();
  poplar::Input<poplar::Vector<T>> input;
  poplar::Output<poplar::Vector<T, ONE_PTR>> output;

  const uint32_t repeat;
  bool compute() {
    for (size_t i = 0; i < input.size(); ++i) {
      float sum = float(input[i]);
      for (uint32_t j = 0; j < repeat; ++j) {
        sum += 1.0;
      }
      output[i] = T(sum);
    }
    return true;
  }
};

template class LinearVertex<float>;
template class LinearVertex<half>;
