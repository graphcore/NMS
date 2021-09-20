// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

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

template <class Compare, class Iterator>
void sift_down(Iterator first, Compare comp,
               typename std::iterator_traits<Iterator>::difference_type len,
               Iterator start) {
  typedef
      typename std::iterator_traits<Iterator>::difference_type difference_type;
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  // left-child of __start is at 2 * __start + 1
  // right-child of __start is at 2 * __start + 2
  difference_type child = start - first;

  if (len < 2 || (len - 2) / 2 < child)
    return;

  child = 2 * child + 1;
  Iterator child_i = first + child;

  if ((child + 1) < len && comp(*child_i, *(child_i + 1))) {
    // right-child exists and is greater than left-child
    ++child_i;
    ++child;
  }

  // check if we are in heap-order
  if (comp(*child_i, *start))
    // we are, __start is larger than it's largest child
    return;

  value_type top(std::move(*start));
  do {
    // we are not in heap-order, swap the parent with it's largest child
    *start = std::move(*child_i);
    start = child_i;

    if ((len - 2) / 2 < child)
      break;

    // recompute the child based off of the updated parent
    child = 2 * child + 1;
    child_i = first + child;

    if ((child + 1) < len && comp(*child_i, *(child_i + 1))) {
      // right-child exists and is greater than left-child
      ++child_i;
      ++child;
    }

    // check if we are in heap-order
  } while (!comp(*child_i, top));
  *start = std::move(top);
}

template <typename T, typename U> class TopKVertex : public poplar::Vertex {
public:
  TopKVertex();
  poplar::Input<poplar::Vector<T>> input; // [N, C]
  poplar::Output<poplar::Vector<U>> topk; // [N, k]

  const uint32_t C;
  const uint32_t K;
  const float score_threshold;

  static constexpr U last_value = std::numeric_limits<U>::max();

  void top_1(const uint32_t N) {
    for (uint32_t i = 0; i < N; ++i) {
      U argmax = last_value;
      T max = -std::numeric_limits<T>::max();
      for (uint32_t j = 0; j < C; ++j) {
        T current = input[i * C + j];
        if (float(current) > score_threshold && current > max) {
          max = current;
          argmax = j;
        }
      }
      topk[i] = argmax;
    }
  }

  void top_i(const uint32_t N, const uint32_t i) {
    auto sortIndex = [&](const U &a, const U &b) {
      return input[i * C + a] > input[i * C + b];
    };
    U score_index = 0, top_index = 0;
    while (score_index < C && top_index < K) {
      if (float(input[i * C + score_index]) > score_threshold) {
        topk[i * K + top_index] = score_index;
        ++top_index;
      }
      ++score_index;
    }
    if (score_index >= C && top_index < K) {
      topk[i * K + top_index] = last_value;
    }
    if (top_index > 1) { // more than 1 valid value -> must sort
      std::make_heap(&topk[i * K], &topk[i * K] + top_index, sortIndex);
      for (uint32_t j = score_index; j < C; ++j) {
        if (input[i * C + topk[i * K]] < input[i * C + j]) {
          topk[i * K] = j;
          sift_down(&topk[i * K], sortIndex, top_index, &topk[i * K]);
        }
      }
      std::sort_heap(&topk[i * K], &topk[i * K] + top_index, sortIndex);
    }
  }
  bool compute() {
    const uint32_t N = input.size() / C;
    if (K == 1) {
      top_1(N);
      return true;
    }

    for (uint32_t i = 0; i < N; ++i) {
      top_i(N, i);
    }
    return true;
  }
};

template class TopKVertex<float, unsigned int>;
template class TopKVertex<float, unsigned short>;
template class TopKVertex<half, unsigned int>;
template class TopKVertex<half, unsigned short>;

template <typename T, typename U>
inline U best(const U *topk, const T *input, const uint32_t K, const uint32_t C,
              const U left, const U right) {
  constexpr U last_value = std::numeric_limits<U>::max();
  const uint32_t top_left = left != last_value ? topk[left * K] : last_value;
  const uint32_t top_right = right != last_value ? topk[right * K] : last_value;
  if (top_left != last_value) {
    if (top_right != last_value) {
      if (input[left * C + top_left] >= input[right * C + top_right]) {
        return left;
      }
      return right;
    } else { // left has value but not right
      return left;
    }
  } else {
    if (top_right != last_value) {
      return right;
    }
  }
  return last_value;
}

template <typename T, typename U>
class BuildWinnerTreeVertex : public poplar::Vertex {
public:
  BuildWinnerTreeVertex();
  poplar::Input<poplar::Vector<T>> input; // [N, C]
  poplar::Input<poplar::Vector<U>>
      topk; // [N,k] sorted in decreasing order (topk of input).
  poplar::Output<poplar::Vector<U>> tree; // [(nearest power of 2 of N) - 1]

  const uint32_t C;
  const uint32_t K;

  static constexpr U last_value = std::numeric_limits<U>::max();

  void leaves(const uint32_t N, const uint32_t level,
              const uint32_t tree_offset) {
    for (U i = 0; i < level; ++i) {
      const U left = i * 2, right = left + 1;
      U val = last_value;
      if (left < N) {
        if (right < N) {
          val = best(topk.begin(), input.begin(), K, C, left, right);
        } else {
          if (topk[left * K] != last_value) {
            val = left;
          }
        }
      }
      assert(val == last_value || val < N);
      tree[tree_offset + i] = val;
    }
  }

  void level_i(const uint32_t N, const uint32_t level,
               const uint32_t out_offset, const uint32_t in_offset) {
    for (U i = 0; i < level; ++i) {
      const U left = in_offset + i * 2, right = left + 1;
      const U tleft = left == last_value ? last_value : tree[left];
      const U tright = right == last_value ? last_value : tree[right];
      U val = best(topk.begin(), input.begin(), K, C, tleft, tright);
      assert(val == last_value || val < N);
      tree[out_offset + i] = val;
    }
  }

  bool compute() {
    const uint32_t N = input.size() / C;
    uint32_t level = (tree.size() + 1) / 2;
    uint32_t tree_offset = tree.size() - level;
    leaves(N, level, tree_offset);
    level /= 2;
    tree_offset -= level;
    while (level >= 1) {
      level_i(N, level, tree_offset, tree_offset + level);
      level /= 2;
      tree_offset -= level;
    }
    return true;
  }
};

template class BuildWinnerTreeVertex<float, unsigned int>;
template class BuildWinnerTreeVertex<float, unsigned short>;
template class BuildWinnerTreeVertex<half, unsigned int>;
template class BuildWinnerTreeVertex<half, unsigned short>;

template <typename U> void markInvalid(U *topk, const uint32_t K) {
  const U last_value = std::numeric_limits<U>::max();
  const U invalid = last_value - 1;
  for (uint32_t i = 0; i < K; ++i) {
    U val = topk[i];
    if (val < invalid) { // first valid value
      topk[i] = i == K - 1 ? last_value : invalid;
      return;
    } else {
      if (topk[i] == last_value) { // no more value
        return;
      }
    }
  }
}

template <typename U> void firstValid(U *topk, const uint32_t K) {
  constexpr U last_value = std::numeric_limits<U>::max();
  constexpr U invalid = last_value - 1;
  for (uint32_t i = 0; i < K; ++i) {
    U val = topk[i];
    if (val < invalid) { // first valid value
      if (i == 0)
        return;
      topk[0] = val;
      topk[i] = i == K - 1 ? last_value : invalid;
      return;
    } else {
      if (topk[i] == last_value) { // no more value
        return;
      }
    }
  }
}

template <typename T, typename U> class MaxTreeVertex : public poplar::Vertex {
public:
  MaxTreeVertex();
  poplar::Input<poplar::Vector<T>> input; // [N, C]
  poplar::InOut<poplar::Vector<U>>
      topk; // [N,k] sorted in decreasing order (topk of input).
  poplar::InOut<poplar::Vector<U>> tree; // [(nearest power of 2 of N) - 1]
  poplar::Output<float> max;
  poplar::Output<uint32_t> argmax;
  const uint32_t C;
  const uint32_t K;
  const uint32_t offset;

  static constexpr U last_value = std::numeric_limits<U>::max();
  static constexpr U invalid = last_value - 1;

  U checkTopk(U index) {
    if (index >= invalid)
      return last_value;
    firstValid(&topk[index * K], K);
    U topkIndex = topk[index * K];
    if (topkIndex >= invalid) {
      topk[index * K] = last_value;
      return last_value;
    }
    return topkIndex;
  }

  U leaves(const uint32_t tree_offset) {
    const U top = tree[0];
    const U left = (top % 2) == 0 ? top : top - 1, right = left + 1;
    checkTopk(left);
    checkTopk(right);
    U val = best(topk.begin(), input.begin(), K, C, left, right);
    tree[tree_offset + (left / 2)] = val;
    return left / 2;
  }

  U level_i(const uint32_t out_offset, const uint32_t in_offset,
            const U index) {
    const U left = (index % 2) == 0 ? in_offset + index : in_offset + index - 1,
            right = left + 1;
    const U tleft = left == last_value ? last_value : tree[left];
    const U tright = right == last_value ? last_value : tree[right];
    checkTopk(tleft);
    checkTopk(tright);
    U val = best(topk.begin(), input.begin(), K, C, tleft, tright);
    tree[left / 2] = val;
    return (left / 2) - out_offset;
  }

  bool updateMax() {
    U best = tree[0];
    if (best == last_value) { // no value remain
      *max = -std::numeric_limits<float>::max();
      *argmax = last_value;
      return true;
    }
    U topk_best = topk[best * K];
    if (topk_best < invalid) { // no change since last update
      uint32_t index = best * C + topk_best;
      *max = float(input[index]);
      *argmax = offset + index;
      return true;
    }
    return false;
  }

  bool compute() {
    if (updateMax())
      return true;
    // need to update the tree
    uint32_t level = (tree.size() + 1) / 2;
    uint32_t tree_offset = tree.size() - level;
    firstValid(&topk[tree[0] * K], K);

    U index = leaves(tree_offset);
    level /= 2;
    tree_offset -= level;
    while (level >= 1) {
      index = level_i(tree_offset, tree_offset + level, index);
      level /= 2;
      tree_offset -= level;
    }
    updateMax();
    return true;
  }
};

template class MaxTreeVertex<float, unsigned int>;
template class MaxTreeVertex<float, unsigned short>;
template class MaxTreeVertex<half, unsigned int>;
template class MaxTreeVertex<half, unsigned short>;

template <typename T, typename U> class MaxTopkVertex : public poplar::Vertex {
public:
  MaxTopkVertex();
  poplar::Input<poplar::Vector<T>> input; // [N, C] N == 1
  poplar::InOut<poplar::Vector<U>>
      topk; // [N,k] sorted in decreasing order (topk of input).
  poplar::Output<T> max;
  poplar::Output<uint32_t> argmax;
  const uint32_t K;
  const uint32_t offset;

  static constexpr U last_value = std::numeric_limits<U>::max();
  static constexpr U invalid = last_value - 1;

  bool compute() {
    assert(topk.size() == K);
    firstValid(topk.begin(), K);
    uint32_t index = topk[0];
    if (index >= invalid) { // no value remain
      *max = -std::numeric_limits<T>::max();
      *argmax = last_value;
      return true;
    }
    *max = input[index];
    *argmax = offset + index;
    return true;
  }
};

template class MaxTopkVertex<float, unsigned int>;
template class MaxTopkVertex<float, unsigned short>;
template class MaxTopkVertex<half, unsigned int>;
template class MaxTopkVertex<half, unsigned short>;

template <typename T, typename U>
class MaxReduceVertex : public poplar::Vertex {
public:
  MaxReduceVertex();
  poplar::Input<poplar::Vector<T>> input;
  poplar::Input<poplar::Vector<U>> indices;
  poplar::Output<T> max;
  poplar::Output<U> argmax;

  bool compute() {
    T max_ = input[0];
    U argmax_ = 0;
    for (uint32_t i = 1; i < indices.size(); ++i) {
      if (input[i] > max_) {
        max_ = input[i];
        argmax_ = i;
      }
    }
    *max = max_;
    *argmax = indices[argmax_];
    return true;
  }
};

template class MaxReduceVertex<float, unsigned int>;
template class MaxReduceVertex<float, unsigned short>;
template class MaxReduceVertex<half, unsigned int>;
template class MaxReduceVertex<half, unsigned short>;

template <typename T> class UpdateBestTreeVertex : public poplar::Vertex {
public:
  UpdateBestTreeVertex();
  poplar::Input<uint32_t> best;
  poplar::InOut<poplar::Vector<T>> topk;

  const uint32_t C;
  const uint32_t K;
  const uint32_t offset;
  bool compute() {
    const uint32_t base_offset = C * offset;
    const uint32_t N = topk.size() / K;
    const uint32_t size = C * N;
    if (best >= base_offset && best < base_offset + size) {
      const uint32_t local_N = (best - base_offset) / C;
      markInvalid(&topk[local_N * K], K);
    }
    return true;
  }
};

template class UpdateBestTreeVertex<unsigned int>;
template class UpdateBestTreeVertex<unsigned short>;

template <typename T, typename U>
class UpdateWinnerTreeVertex : public poplar::Vertex {
public:
  UpdateWinnerTreeVertex();
  poplar::Input<poplar::Vector<T>> input; // [N, C]
  poplar::InOut<poplar::Vector<U>>
      topk; // [N,k] sorted in decreasing order (topk of input).
  poplar::InOut<poplar::Vector<U>> tree; // [(nearest power of 2 of N) - 1]

  const uint32_t C;
  const uint32_t K;

  U level_1(const uint32_t N, const uint32_t tree_offset) {
    const U top = tree[0];
    const U left = (top % 2) == 0 ? top : top - 1, right = left + 1;
    U val = best(topk.begin(), input.begin(), K, C, left, right);
    tree[tree_offset + (left / 2)] = val;
    return left / 2;
  }

  U level_i(const uint32_t N, const uint32_t out_offset,
            const uint32_t in_offset, const U index) {
    const U last_value = std::numeric_limits<U>::max();
    const U left = (index % 2) == 0 ? in_offset + index : in_offset + index - 1,
            right = left + 1;
    const U tleft = left == last_value ? last_value : tree[left];
    const U tright = right == last_value ? last_value : tree[right];
    U val = best(topk.begin(), input.begin(), K, C, tleft, tright);
    tree[left / 2] = val;
    return left / 2;
  }

  bool compute() {
    const U last_value = std::numeric_limits<U>::max();
    const uint32_t N = input.size() / C;
    uint32_t level = (tree.size() + 1) / 2;
    uint32_t tree_offset = tree.size() - level;
    if (tree[0] == last_value)
      return true;
    markInvalid(&topk[tree[0]], K);
    firstValid(&topk[tree[0]], K);

    U index = level_1(N, tree_offset);
    level /= 2;
    tree_offset -= level;
    while (level >= 1) {
      index = level_i(N, tree_offset, tree_offset + level, index);
      level /= 2;
      tree_offset -= level;
    }

    return true;
  }
};

template class UpdateWinnerTreeVertex<float, unsigned int>;
template class UpdateWinnerTreeVertex<float, unsigned short>;
template class UpdateWinnerTreeVertex<half, unsigned int>;
template class UpdateWinnerTreeVertex<half, unsigned short>;

template <typename T> class UpdateBestVertex : public poplar::Vertex {
public:
  UpdateBestVertex();
  poplar::Input<uint32_t> best;
  poplar::InOut<poplar::Vector<T>> scores;

  const uint32_t C;
  const uint32_t offset;

  bool compute() {
    const uint32_t base_offset = C * offset;
    if (best >= base_offset && best < base_offset + scores.size()) {
      scores[best - base_offset] = -std::numeric_limits<T>::max();
    }
    return true;
  }
};

template class UpdateBestVertex<float>;
template class UpdateBestVertex<half>;

template <typename T, typename C>
class UpdateAnswerVertex : public poplar::Vertex {
public:
  UpdateAnswerVertex();
  poplar::Input<uint32_t> best_indices;                 // [bs]
  poplar::Input<T> best_scores;                         // [bs]
  poplar::Input<poplar::Vector<T, ONE_PTR>> best_boxes; // [bs, 4]
  poplar::Input<C> best_classes;                        // [bs]

  poplar::InOut<poplar::Vector<int32_t, ONE_PTR>> lengths; // [K, bs]

  poplar::InOut<poplar::Vector<int32_t>> top_indices;    // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_scores;  // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_boxes;   // [K, bs, 4]
  poplar::InOut<poplar::Vector<C, ONE_PTR>> top_classes; // [K, bs]
  poplar::Input<uint32_t> i;

  const float score_threshold;
  const int32_t K;
  const uint32_t offset;
  bool compute() {
    const uint32_t size = top_indices.size();
    if (i >= offset && i < offset + size) {
      const uint32_t j = i - offset;
      if (float(*best_scores) > score_threshold) { // let's copy the values
        top_indices[j] = best_indices;
        top_scores[j] = best_scores;
        top_classes[j] = best_classes;
        std::memcpy(&top_boxes[j * 4], &best_boxes[0], 4 * sizeof(T));
      } else { // we discard the value, so we need to update lengths
        if (lengths[j] == K) {
          lengths[j] = int32_t(i);
        }
      }
    }
    return true;
  }
};

template class UpdateAnswerVertex<float, int>;
template class UpdateAnswerVertex<float, unsigned int>;
template class UpdateAnswerVertex<half, int>;
template class UpdateAnswerVertex<half, unsigned int>;

template <typename T, typename U>
class UpdateAnswerMultiVertex : public poplar::Vertex {
public:
  UpdateAnswerMultiVertex();
  poplar::Input<U> best_indices;                        // [bs]
  poplar::Input<T> best_scores;                         // [bs]
  poplar::Input<poplar::Vector<T, ONE_PTR>> best_boxes; // [bs, 4]

  poplar::InOut<poplar::Vector<int32_t, ONE_PTR>> lengths; // [K, bs]

  poplar::InOut<poplar::Vector<int32_t>> top_indices;          // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_scores;        // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_boxes;         // [K, bs, 4]
  poplar::InOut<poplar::Vector<int32_t, ONE_PTR>> top_classes; // [K, bs]
  poplar::Input<uint32_t> i;

  const float score_threshold;
  const int32_t K;
  const int32_t C;
  const uint32_t offset;
  bool compute() {
    const uint32_t size = top_indices.size();
    if (i >= offset && i < offset + size) {
      const uint32_t j = i - offset;
      if (float(*best_scores) > score_threshold) { // let's copy the values
        top_indices[j] = best_indices / C;
        top_scores[j] = best_scores;
        top_classes[j] = best_indices % C;
        std::memcpy(&top_boxes[j * 4], &best_boxes[0], 4 * sizeof(T));
      } else { // we discard the value, so we need to update lengths
        if (lengths[j] == K) {
          lengths[j] = int32_t(i);
        }
      }
    }
    return true;
  }
};

template class UpdateAnswerMultiVertex<float, unsigned int>;
template class UpdateAnswerMultiVertex<float, unsigned short>;
template class UpdateAnswerMultiVertex<half, unsigned int>;
template class UpdateAnswerMultiVertex<half, unsigned short>;

template <typename T, typename U> class SliceVertex : public poplar::Vertex {
public:
  SliceVertex();
  poplar::Input<poplar::Vector<T, ONE_PTR>> input;
  poplar::Input<U> index;
  poplar::Output<T> output;
  const uint32_t C;
  bool compute() {
    *output = input[index / C];
    return true;
  }
};

template class SliceVertex<float, unsigned int>;
template class SliceVertex<float, unsigned short>;
template class SliceVertex<half, unsigned int>;
template class SliceVertex<half, unsigned short>;
template class SliceVertex<unsigned int, unsigned int>;
template class SliceVertex<unsigned int, unsigned short>;
template class SliceVertex<int, unsigned int>;
template class SliceVertex<int, unsigned short>;

template <typename T, typename C> class GatherVertex : public poplar::Vertex {
public:
  GatherVertex();
  poplar::Input<uint32_t> index;
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes;
  poplar::Input<poplar::Vector<C>> classes;
  poplar::InOut<poplar::Vector<T, ONE_PTR>> output;
  poplar::InOut<C> outputClass;

  const uint32_t offset;
  bool compute() {
    if (index >= offset && index < offset + classes.size()) {
      const uint32_t i = index - offset;
      std::memcpy(&output[0], &boxes[i * 4], 4 * sizeof(T));
      *outputClass = classes[i];
    }
    return true;
  }
};

template class GatherVertex<float, int>;
template class GatherVertex<half, int>;
template class GatherVertex<float, unsigned int>;
template class GatherVertex<half, unsigned int>;

template <typename T, typename U>
class GatherMultiVertex : public poplar::Vertex {
public:
  GatherMultiVertex();
  poplar::Input<U> index;

  poplar::Input<poplar::Vector<T>> boxes;
  poplar::InOut<poplar::Vector<T, ONE_PTR>> output;

  const uint32_t C;
  const uint32_t offset;
  bool compute() {
    const size_t N = boxes.size() / 4;
    const uint32_t base_offset = offset * C;
    if (index >= base_offset && index < base_offset + N * C) {
      const uint32_t i = index / C - offset;
      std::memcpy(&output[0], &boxes[i * 4], 4 * sizeof(T));
    }
    return true;
  }
};

template class GatherMultiVertex<float, unsigned int>;
template class GatherMultiVertex<float, unsigned short>;
template class GatherMultiVertex<half, unsigned int>;
template class GatherMultiVertex<half, unsigned short>;

#ifdef __IPU__
template <typename T>
inline float computeIOU(const T *boxes, const T *bestBox,
                        const float &bestArea) {
  T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
  T x1 = boxes[0], y1 = boxes[1], x2 = boxes[2], y2 = boxes[3];
  float area = float(x2 - x1) * float(y2 - y1);
  T xx1 = std::max(x1, bx1);
  T yy1 = std::max(y1, by1);
  T xx2 = std::min(x2, bx2);
  T yy2 = std::min(y2, by2);
  float w = std::max(0.0f, float(xx2 - xx1));
  float h = std::max(0.0f, float(yy2 - yy1));
  float inter = w * h;

  float iou = inter / (area + bestArea - inter);
  return iou;
}
inline float computeIOU(const half *boxes, const half *bestBox,
                        const float &bestArea) {
  half2 *boxes_ = (half2 *)boxes, *bestBox_ = (half2 *)bestBox;
  half2 x1y1 = boxes_[0], x2y2 = boxes_[1];
  half2 bx1by1 = bestBox_[0], bx2by2 = bestBox_[1];
  half2 diff = x2y2 - x1y1;
  float area = float(diff[0]) * float(diff[1]);
  half2 xx1yy1 = half2_fmax(x1y1, bx1by1);
  half2 xx2yy2 = half2_fmin(x2y2, bx2by2);
  half2 zero = {0.0, 0.0};
  half2 diff2 = xx2yy2 - xx1yy1;
  half2 wh = half2_fmax(zero, diff2);
  float inter = float(wh[0]) * float(wh[1]);
  float iou = inter / (area + bestArea - inter);
  return iou;
}

#else
template <typename T>
inline float computeIOU(const T *boxes, const T *bestBox,
                        const float &bestArea) {
  T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
  T x1 = boxes[0], y1 = boxes[1], x2 = boxes[2], y2 = boxes[3];
  float area = float(x2 - x1) * float(y2 - y1);
  T xx1 = std::max(x1, bx1);
  T yy1 = std::max(y1, by1);
  T xx2 = std::min(x2, bx2);
  T yy2 = std::min(y2, by2);
  float w = std::max(0.0f, float(xx2 - xx1));
  float h = std::max(0.0f, float(yy2 - yy1));
  float inter = w * h;

  float iou = inter / (area + bestArea - inter);
  return iou;
}
#endif

template <typename T, typename C> class NmsVertex : public poplar::Vertex {
public:
  NmsVertex();
  poplar::InOut<poplar::Vector<T>> scores; // [bs, N]
  poplar::Input<poplar::Vector<C, ONE_PTR>> classes;
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes; // [bs, N, 4]
  poplar::Input<C> bestClass;
  poplar::Input<poplar::Vector<T, ONE_PTR>> bestBox; // [4]

  const float sigma;
  const float threshold;
  const float score_threshold;

  bool compute() {
    T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
    float barea = float(bx2 - bx1) * float(by2 - by1);
    for (size_t i = 0; i < scores.size(); ++i) {
      if (scores[i] > T(score_threshold) && classes[i] == bestClass) {
        float iou = computeIOU(&boxes[i * 4], &bestBox[0], barea);
        if (sigma == 0.0f) {
          if (iou > threshold) {
            scores[i] = -std::numeric_limits<T>::max();
          }
        } else {
          float weight = std::exp(-(iou * iou) / sigma); // paper version
          scores[i] *= weight;
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

template <typename T> class NmsMultiVertex : public poplar::Vertex {
public:
  NmsMultiVertex();
  poplar::InOut<poplar::Vector<T>> scores;         // [bs, N, C]
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes; // [bs, N, 4]
  poplar::Input<uint32_t> bestIndices;
  poplar::Input<poplar::Vector<T>> bestBox; // [4]

  const float sigma;
  const float threshold;
  const float score_threshold;
  const uint32_t C;

  bool compute() {
    T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
    float barea = float(bx2 - bx1) * float(by2 - by1);
    for (size_t i = 0; i < scores.size() / C; ++i) {
      const int32_t class_i = bestIndices % C;
      const int32_t score_i = i * C + class_i;
      if (scores[score_i] > T(score_threshold)) {
        float iou = computeIOU(&boxes[i * 4], &bestBox[0], barea);
        if (sigma == 0.0f) {
          if (iou > threshold) {
            scores[score_i] = -std::numeric_limits<T>::max();
          }
        } else {
          float weight = std::exp(-(iou * iou) / sigma); // paper version
          scores[score_i] *= weight;
        }
      }
    }
    return true;
  }
};

template class NmsMultiVertex<float>;
template class NmsMultiVertex<half>;

template <typename T, typename U> class NmsTreeVertex : public poplar::Vertex {
public:
  NmsTreeVertex();
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes; // [N, 4]
  poplar::Input<uint32_t> bestIdx;
  poplar::InOut<poplar::Vector<U>> topk;    // [N, K]
  poplar::Input<poplar::Vector<T>> bestBox; // [4]

  const float threshold;
  const uint32_t C;
  const uint32_t K;

  bool compute() {
    const U last_value = std::numeric_limits<U>::max();
    const U invalid = last_value - 1;
    const T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2],
            by2 = bestBox[3];
    const float barea = float(bx2 - bx1) * float(by2 - by1);
    const U bclass = bestIdx % C;
    for (size_t i = 0; i < topk.size(); ++i) {
      const U topk_i = topk[i];
      if (topk_i < invalid) {
        const U class_i = topk_i % C;
        if (class_i == bclass) {
          float iou = computeIOU(&boxes[i / K * 4], &bestBox[0], barea);
          if (iou > threshold) {
            topk[i] = invalid;
          }
        }
      }
    }
    return true;
  }
};

template class NmsTreeVertex<float, unsigned int>;
template class NmsTreeVertex<float, unsigned short>;
template class NmsTreeVertex<half, unsigned int>;
template class NmsTreeVertex<half, unsigned short>;

template <typename T> class ConditionVertex : public poplar::Vertex {
public:
  ConditionVertex();
  poplar::Input<poplar::Vector<T>> bestScores; // [bs]
  poplar::Input<uint32_t> i;
  poplar::Input<uint32_t> numIter;
  poplar::Output<bool> condition;
  const float score_threshold;

  bool compute() {
    if (i == 0) {
      *condition = true;
      return true;
    }
    bool res = i < numIter;
    uint32_t s_i = 0;
    while (res && s_i < bestScores.size()) {
      res = res || (float(bestScores[s_i]) > score_threshold);
      ++s_i;
    }
    *condition = res;
    return true;
  }
};

template class ConditionVertex<float>;
template class ConditionVertex<half>;
