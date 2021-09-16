// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
//#include "print.h"
#include <array>
#include <cmath>
//#include <ipudef.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
//#include <print.h>

using namespace poplar;

class addOneVertex : public poplar::Vertex {
public:
  addOneVertex();

  poplar::Input<int> batch_size;
  poplar::Vector<poplar::InOut<int>> var;

  bool compute() {
    for (int i = 0; i < batch_size; ++i) {
      var[i] = var[i] + 1;
    }
    return true;
  }
};

template <typename T> class GetMaxIndexVertex : public poplar::Vertex {

public:
  GetMaxIndexVertex();

  poplar::Input<int> in_row_start;
  poplar::Input<int> in_row_end;
  poplar::Input<float> is_first_round;
  poplar::Vector<poplar::Input<poplar::Vector<T>>> scores;
  poplar::Vector<poplar::Input<poplar::Vector<unsigned int>>> indexes;
  poplar::Vector<poplar::InOut<unsigned int>> max_index;
  poplar::Vector<poplar::InOut<T>> max_score;

  bool compute() {
    unsigned int length = in_row_end - in_row_start;
    max_score[0] = 0.0f;

    for (unsigned int i = 0; i < length; ++i) {
      // if (!is_first_round)
      // printf("%d max:%f max_score:%f max_idx:%d f_idx:%d\n", i, max_score[0],
      // scores[0][i], max_index[0], indexes[i][0]); printf("%d f_score:%f\n",
      // i, scores[0][i]);
      if (max_score[0] < scores[0][i]) {
        max_score[0] = scores[0][i];
        if (is_first_round == 1.0f) {
          max_index[0] = i + in_row_start;
        } else {
          max_index[0] = indexes[i][0];
        }
      }
    }

    return true;
  }
};
template class GetMaxIndexVertex<float>;
template class GetMaxIndexVertex<half>;

template <typename T> class PartialSetIthKeepVertex : public poplar::Vertex {

public:
  PartialSetIthKeepVertex();
  Input<int> in_row_start;
  Input<int> in_row_end;
  //   Vector<Input<int>>                                  j_tensor;
  Input<int> batch_size;
  poplar::Vector<poplar::InOut<Vector<unsigned int>>> result;
  poplar::Vector<poplar::InOut<Vector<T>>> resultbox;
  Vector<Input<int>> index;
  poplar::Vector<Input<Vector<T>>> box_i;
  poplar::Vector<Input<poplar::Vector<unsigned int>>> sorted_index;

  bool compute() {
    for (unsigned sample = 0; sample < batch_size; sample++) {
      if (index[sample] < in_row_start || index[sample] >= in_row_end) {
        return true;
      }

      int local_idx = index[sample] - in_row_start;
      result[sample][local_idx] = sorted_index[sample][0];
      resultbox[sample][local_idx * 4] = box_i[sample][0];
      resultbox[sample][local_idx * 4 + 1] = box_i[sample][1];
      resultbox[sample][local_idx * 4 + 2] = box_i[sample][2];
      resultbox[sample][local_idx * 4 + 3] = box_i[sample][3];
      // index[sample] = index[sample] + 1;
    }
    return true;
  }
};

template <typename T> class SetIthKeepVertex : public poplar::Vertex {

public:
  SetIthKeepVertex();
  //   Vector<Input<int>>                                  j_tensor;
  Input<int> batch_size;
  poplar::Vector<poplar::InOut<Vector<unsigned int>>> result;
  poplar::Vector<poplar::InOut<Vector<T>>> resultbox;
  Vector<InOut<int>> index;
  poplar::Vector<Input<Vector<T>>> box_i;
  Vector<InOut<unsigned int>> sorted_index;

  bool compute() {
    for (unsigned sample = 0; sample < batch_size; sample++) {
      result[sample][index[sample]] = sorted_index[0];
      resultbox[sample][index[sample] * 4] = box_i[sample][0];
      resultbox[sample][index[sample] * 4 + 1] = box_i[sample][1];
      resultbox[sample][index[sample] * 4 + 2] = box_i[sample][2];
      resultbox[sample][index[sample] * 4 + 3] = box_i[sample][3];
      index[sample] = index[sample] + 1;
    }
    return true;
  }
};
template class SetIthKeepVertex<half>;
template class SetIthKeepVertex<float>;
template class PartialSetIthKeepVertex<half>;
template class PartialSetIthKeepVertex<float>;

class fillZeroVertex : public poplar::Vertex {
public:
  fillZeroVertex();

  poplar::Vector<poplar::InOut<int>> var; // {L}

  bool compute() {
    int L = var.size();

    for (int i = 0; i < L; i++) {
      var[i] = 0;
    }
    return true;
  }
};

class fillTrueVertex : public poplar::Vertex {
public:
  fillTrueVertex();

  poplar::Vector<poplar::InOut<float>> keep; // {L}

  bool compute() {
    int L = keep.size();

    for (int i = 0; i < L; i++) {
      keep[i] = 1.0f;
    }
    return true;
  }
};

template <typename T> class NmsCoreVertex : public Vertex {
public:
  Vector<InOut<float>> keep_r;
  Vector<InOut<T>> score_r;
  Vector<InOut<T>> box_r;
  Input<Vector<unsigned int>> sorted_index;
  Input<float> nms_thresh;

  Input<unsigned int> idx;
  Vector<Input<T>> box_i; // vector of 4 elements filled per vertice
  Input<int> finish_r;

  // The compute method performs core computation
  bool compute() {
    if (sorted_index[0] == idx) {
      score_r[0] = 0.0f;
      return true;
    }
    float nms = nms_thresh; // Config parameter

    T box_b[4] = {box_i[0], box_i[1], box_i[2], box_i[3]};
    T box_s[4] = {box_r[0], box_r[1], box_r[2], box_r[3]};

    if (keep_r[0] == 1.0f and finish_r != 1) {
      // xy1   = max(boxes[r, :2], boxes_i[:2])    --> 2 scalars
      // xy2   = min(boxes[r, 2:], boxes_i[2:])    --> 2 scalars

      T xy1_0 = (box_r[0] > box_i[0]) ? box_r[0] : box_i[0];
      T xy1_1 = (box_r[1] > box_i[1]) ? box_r[1] : box_i[1];

      T xy2_0 = (box_r[2] < box_i[2]) ? box_r[2] : box_i[2];
      T xy2_1 = (box_r[3] < box_i[3]) ? box_r[3] : box_i[3];

      float tmp0 = float(xy2_0) - float(xy1_0); //+ 1.0f;
      float tmp1 = float(xy2_1) - float(xy1_1); //+ 1.0f;
      if (tmp0 < 0.0f)
        tmp0 = 0.0f; //-tmp0;
      if (tmp1 < 0.0f)
        tmp1 = 0.0f; //-tmp1;

      float inter = tmp0 * tmp1;
      float targetBoxArea = (float(box_b[2]) - float(box_b[0])) *
                            (float(box_b[3]) - float(box_b[1]));
      float currBoxArea = (float(box_s[2]) - float(box_s[0])) *
                          (float(box_s[3]) - float(box_s[1]));
      float denominator = currBoxArea + targetBoxArea - inter;
      if (denominator == 0.0f) {
        denominator = 1.0f;
      }
      float area_cross = inter / denominator;
      bool area_cond = area_cross < nms;
      // if (idx == 8939)
      // {
      //     printf("%f,%f,%f,%f\n", box_i[0], box_i[1], box_i[2], box_i[3]);
      //     printf("%f,%f,%f,%f\n", box_r[0], box_r[1], box_r[2], box_r[3]);
      //     printf("%f\n", targetBoxArea);
      //     printf("%f\n", currBoxArea);
      //     printf("%f\n", inter);
      //     // printf("%d\n", score_cond);
      //     printf("%f\n", area_cross);
      //     printf("%d\n", area_cond);
      // }
      bool cond = !area_cond;
      // clear the corresponding data field if mask is 0
      if (cond == true) {
        score_r[0] = 0.0f;
        box_r[0] = 0.0f;
        box_r[1] = 0.0f;
        box_r[2] = 0.0f;
        box_r[3] = 0.0f;
        // class_r[0] = 0;
        // area_r[0]  = 0.0f;
        keep_r[0] = 0.0f;
      } else {
        keep_r[0] = 1.0f;
        // rest of data field's value remains
      }
    }

    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};
template class NmsCoreVertex<half>;
template class NmsCoreVertex<float>;

template <typename T> class CalcNNZVertex : public Vertex {
  // Found the index of the i-th nonzero in a sample of N elements
public:
  Input<int> in_row_start;
  Input<int> in_row_end;
  Vector<Input<Vector<T>>> InputVector; // shape: [bs, N]
  Vector<Input<int>> i_tensor;          // shape: [bs]
  Input<int> length;                    // N
  Input<int> batch_size;                // bs

  Output<Vector<int>> ith_nonzero; // shape: [bs]  (index of i-th nonzero)

  // The compute method performs core computation
  bool compute() {
    bool found = false;
    int num_nonzeros_spotted = 0;
    int target = 0;

    for (unsigned sample = 0; sample < batch_size; sample++) {
      found = false;
      num_nonzeros_spotted = 0;
      int target_front = i_tensor[sample] - in_row_start;
      int target_behind = in_row_end - i_tensor[sample];
      if (target_front <= 0) {
        ith_nonzero[sample] = 0;
        return true;
      } else if (target_behind < 0) {
        ith_nonzero[sample] = length;
        return true;
      } else if (target_front >= 0 && target_behind >= 0) {
        target = i_tensor[sample] - in_row_start;
      }

      for (int idx = 0; idx < length; idx++) {
        // if(InputVector[sample][idx] != 0.0f)
        // {
        num_nonzeros_spotted++;

        if (num_nonzeros_spotted == target) {
          // printf("target:%d\n", target);
          // printf("i_tensor[sample] :%d\n", i_tensor[sample] );
          // printf("idx:%d\n", idx + 1);
          if (in_row_start == 0 && i_tensor[sample] <= length)
            ith_nonzero[sample] = idx;
          else
            ith_nonzero[sample] = idx;
          found = true;
          break;
        }
        if (found)
          break;
      }
    }

    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};
template class CalcNNZVertex<half>;
template class CalcNNZVertex<float>;

template <typename T> class PartialFetchVertex : public Vertex {
public:
  Input<int> in_row_start;
  Input<int> in_row_end;
  Vector<Input<Vector<T>>> in_tensor;
  Vector<Input<int>> j_tensor;

  Input<int> batch_size;
  Input<Vector<unsigned int>> sorted_index;
  Vector<Output<T>> out_val;

  bool compute() {
    for (unsigned sample = 0; sample < batch_size; sample++) {
      int j_idx = sorted_index[j_tensor[sample]];
      if (j_idx < in_row_start || j_idx > in_row_end) {
        out_val[sample] = 0.0f;
        continue;
      }
      out_val[sample] = in_tensor[sample][j_idx - in_row_start];
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};

template <typename T> class PartialFetchBoxVertex : public Vertex {
public:
  Input<int> in_row_start;
  Input<int> in_row_end;
  Vector<Input<Vector<T>>>
      in_tensor; // Per Vertex sees subtensor of shape [bs, (5*Top_n)*4]
  // Vector<Input<int>> j_tensor;            // Per Vertex sees subtensor of
  // shape [bs], value within [0, 5*top_n)
  Input<int> batch_size; // bs
  Input<int> length;     // Suppose to be 5*top_n

  Input<Vector<unsigned int>> sorted_index;
  Vector<Output<Vector<T>>>
      out_val; // Per Vertex fill sub-tensor of shape [bs, 4]

  bool compute() {
    int bs = batch_size;
    for (int sample = 0; sample < batch_size; sample++) {
      int j_idx = 4 * sorted_index[0];
      if (j_idx < in_row_start || j_idx > in_row_end) {
        out_val[sample][0] = 0.0f;
        out_val[sample][1] = 0.0f;
        out_val[sample][2] = 0.0f;
        out_val[sample][3] = 0.0f;
        continue;
      }
      int in_idx_0 = (j_idx - in_row_start);
      int in_idx_1 = (j_idx - in_row_start) + 1;
      int in_idx_2 = (j_idx - in_row_start) + 2;
      int in_idx_3 = (j_idx - in_row_start) + 3;

      out_val[sample][0] = in_tensor[sample][in_idx_0];
      out_val[sample][1] = in_tensor[sample][in_idx_1];
      out_val[sample][2] = in_tensor[sample][in_idx_2];
      out_val[sample][3] = in_tensor[sample][in_idx_3];
    }

    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};

template <typename T> class FetchVertex : public Vertex {
public:
  Vector<Input<Vector<T>>> in_tensor;
  Vector<Input<int>> j_tensor;
  Input<int> batch_size;

  Vector<Output<T>> out_val;

  bool compute() {
    for (unsigned sample = 0; sample < batch_size; sample++) {
      int j_idx = j_tensor[sample];
      out_val[sample] = in_tensor[sample][j_idx];
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};

class FetchBoxVertex : public Vertex {
public:
  Vector<Input<Vector<half>>>
      in_tensor; // Per Vertex sees subtensor of shape [bs, (5*Top_n)*4]
  Vector<Input<int>> j_tensor; // Per Vertex sees subtensor of shape [bs], value
                               // within [0, 5*top_n)
  Input<int> batch_size;       // bs
  Input<int> length;           // Suppose to be 5*top_n

  Vector<Output<Vector<half>>>
      out_val; // Per Vertex fill sub-tensor of shape [bs, 4]

  bool compute() {
    int bs = batch_size;
    for (int sample = 0; sample < batch_size; sample++) {
      int j_idx = j_tensor[sample];

      int in_idx_0 = j_idx * 4;
      int in_idx_1 = j_idx * 4 + 1;
      int in_idx_2 = j_idx * 4 + 2;
      int in_idx_3 = j_idx * 4 + 3;

      out_val[bs][0] = in_tensor[bs][in_idx_0];
      out_val[bs][1] = in_tensor[bs][in_idx_1];
      out_val[bs][2] = in_tensor[bs][in_idx_2];
      out_val[bs][3] = in_tensor[bs][in_idx_3];
    }

    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};

template class FetchVertex<half>;
template class FetchVertex<float>;
template class FetchVertex<int>;
template class PartialFetchVertex<half>;
template class PartialFetchVertex<float>;
template class PartialFetchVertex<int>;
template class PartialFetchBoxVertex<half>;
template class PartialFetchBoxVertex<float>;
// template class PartialFetchVertex<half>;

template <typename T> class CalcOnesVertex : public Vertex {
public:
  Vector<Input<Vector<T>>> InputVector; // [bs, N] shaped
  Input<int> length;                    // N
  Input<int> batch_size;                // bs

  Output<Vector<int>> v_num_ones; // [bs] shaped

  // The compute method performs core computation: Compute number of ones in
  // InputVector
  bool compute() {
    for (int sample = 0; sample < batch_size; sample++) {
      int num_ones = 0;
      for (int idx = 0; idx < length; idx++) {
        if (InputVector[sample][idx] == T(1.0f))
          num_ones += 1;
      }
      v_num_ones[sample] = num_ones;
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};
template class CalcOnesVertex<half>;
template class CalcOnesVertex<float>;

template <typename T> class CalcNzVertex : public Vertex {
public:
  Vector<Input<Vector<T>>> InputVector; // [bs, N] shaped
  Input<int> length;                    // N
  Input<int> batch_size;                // bs

  Output<Vector<int>> v_num_nonzero; // [bs] shaped

  // The compute method performs core computation
  bool compute() {
    for (int sample = 0; sample < batch_size; sample++) {
      int num_nonzeros = 0;
      for (int idx = 0; idx < length; idx++) {
        if (InputVector[sample][idx] != T(0.0))
          num_nonzeros += 1;
      }
      v_num_nonzero[sample] = num_nonzeros;
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};
template class CalcNzVertex<half>;
template class CalcNzVertex<float>;

template <typename T> class UpdateStateVertex : public Vertex {
public:
  // poplar::Vector<poplar::Input<Vector<bool>>>  keep;
  // Input<Vector<int>> num_ones_in_keep;       // [bs] shaped
  Input<Vector<int>> num_nonzeros_in_scores; // [bs] shaped
  Input<int> batch_size;                     //  bs
  // Vector<Input<unsigned int>> sorted_index;
  InOut<Vector<int>> iTensor; // [bs] shaped
  Output<Vector<int>> finish; // [bs] shaped

  // The compute method performs core computation
  bool compute() {
    for (int sample = 0; sample < batch_size; sample++) {
      int i = iTensor[sample];
      // int keep_num_ones = num_ones_in_keep[sample];
      int scores_num_nonzeros = num_nonzeros_in_scores[sample];

      // if(i >= keep_num_ones || i >= scores_num_nonzeros)
      // control the loop times
      if (i >= scores_num_nonzeros) {
        finish[sample] = 1;
        break;
      } else
        iTensor[sample] = (i + 1);
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const { return 50; }
};
template class UpdateStateVertex<float>;
template class UpdateStateVertex<half>;
