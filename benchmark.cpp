// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <lyra/lyra.hpp>

#include "ipu_utils.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <poplar/CycleCount.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

#include <popnn/Loss.hpp>
#include <popnn/codelets.hpp>

#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

#include "nms.hpp"
#include "reference.hpp"
#include "utils.hpp"

#include <unordered_set>

using namespace poplar;

float computeError(const std::vector<float> &v1, const std::vector<float> &v2) {
  assert(v1.size() == v2.size());
  double errSum = 0.0, sum1 = 0.0, sum2 = 0.0;
  for (uint32_t i = 0; i < v1.size(); ++i) {
    double err = double(v1[i]) - double(v2[i]);
    errSum += err * err;
    sum1 += double(v1[i]) * double(v1[i]);
    sum2 += double(v2[i]) * double(v2[i]);
  }
  return errSum / (std::sqrt(sum1 * sum2) + 1.0e-8);
}

float computeError(const std::vector<uint32_t> &v1,
                   const std::vector<uint32_t> &v2) {
  assert(v1.size() == v2.size());
  uint32_t correct = 0;
  for (uint32_t i = 0; i < v1.size(); ++i) {
    if (v1[i] == v2[i]) {
      ++correct;
    }
  }
  return float(correct) / float(v1.size());
}

float computeError(const std::vector<uint32_t> &v1,
                   const std::vector<uint32_t> &v2, uint32_t bs, uint32_t N) {
  Shape s{bs, N};
  NDArray<uint32_t> nd1{s, v1}, nd2{s, v2};
  // std::cerr << "ND1 ";
  // nd1.print(std::cerr);
  // std::cerr << "ND2 ";
  // nd2.print(std::cerr);

  assert(v1.size() == v2.size());
  uint32_t correct = 0;
  std::vector<std::unordered_set<uint32_t>> sets1(bs), sets2(bs);
  for (size_t b = 0; b < bs; ++b) {
    for (size_t i = 0; i < N; ++i) {
      sets1[b].insert(nd1[{b, i}]);
      sets2[b].insert(nd2[{b, i}]);
    }
  }
  for (size_t b = 0; b < bs; ++b) {
    const auto &sets2_b = sets2[b];
    for (const auto &e : sets1[b]) {
      if (sets2_b.find(e) != sets2_b.end())
        ++correct;
    }
  }
  return float(correct) / float(v1.size());
}

class Benchmark {
  DataGenerator gen_;
  uint32_t N_;
  uint32_t K_;
  float threshold_;
  float scoreThreshold_;
  float sigma_;
  uint32_t numClasses_;
  uint32_t batchSize_;
  bool fp16_;
  uint32_t tiles_;
  bool verif_;
  bool measureCycles_;
  bool useGather_;
  bool large_;
  uint32_t topk_;

  Device device_;
  Target target_;
  Graph graph_;

  std::vector<float> scores_;
  std::vector<float> boxes_;
  std::vector<uint32_t> classes_;
  std::vector<uint32_t> refOutput_;

  Tensor scoresT_;
  Tensor boxesT_;
  Tensor classesT_;
  Tensor outputT_;

private:
  void prepareHostData() {
    scores_ =
        gen_.generate(N_ * batchSize_ * std::max(numClasses_, uint32_t(1)));
    boxes_.reserve(batchSize_ * N_ * 4);
    for (size_t i = 0; i < batchSize_; ++i) {
      for (size_t j = 0; j < N_; ++j) {
        float x1 = gen_.generate();
        float y1 = gen_.generate();
        boxes_.push_back(x1);
        boxes_.push_back(y1);
        boxes_.push_back(x1 + std::abs(gen_.generate()));
        boxes_.push_back(y1 + std::abs(gen_.generate()));
      }
    }
    classes_ = gen_.labels(N_ * batchSize_);
    if (verif_) {
      assert(numClasses_ >= 1);
      Shape boxesS{batchSize_, N_, 4};
      NDArray<float> boxesA(boxesS, boxes_);
      if (numClasses_ > 1) {
        Shape scoresS{batchSize_, N_, numClasses_};
        NDArray<float> scoresA(scoresS, scores_);
        auto output = NmsMulti(scoresA, boxesA, K_, threshold_);
        refOutput_ = output.copy_data();
      } else {
        Shape scoresS{batchSize_, N_};
        NDArray<float> scoresA(scoresS, scores_);
        NDArray<uint32_t> classesA(scoresS, classes_);
        auto output = Nms(scoresA, boxesA, classesA, K_, threshold_);
        refOutput_ = output.copy_data();
      }
    }
  }

  template <typename T>
  void initConstant(poplar::Tensor &t, const T &v, const poplar::Type &type,
                    const poplar::ArrayRef<std::size_t> &shape,
                    uint32_t grainSize, const std::string &name) {
    t = graph_.addConstant(type, shape, v, name);
    poputil::mapTensorLinearly(graph_, t, 1, grainSize);
  }

  template <typename T>
  void initConstant(poplar::Tensor &t, const std::vector<T> &v,
                    const poplar::Type &type,
                    const poplar::ArrayRef<std::size_t> &shape,
                    uint32_t grainSize, const std::string &name) {
    ArrayRef<T> ref{v};
    t = graph_.addConstant(type, shape, ref, name);
    poputil::mapTensorLinearly(graph_, t, 1, grainSize);
  }

  template <typename T>
  void initData(poplar::Tensor &t, const std::vector<T> &v,
                const poplar::Type &type,
                const poplar::ArrayRef<std::size_t> &shape, uint32_t grainSize,
                const std::string &name) {
    ArrayRef<T> ref{v};
    t = graph_.addVariable(type, shape, name);
    poputil::mapTensorLinearly(graph_, t, 1, grainSize);
    graph_.setInitialValue(t, ref);
  }

  void prepareIpuData() {
    Type dataType = fp16_ ? poplar::HALF : poplar::FLOAT;
    initData<float>(boxesT_, boxes_, dataType, {batchSize_, N_, 4}, 4, "boxes");
    if (numClasses_ > 1) {
      initData<float>(scoresT_, scores_, dataType,
                      {batchSize_, N_, numClasses_}, numClasses_, "scores");
    } else {
      initData<float>(scoresT_, scores_, dataType, {batchSize_, N_}, 1,
                      "scores");
      initData(classesT_, classes_, poplar::INT, {batchSize_, N_}, 1,
               "classes");
    }
  }

  void buildGraph(poplar::program::Sequence &prog) {
    poplar::program::Sequence nmsSeq;
    poplar::Tensor scores, boxes, classes, lengths;
    if (large_) {
      std::tie(outputT_, scores, boxes, classes, lengths) = nmsMultiLarge(
          graph_, nmsSeq, scoresT_, boxesT_, threshold_, K_, scoreThreshold_,
          useGather_, topk_, "nmsMultiLargeBenchmark");
    } else {
      if (numClasses_ > 1) {
        std::tie(outputT_, scores, boxes, classes, lengths) =
            nmsMulti(graph_, nmsSeq, scoresT_, boxesT_, threshold_, K_,
                     scoreThreshold_, sigma_, useGather_, "nmsMultiBenchmark");

      } else {
        if (numClasses_ == 1) {
          std::tie(outputT_, scores, boxes, classes, lengths) =
              nms(graph_, nmsSeq, scoresT_, boxesT_, classesT_, threshold_, K_,
                  scoreThreshold_, sigma_, useGather_, "nmsBenchmark");
        } else {
          std::tie(outputT_, scores, boxes, lengths) =
              nms(graph_, nmsSeq, scoresT_, boxesT_, threshold_, K_,
                  scoreThreshold_, sigma_, useGather_, "nmsBenchmark");
        }
      }
    }
    if (measureCycles_) {
      auto cycles =
          poplar::cycleCount(graph_, nmsSeq, 0, poplar::SyncType::EXTERNAL);
      prog.add(nmsSeq);
      prog.add(poplar::program::PrintTensor("NMS cycles", cycles));
    } else {
      prog.add(nmsSeq);
    }
    if (verif_) {
      graph_.createHostRead("output-read", outputT_);
    }
  }
  template <typename T>
  void readTensor(Engine &engine, const std::string &name, std::vector<T> &v) {
    engine.readTensor(name, v.data(), v.data() + v.size() * sizeof(T));
  }

public:
  Benchmark(uint32_t N, uint32_t K, float threshold, float scoreThreshold,
            float sigma, uint32_t classes, uint32_t batchSize, bool fp16,
            uint32_t tiles, bool verif, bool measureCycles, bool useGather,
            bool large, uint32_t topk)
      : gen_{1.0, classes}, N_{N}, K_{K}, threshold_{threshold},
        scoreThreshold_{scoreThreshold}, sigma_{sigma}, numClasses_{classes},
        batchSize_{batchSize}, fp16_{fp16}, tiles_{tiles}, verif_{verif},
        measureCycles_(measureCycles),
        useGather_{useGather}, large_{large}, topk_{topk}, device_{getDevice()},
        target_{device_.getTarget()}, graph_{target_} {
    if (tiles_ > 0) {
      device_ = device_.createVirtualDevice(tiles_);
      target_ = device_.getTarget();
      graph_ = Graph{target_};
    }
    popops::addCodelets(graph_);
    graph_.addCodelets("codelet.cpp");
    popnn::addCodelets(graph_);

    prepareHostData();
    prepareIpuData();
  }

  void run() {
    program::Sequence prog;
    buildGraph(prog);
    Engine engine(graph_, prog);
    engine.load(device_);
    auto start = std::chrono::high_resolution_clock::now();
    engine.run();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Elapsed time: " << elapsed.count() * 1000.0 << " ms\n";
    if (verif_) {
      std::vector<uint32_t> output;
      output.resize(refOutput_.size());
      readTensor(engine, "output-read", output);
      std::cerr << "Output error = " << computeError(output, refOutput_)
                << std::endl;
      std::cerr << "Inclusion error = "
                << computeError(output, refOutput_, batchSize_, K_)
                << std::endl;
      std::cerr << "ref " << to_string(refOutput_) << std::endl;
      std::cerr << "ipu " << to_string(output) << std::endl;
    }
  }
};

int main(int argc, const char **argv) {
  uint32_t N = 10000;
  uint32_t classes = 1;
  uint32_t num_detections = 300;
  uint32_t topk = 1;
  float threshold = 0.5;
  float score_threshold = 0.0;
  float sigma = 0.0;
  uint32_t tiles = 0;
  uint32_t batch_size = 1;
  bool show_help = false;
  bool fp16 = false;
  bool verif = false;
  bool measureCycles = false;
  bool useGather = false;
  bool large = false;
  auto cli =
      lyra::help(show_help) |
      lyra::opt(N, "N")["-N"]("Number of scores/boxes.") |
      lyra::opt(num_detections, "K")["-K"]("Number of detections.") |
      lyra::opt(classes, "C")["-C"]("Number of classes.") |
      lyra::opt(tiles, "t")["-t"]("Number of tiles on the IPU.") |
      lyra::opt(threshold, "T")["-T"]("Threshold.") |
      lyra::opt(score_threshold, "S")["-S"]("Score threshold.") |
      lyra::opt(sigma, "s")["-s"]("Sigma.") |
      lyra::opt(fp16, "0|1")["-F"]("Use fp16.") |
      lyra::opt(useGather, "0|1")["-G"]("Use gather.") |
      lyra::opt(large, "0|1")["-l"]("Use large nms.") |
      lyra::opt(topk, "k")["-k"]("Number of topk for large nms.") |
      lyra::opt(verif, "0|1")["-v"](
          "Verification of results against reference version.") |
      lyra::opt(measureCycles,
                "0|1")["-m"]("Measure on device cycles around NMS.") |
      lyra::opt(batch_size, "batch size")["-b"]["--batch_size"]("Batch size.");
  auto result = cli.parse({argc, argv});
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage() << std::endl
              << cli << "\n";
    return 1;
  }
  if (show_help) {
    std::cout << cli << "\n";
    return 0;
  }
  std::cout << "N=" << N << "\tK=" << num_detections << "\tT=" << threshold
            << "\tC=" << classes << "\tbatch size=" << batch_size
            << "\tfp16=" << fp16 << "\n";
  Benchmark benchmark{N,
                      num_detections,
                      threshold,
                      score_threshold,
                      sigma,
                      classes,
                      batch_size,
                      fp16,
                      tiles,
                      verif,
                      measureCycles,
                      useGather,
                      large,
                      topk};
  benchmark.run();
  return 0;
}
