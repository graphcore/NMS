// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ipu_test.hpp"
#include "reference.hpp"
#include "utils.hpp"
// #include <popops/codelets.hpp>
#include "ipu_utils.hpp"
#include "nms.hpp"
#include <popnn/Loss.hpp>
#include <popnn/codelets.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Gather.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <random>
#include <vector>
#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>
#include <poplar/CycleCount.hpp>
#include <popops/Reduce.hpp>

poplar::Device device = getDevice();
poplar::Target target = device.getTarget();

TEST_CASE("sort", "[sort]") {
  NDArray<float> scores{{4, 16}};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      scores[{i, j}] = float(j);
    }
  }
  auto index = sort_indices(scores);
  REQUIRE_THAT(index.data(),
               Catch::Equals<uint32_t>(
                   {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

TEST_CASE("argmax", "[argmax]") {
  NDArray<float> scores{{4, 16}};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      scores[{i, j}] = float(j);
    }
  }
  auto index = argmax(scores);
  REQUIRE_THAT(index.data(), Catch::Equals<uint32_t>({15, 15, 15, 15}));
}

TEST_CASE("area", "[area]") {
  NDArray<float> boxes{{4, 16, 4}};
  std::mt19937 eng(42);
  std::uniform_real_distribution<float> dist(0.0, 0.5);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      float x1 = dist(eng);
      float y1 = dist(eng);
      boxes[{i, j, 0}] = x1;
      boxes[{i, j, 1}] = y1;
      boxes[{i, j, 2}] = x1 + 0.5;
      boxes[{i, j, 3}] = y1 + 0.5;
    }
  }
  auto areas = compute_area(boxes);
  REQUIRE_THAT(
      areas.data(),
      Catch::Approx<float>(
          {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
           0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
           0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
           0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
           0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
           0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
           0.25f, 0.25f, 0.25f, 0.25f}));
}

std::pair<poplar::Tensor, poplar::Tensor>
reducemax(poplar::Graph &graph, program::Sequence &prog, poplar::Tensor &inputs,
          poplar::Tensor &indices, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(inputs, indices));
  const size_t batchSize = inputs.dim(0);
  std::vector<std::vector<poplar::Tensor>> batchMax(batchSize),
      batchIndices(batchSize);
  std::vector<size_t> baseShape = inputs.shape();
  const auto baseMapping = graph.getTileMapping(inputs);
  size_t index = 0;
  for (size_t i = 0; i < baseMapping.size(); ++i) {
    if (baseMapping[i].size() > 0) {
      assert(baseMapping[i].size() == 1);
      std::vector<size_t> indices_ =
          poputil::unflattenIndex(baseShape, baseMapping[i].front().begin());
      size_t batch = indices_[0];
      batchIndices[batch].push_back(indices.flatten()[index].reshape({1}));
      batchMax[batch].push_back(inputs.flatten()[index].reshape({1}));
      ++index;
    }
  }
  poplar::Tensor max =
      graph.addVariable(inputs.elementType(), {inputs.dim(0)}, {di, "myMax"});
  graph.setTileMapping(max, 1);
  poplar::Tensor argmax =
      graph.addVariable(indices.elementType(), {batchSize}, {di, "myArgmax"});
  graph.setTileMapping(argmax, 1);

  poplar::ComputeSet reduceCs = graph.addComputeSet({dc, "reduceMaxArgmax"});
  for (size_t i = 0; i < batchSize; ++i) {
    VertexRef vtx = graph.addVertex(
        reduceCs, poputil::templateVertex("MaxReduceVertex", max.elementType(),
                                          argmax.elementType()));
    poplar::Tensor maxIndex_i = poplar::concat(batchIndices[i]);
    poplar::Tensor max_i = poplar::concat(batchMax[i]);
    graph.setTileMapping(max_i, 1 + i);
    graph.setTileMapping(maxIndex_i, 1 + i);
    graph.connect(vtx["indices"], maxIndex_i);
    graph.connect(vtx["input"], max_i);
    graph.connect(vtx["max"], max[i]);
    graph.connect(vtx["argmax"], argmax[i]);
    graph.setPerfEstimate(vtx, indices.dim(1) + 1);
    graph.setTileMapping(vtx, 1 + i);
  }
  prog.add(program::Execute(reduceCs));
  return {max, argmax};
}

TEST_CASE("maxArgmax1", "[maxArgmax]") {
  NDArray<float> inputs{{1, 1472}};
  NDArray<uint32_t> indices{{1, 1472}};
  for (size_t i = 0; i < 1472; ++i) {
    indices[{0, i}] = i + 10;
    inputs[{0, i}] = float(i);
  }
  Graph graph{target};
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  graph.addCodelets("codelet.cpp");
  poplar::program::Sequence prog;
  poplar::Tensor inputsIPU =
      graph.addVariable(poplar::FLOAT, {1, 1472}, "inputs");
  graph.setInitialValue(inputsIPU, poplar::ArrayRef<float>{inputs.data()});
  poplar::Tensor indicesIPU =
      graph.addVariable(poplar::UNSIGNED_INT, {1, 1472}, "indices");
  graph.setInitialValue(indicesIPU, poplar::ArrayRef<uint32_t>{indices.data()});
  poputil::mapTensorLinearly(graph, inputsIPU, 1, 1);
  poputil::mapTensorLinearly(graph, indicesIPU, 1, 1);
  poplar::Tensor max1, argmax1;
  std::tie(max1, argmax1) =
      localMaxAndArgMax(graph, inputsIPU, indicesIPU, inputsIPU.elementType(),
                        prog, {"popnnVersion"});
  // auto cycle = poplar::cycleCount(graph, prog, 0,
  // poplar::SyncType::INTERNAL); prog.add(program::PrintTensor("cycle",
  // cycle));
  prog.add(program::PrintTensor("max", max1));
  prog.add(program::PrintTensor("argmax", argmax1));
  Engine engine{
      graph, prog, {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
  engine.load(device);
  engine.run(0);
}

TEST_CASE("maxArgmax2", "[maxArgmax]") {
  NDArray<float> inputs{{1, 1472}};
  NDArray<uint32_t> indices{{1, 1472}};
  for (size_t i = 0; i < 1472; ++i) {
    indices[{0, i}] = i;
    inputs[{0, i}] = float(i);
  }
  Graph graph{target};
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  graph.addCodelets("codelet.cpp");
  poplar::program::Sequence prog;
  poplar::Tensor inputsIPU =
      graph.addVariable(poplar::FLOAT, {1, 1472}, "inputs");
  graph.setInitialValue(inputsIPU, poplar::ArrayRef<float>{inputs.data()});
  poplar::Tensor indicesIPU =
      graph.addVariable(poplar::UNSIGNED_INT, {1, 1472}, "indices");
  graph.setInitialValue(indicesIPU, poplar::ArrayRef<uint32_t>{indices.data()});
  poputil::mapTensorLinearly(graph, inputsIPU, 1, 1);
  poputil::mapTensorLinearly(graph, indicesIPU, 1, 1);
  poplar::Tensor max2, argmax2;
  std::tie(max2, argmax2) =
      reducemax(graph, prog, inputsIPU, indicesIPU, {"myVersion"});
  // auto cycle = poplar::cycleCount(graph, prog, 0,
  // poplar::SyncType::INTERNAL); prog.add(program::PrintTensor("cycle",
  // cycle));
  // prog.add(program::PrintTensor("max2", max2));
  // prog.add(program::PrintTensor("argmax2", argmax2));
  // printMapping(graph.getTileMapping(inputsIPU));
  // printMapping(graph.getTileMapping(indicesIPU));
  Engine engine{
      graph, prog, {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
  engine.load(device);
  engine.run(0);
}

TEST_CASE("maxArgmax3", "[maxArgmax]") {
  NDArray<float> inputs{{1, 1472}};
  NDArray<uint32_t> indices{{1, 1472}};
  for (size_t i = 0; i < 1472; ++i) {
    indices[{0, i}] = i;
    inputs[{0, i}] = float(i);
  }
  Graph graph{target};
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  graph.addCodelets("codelet.cpp");
  poplar::program::Sequence prog;
  poplar::Tensor inputsIPU =
      graph.addVariable(poplar::FLOAT, {1, 1472}, "inputs");
  graph.setInitialValue(inputsIPU, poplar::ArrayRef<float>{inputs.data()});
  poplar::Tensor indicesIPU =
      graph.addVariable(poplar::UNSIGNED_INT, {1, 1472}, "indices");
  graph.setInitialValue(indicesIPU, poplar::ArrayRef<uint32_t>{indices.data()});
  poputil::mapTensorLinearly(graph, inputsIPU, 1, 1);
  poputil::mapTensorLinearly(graph, indicesIPU, 1, 1);
  popops::ReduceParams params(popops::Operation::MAX);
  poplar::Tensor max3 = popops::reduce(graph, inputsIPU, poplar::FLOAT, {1},
                                       params, prog, {"reducePoplibs"});
  // auto cycle = poplar::cycleCount(graph, prog, 0,
  // poplar::SyncType::INTERNAL); prog.add(program::PrintTensor("cycle",
  // cycle));
  // prog.add(program::PrintTensor("max3", max3));
  Engine engine{
      graph, prog, {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
  engine.load(device);
  engine.run(0);
}

TEST_CASE("maxArgmax4", "[maxArgmax]") {
  NDArray<float> inputs{{1, 1472}};
  NDArray<uint32_t> indices{{1, 1472}};
  for (size_t i = 0; i < 1472; ++i) {
    indices[{0, i}] = i;
    inputs[{0, i}] = float(i);
  }
  Graph graph{target};
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  graph.addCodelets("codelet.cpp");
  poplar::program::Sequence prog;
  poplar::Tensor inputsIPU =
      graph.addVariable(poplar::FLOAT, {1, 1472}, "inputs");
  graph.setInitialValue(inputsIPU, poplar::ArrayRef<float>{inputs.data()});
  poplar::Tensor indicesIPU =
      graph.addVariable(poplar::UNSIGNED_INT, {1, 1472}, "indices");
  graph.setInitialValue(indicesIPU, poplar::ArrayRef<uint32_t>{indices.data()});
  poputil::mapTensorLinearly(graph, inputsIPU, 1, 1);
  poputil::mapTensorLinearly(graph, indicesIPU, 1, 1);
  poplar::Tensor max1, argmax1;
  std::tie(max1, argmax1) =
      popnn::maxAndArgMax(graph, inputsIPU, prog, {"popnnVersion"});
  // auto cycle = poplar::cycleCount(graph, prog, 0,
  // poplar::SyncType::INTERNAL); prog.add(program::PrintTensor("cycle",
  // cycle));
  // prog.add(program::PrintTensor("max", max1));
  // prog.add(program::PrintTensor("argmax", argmax1));
  Engine engine{
      graph, prog, {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
  engine.load(device);
  engine.run(0);
}

TEST_CASE("nms", "[nms]") {
  std::vector<float> boxesData{0, 0,     1, 1,     0, 0.1f, 1, 1.1f,
                               0, -0.1f, 1, 0.9f,  0, 10,   1, 11,
                               0, 10.1f, 1, 11.1f, 0, 100,  1, 101};
  std::vector<float> scoresData{.9f, .75f, .6f, .95f, .5f, .3f};
  NDArray<uint32_t> classes{{1, 6}, 1};

  SECTION("Reference") {
    NDArray<float> boxes{{1, 6, 4}, boxesData};
    NDArray<float> scores{{1, 6}, scoresData};
    auto res = Nms(scores, boxes, classes, 3, 0.5);
    res.print(std::cerr);
    REQUIRE_THAT(res.data(), Catch::Equals<uint32_t>({3, 0, 5}));
  }
  SECTION("IPU fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {1, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {1, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {1, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) =
        nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 3);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData, Catch::Equals<int32_t>({3, 0, 5}));
  }
}

TEST_CASE("nms bs2", "[nms]") {
  std::vector<float> boxesData{
      0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
      0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101,

      0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
      0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101};
  std::vector<float> scoresData{.9f, .75f, .6f, .95f, .5f, .3f,
                                .9f, .75f, .6f, .95f, .5f, .3f};
  NDArray<uint32_t> classes{{2, 6}, 1};

  SECTION("Reference") {
    NDArray<float> boxes{{2, 6, 4}, boxesData};
    NDArray<float> scores{{2, 6}, scoresData};
    auto res = Nms(scores, boxes, classes, 3, 0.5);
    REQUIRE_THAT(res.data(), Catch::Equals<uint32_t>({3, 0, 5, 3, 0, 5}));
    auto res2 = Nms(scores, boxes, classes, 5, 0.5);
    REQUIRE_THAT(res2.data(),
                 Catch::Equals<uint32_t>({3, 0, 5, 4294967295, 4294967295, 3, 0,
                                          5, 4294967295, 4294967295}));
  }
  SECTION("IPU fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {2, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) =
        nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 3);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData, Catch::Equals<int32_t>({3, 0, 5, 3, 0, 5}));
  }

  SECTION("IPU fp32 k=5") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {2, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) =
        nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 5);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    ReadHandle<uint32_t> classesRead{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 5, -1, -1, 3, 0, 5, -1, -1}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({3, 3}));
    const auto &classesAnsData = classesRead.read(engine);
    REQUIRE_THAT(classesAnsData,
                 Catch::Equals<uint32_t>({1, 1, 1, 2147483647, 2147483647, 1, 1,
                                          1, 2147483647, 2147483647}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(scoresAnsData,
                 Catch::Equals<float>({0.95f, 0.9f, 0.3f, 0.0f, 0.0f, 0.95f,
                                       0.9f, 0.3f, 0.0f, 0.0f}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(
        boxesAnsData,
        Catch::Equals<float>(
            {0.0f, 10.0f,  1.0f, 11.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 100.0f,
             1.0f, 101.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 10.0f,  1.0f, 11.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 100.0f,
             1.0f, 101.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
  }

  SECTION("IPU fp32 score_threshold") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {2, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) =
        nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 5, 0.5);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    ReadHandle<uint32_t> classesRead{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, -1, -1, -1, 3, 0, -1, -1, -1}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({2, 2}));
    const auto &classesAnsData = classesRead.read(engine);
    REQUIRE_THAT(
        classesAnsData,
        Catch::Equals<uint32_t>({1, 1, 2147483647, 2147483647, 2147483647, 1, 1,
                                 2147483647, 2147483647, 2147483647}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(scoresAnsData,
                 Catch::Equals<float>({0.95f, 0.9f, 0.0f, 0.0f, 0.0f, 0.95f,
                                       0.9f, 0.0f, 0.0f, 0.0f}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(
        boxesAnsData,
        Catch::Equals<float>(
            {0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
  }

  SECTION("IPU fp32 soft-nms") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {2, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) =
        nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 6, 0.0, 0.5);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    ReadHandle<uint32_t> classesRead{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    /*  tensorflow values.
        self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
        self.assertAllClose(selected_scores,
                            [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                            rtol=1e-2, atol=1e-2)

     */
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 5, 1, 4, 2, 3, 0, 5, 1, 4, 2}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({6, 6}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(
        scoresAnsData,
        Catch::Approx<float>({0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636,
                              0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636}));
    const auto &classesAnsData = classesRead.read(engine);
    REQUIRE_THAT(classesAnsData,
                 Catch::Equals<uint32_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(boxesAnsData,
                 Catch::Equals<float>(
                     {0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9,
                      0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9}));
  }
  SECTION("IPU fp32 soft-nms gather") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {2, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) = nms(
        graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 6, 0.0, 0.5, true);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    ReadHandle<uint32_t> classesRead{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    /*  tensorflow values.
        self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
        self.assertAllClose(selected_scores,
                            [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                            rtol=1e-2, atol=1e-2)

     */
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 5, 1, 4, 2, 3, 0, 5, 1, 4, 2}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({6, 6}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(
        scoresAnsData,
        Catch::Approx<float>({0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636,
                              0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636}));
    const auto &classesAnsData = classesRead.read(engine);
    REQUIRE_THAT(classesAnsData,
                 Catch::Equals<uint32_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(boxesAnsData,
                 Catch::Equals<float>(
                     {0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9,
                      0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9}));
  }
  SECTION("IPU fp32 soft-nms gather inplace") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});
    poplar::Tensor classesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {2, 6}, "classes");
    graph.setInitialValue(classesIPU,
                          poplar::ArrayRef<uint32_t>{classes.data()});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) =
        nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 6, 0.0, 0.5,
            true, true);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    ReadHandle<uint32_t> classesRead{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    /*  tensorflow values.
        self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
        self.assertAllClose(selected_scores,
                            [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                            rtol=1e-2, atol=1e-2)

     */
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 5, 1, 4, 2, 3, 0, 5, 1, 4, 2}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({6, 6}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(
        scoresAnsData,
        Catch::Approx<float>({0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636,
                              0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636}));
    const auto &classesAnsData = classesRead.read(engine);
    REQUIRE_THAT(classesAnsData,
                 Catch::Equals<uint32_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(boxesAnsData,
                 Catch::Equals<float>(
                     {0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9,
                      0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9}));
  }
  SECTION("IPU fp32 class-less soft-nms") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    poplar::Tensor scoresAns, boxesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, lengthsAns) =
        nms(graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0, 0.5);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    /*  tensorflow values.
        self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
        self.assertAllClose(selected_scores,
                            [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                            rtol=1e-2, atol=1e-2)

     */
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 5, 1, 4, 2, 3, 0, 5, 1, 4, 2}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({6, 6}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(
        scoresAnsData,
        Catch::Approx<float>({0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636,
                              0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(boxesAnsData,
                 Catch::Equals<float>(
                     {0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9,
                      0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9}));
  }
  SECTION("IPU fp32 class-less soft-nms gather") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    poplar::Tensor scoresAns, boxesAns, indicesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, lengthsAns) =
        nms(graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0, 0.5, true);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> lengths{lengthsAns, "lengths-read", graph};
    ReadHandle<float> scoresRead{scoresAns, "scores-read", graph};
    ReadHandle<float> boxesRead{boxesAns, "boxes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    /*  tensorflow values.
        self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
        self.assertAllClose(selected_scores,
                            [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                            rtol=1e-2, atol=1e-2)

     */
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 5, 1, 4, 2, 3, 0, 5, 1, 4, 2}));
    const auto &lengthsData = lengths.read(engine);
    REQUIRE_THAT(lengthsData, Catch::Equals<int32_t>({6, 6}));
    const auto &scoresAnsData = scoresRead.read(engine);
    REQUIRE_THAT(
        scoresAnsData,
        Catch::Approx<float>({0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636,
                              0.95, 0.9, 0.3, 0.196612, 0.131075, 0.0646636}));
    const auto &boxesAnsData = boxesRead.read(engine);
    REQUIRE_THAT(boxesAnsData,
                 Catch::Equals<float>(
                     {0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9,
                      0, 10,  1, 11,  0, 0,    1, 1,    0, 100,  1, 101,
                      0, 0.1, 1, 1.1, 0, 10.1, 1, 11.1, 0, -0.1, 1, 0.9}));
  }
}

TEST_CASE("gather scores", "[gather]") {
  NDArray<float> scores{{4, 16}};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      scores[{i, j}] = float(j);
    }
  }
  SECTION("IPU fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {4, 16}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scores.data()});
    graph.setTileMapping(scoresIPU, 0);
    std::vector<uint32_t> indices{0, 0, 1, 1, 2, 2, 3, 3};
    poplar::Tensor indicesIPU =
        graph.addVariable(poplar::UNSIGNED_INT, {4, 2}, "indices");
    graph.setInitialValue(indicesIPU, poplar::ArrayRef<uint32_t>{indices});
    graph.setTileMapping(indicesIPU, 0);
    prog.add(program::PrintTensor("scores", scoresIPU));
    prog.add(program::PrintTensor("indices", indicesIPU));
    // auto ipu_res = // core dump
    //     popops::gather(graph, scoresIPU, indicesIPU, 1 /* ?  */, {}, {1, 1},
    //                    {0, 1, 2}, {0, 1}, prog, "gatherIndices");
    auto ipu_res =
        popops::multiSlice(graph, scoresIPU, indicesIPU, {0, 1}, {1, 1}, prog,
                           popops::SlicePlan(), {}, "multiSliceIndices");
    prog.add(program::PrintTensor("res", ipu_res));
    ReadHandle<float> res{ipu_res, "indexes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = res.read(engine);
    REQUIRE_THAT(indicesData, Catch::Equals<float>({0, 1, 2, 3}));
  }
}

TEST_CASE("nmsmulti", "[nmsmulti]") {
  std::vector<float> boxesData{0, 0,     1, 1,     0, 0.1f, 1, 1.1f,
                               0, -0.1f, 1, 0.9f,  0, 10,   1, 11,
                               0, 10.1f, 1, 11.1f, 0, 100,  1, 101};
  std::vector<float> scoresData{.9f,  .1f,  .25f, .75f, .4f, .6f,
                                .95f, .05f, .5f,  .5f,  .3f, .7f};

  SECTION("Reference") {
    NDArray<float> boxes{{1, 6, 4}, boxesData};
    NDArray<float> scores{{1, 6, 2}, scoresData};
    auto res = NmsMulti(scores, boxes, 6, 0.5);
    res.print(std::cerr);
    REQUIRE_THAT(res.data(),
                 Catch::Equals<uint32_t>({3, 0, 0, 0, 1, 1, 5, 1, 4, 1, 5, 0}));
  }
  SECTION("IPU fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {1, 6, 2}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {1, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) =
        nmsMulti(graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> classes{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData, Catch::Equals<int32_t>({3, 0, 1, 5, 4, 5}));
    const auto &classesData = classes.read(engine);
    REQUIRE_THAT(classesData, Catch::Equals<int32_t>({0, 0, 1, 1, 1, 0}));
  }
}

TEST_CASE("nmsmulti bs2", "[nmsmulti]") {
  std::vector<float> boxesData{
      0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
      0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101,

      0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
      0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101};
  std::vector<float> scoresData{
      .9f, .1f, .25f, .75f, .4f, .6f, .95f, .05f, .5f, .5f, .3f, .7f,

      .9f, .1f, .25f, .75f, .4f, .6f, .95f, .05f, .5f, .5f, .3f, .7f};

  SECTION("Reference") {
    NDArray<float> boxes{{2, 6, 4}, boxesData};
    NDArray<float> scores{{2, 6, 2}, scoresData};
    auto res = NmsMulti(scores, boxes, 6, 0.5);
    res.print(std::cerr);
    REQUIRE_THAT(res.data(),
                 Catch::Equals<uint32_t>({3, 0, 0, 0, 1, 1, 5, 1, 4, 1, 5, 0,
                                          3, 0, 0, 0, 1, 1, 5, 1, 4, 1, 5, 0}));
  }
  SECTION("IPU fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 2}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) =
        nmsMulti(graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> classes{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5}));
    const auto &classesData = classes.read(engine);
    REQUIRE_THAT(classesData,
                 Catch::Equals<int32_t>({0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0}));
  }
  SECTION("IPU useGather fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 2}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) =
        nmsMulti(graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0, 0.0, true);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> classes{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5}));
    const auto &classesData = classes.read(engine);
    REQUIRE_THAT(classesData,
                 Catch::Equals<int32_t>({0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0}));
  }
  SECTION("IPU useGather fp32 inPlace") {
    Graph graph{target};
    poplar::program::Sequence prog;
    popops::addCodelets(graph);
    popnn::addCodelets(graph);
    graph.addCodelets("codelet.cpp");
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 2}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {2, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) = nmsMulti(
        graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0, 0.0, true, true);
    ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
    ReadHandle<int32_t> classes{classesAns, "classes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData,
                 Catch::Equals<int32_t>({3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5}));
    const auto &classesData = classes.read(engine);
    REQUIRE_THAT(classesData,
                 Catch::Equals<int32_t>({0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0}));
  }
}

TEST_CASE("topk1", "[tournament]") {
  Graph graph{target};
  graph.addCodelets("codelet.gp");
  SECTION("test k=1") {
    const uint32_t N = 2, C = 3, K = 1;
    Variable<float> input{"input", {N * C}, {0.0, 1.0, 2.0, 5.0, 3.0, 4.0}};
    Variable<unsigned short> topk{"topk", {N * K}};
    Codelet codelet{
        "TopKVertex", graph, {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.initialValue("score_threshold", float(1.0));
    codelet.loadAndRun(device);
    REQUIRE_THAT(topk.data(), Catch::Equals<unsigned short>({2, 0}));
  }
  SECTION("test k=1 no val") {
    const uint32_t N = 2, C = 3, K = 1;
    Variable<float> input{"input", {N * C}, {0.0, 1.0, 0.5, 4.0, 3.0, 5.0}};
    Variable<unsigned short> topk{"topk", {N * K}};
    Codelet codelet{
        "TopKVertex", graph, {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.initialValue("score_threshold", float(1.0));
    codelet.loadAndRun(device);
    REQUIRE_THAT(topk.data(), Catch::Equals<unsigned short>({65535, 2}));
  }
}

TEST_CASE("topk2", "[tournament]") {
  Graph graph{target};
  graph.addCodelets("codelet.gp");
  SECTION("test k=2") {
    const uint32_t N = 2, C = 3, K = 2;
    Variable<float> input{"input", {N * C}, {0.0, 1.0, 2.0, 5.0, 3.0, 4.0}};
    Variable<unsigned short> topk{"topk", {N * K}};
    Codelet codelet{
        "TopKVertex", graph, {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.initialValue("score_threshold", float(1.0));
    codelet.loadAndRun(device, true);
    REQUIRE_THAT(topk.data(), Catch::Equals<unsigned short>({2, 65535, 0, 2}));
  }
  SECTION("test k=2 no val") {
    const uint32_t N = 2, C = 3, K = 2;
    Variable<float> input{"input", {N * C}, {0.0, 1.0, 0.5, 4.0, 3.0, 5.0}};
    Variable<unsigned short> topk{"topk", {N * K}};
    Codelet codelet{
        "TopKVertex", graph, {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.initialValue("score_threshold", float(1.0));
    codelet.loadAndRun(device);
    REQUIRE_THAT(topk.data(),
                 Catch::Equals<unsigned short>({65535, 65535, 2, 0}));
  }
}

TEST_CASE("tree1", "[tournament]") {
  Graph graph{target};
  graph.addCodelets("codelet.gp");
  SECTION("test k=1") {
    const uint32_t N = 4, C = 3, K = 1;
    Variable<float> input{
        "input",
        {N * C},
        {0.0, 1.0, 2.0, 5.0, 3.0, 4.0, 0.0, 1.0, 2.0, 4.0, 3.0, 5.0}};
    Variable<unsigned short> topk{"topk", {N * K}, {2, 0, 2, 2}};
    Variable<unsigned short> tree{"tree", {N - 1}};
    Codelet codelet{"BuildWinnerTreeVertex",
                    graph,
                    {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk, tree);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.loadAndRun(device, true);
    REQUIRE_THAT(tree.data(), Catch::Equals<unsigned short>({1, 1, 3}));
  }
  SECTION("test k=1 missing topk") {
    const uint32_t N = 4, C = 3, K = 1;
    Variable<float> input{
        "input",
        {N * C},
        {0.0, 1.0, 2.0, 5.0, 3.0, 4.0, 0.0, 1.0, 2.0, 4.0, 3.0, 5.0}};
    Variable<unsigned short> topk{"topk", {N * K}, {65535, 0, 2, 2}};
    Variable<unsigned short> tree{"tree", {N - 1}};
    Codelet codelet{"BuildWinnerTreeVertex",
                    graph,
                    {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk, tree);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.loadAndRun(device);
    REQUIRE_THAT(tree.data(), Catch::Equals<unsigned short>({1, 1, 3}));
  }
  SECTION("test k=1 missing topk") {
    const uint32_t N = 4, C = 3, K = 1;
    Variable<float> input{
        "input",
        {N * C},
        {0.0, 1.0, 2.0, 5.0, 3.0, 4.0, 0.0, 1.0, 2.0, 4.0, 3.0, 5.0}};
    Variable<unsigned short> topk{"topk", {N * K}, {65535, 65535, 2, 2}};
    Variable<unsigned short> tree{"tree", {N - 1}};
    Codelet codelet{"BuildWinnerTreeVertex",
                    graph,
                    {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk, tree);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.loadAndRun(device);
    REQUIRE_THAT(tree.data(), Catch::Equals<unsigned short>({3, 65535, 3}));
  }
}

TEST_CASE("updatetree1", "[tournament]") {
  Graph graph{target};
  graph.addCodelets("codelet.gp");
  SECTION("test k=1") {
    const uint32_t N = 4, C = 3, K = 1;
    Variable<float> input{
        "input",
        {N * C},
        {0.0, 1.0, 2.0, 5.0, 3.0, 4.0, 0.0, 1.0, 2.0, 4.0, 3.0, 5.0}};
    Variable<unsigned short> topk{
        "topk", {N * K}, {2, 0, 2, 2}, poplar::UNSIGNED_SHORT, true};
    Variable<unsigned short> tree{
        "tree", {N - 1}, {1, 1, 3}, poplar::UNSIGNED_SHORT, true};
    Codelet codelet{"UpdateWinnerTreeVertex",
                    graph,
                    {poplar::FLOAT, poplar::UNSIGNED_SHORT}};
    codelet.connect(input, topk, tree);
    codelet.initialValue("C", C);
    codelet.initialValue("K", K);
    codelet.loadAndRun(device);
    REQUIRE_THAT(topk.data(), Catch::Equals<unsigned short>({2, 65535, 2, 2}));
    REQUIRE_THAT(tree.data(), Catch::Equals<unsigned short>({3, 0, 3}));
  }
}

void runNmsLarge(uint32_t minPerTile, size_t bs, uint32_t k,
                 const std::vector<float> &boxesData,
                 const std::vector<float> &scoresData,
                 const std::vector<int32_t> &indicesCorrect,
                 const std::vector<int32_t> &classesCorrect) {
  Graph graph{target};
  poplar::program::Sequence prog;
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  graph.addCodelets("codelet.cpp");
  poplar::Tensor scoresIPU =
      graph.addVariable(poplar::FLOAT, {bs, 6, 2}, "scores");
  graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
  poplar::Tensor boxesIPU =
      graph.addVariable(poplar::FLOAT, {bs, 6, 4}, "boxes");
  graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

  poplar::Tensor scoresAns, boxesAns, classesAns, indicesAns, lengths;
  std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengths) =
      nmsMultiLarge(graph, prog, scoresIPU, boxesIPU, 0.5, 6, 0.0, false, k,
                    {"nmsLarge"}, minPerTile);
  ReadHandle<int32_t> indices{indicesAns, "indexes-read", graph};
  ReadHandle<int32_t> classes{classesAns, "classes-read", graph};
  Engine engine{
      graph, prog, {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
  engine.load(device);
  engine.run(0);
  const auto &indicesData = indices.read(engine);
  REQUIRE_THAT(indicesData, Catch::Equals<int32_t>(indicesCorrect));
  const auto &classesData = classes.read(engine);
  REQUIRE_THAT(classesData, Catch::Equals<int32_t>(classesCorrect));
}

TEST_CASE("nmslarge1", "[nmslarge]") {
  const int32_t inv = std::numeric_limits<int32_t>::max();
  std::vector<float> boxesData{0, 0,     1, 1,     0, 0.1f, 1, 1.1f,
                               0, -0.1f, 1, 0.9f,  0, 10,   1, 11,
                               0, 10.1f, 1, 11.1f, 0, 100,  1, 101};
  std::vector<float> scoresData{.9f,  .1f,  .25f, .75f, .4f, .6f,
                                .95f, .05f, .5f,  .5f,  .3f, .7f};

  SECTION("Reference") {
    NDArray<float> boxes{{1, 6, 4}, boxesData};
    NDArray<float> scores{{1, 6, 2}, scoresData};
    auto res = NmsMulti(scores, boxes, 6, 0.5);
    res.print(std::cerr);
    REQUIRE_THAT(res.data(),
                 Catch::Equals<uint32_t>({3, 0, 0, 0, 1, 1, 5, 1, 4, 1, 5, 0}));
  }
  SECTION("minPerTile=1") {
    runNmsLarge(1, 1, 1, boxesData, scoresData, {3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=1 k=2") {
    runNmsLarge(1, 1, 2, boxesData, scoresData, {3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0});
  }
  SECTION("minPerTile=2") {
    runNmsLarge(2, 1, 1, boxesData, scoresData, {3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=2 k=2") {
    runNmsLarge(2, 1, 2, boxesData, scoresData, {3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0});
  }
  SECTION("minPerTile=3") {
    runNmsLarge(3, 1, 1, boxesData, scoresData, {3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=3 k=2") {
    runNmsLarge(3, 1, 2, boxesData, scoresData, {3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0});
  }
  SECTION("minPerTile=6") {
    runNmsLarge(6, 1, 1, boxesData, scoresData, {3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=6 k=2") {
    runNmsLarge(6, 1, 2, boxesData, scoresData, {3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0});
  }
}

TEST_CASE("nmslarge2", "[nmslarge]") {
  const int32_t inv = std::numeric_limits<int32_t>::max();
  std::vector<float> boxesData{
      0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
      0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101,

      0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
      0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101};
  std::vector<float> scoresData{
      .9f, .1f, .25f, .75f, .4f, .6f, .95f, .05f, .5f, .5f, .3f, .7f,

      .9f, .1f, .25f, .75f, .4f, .6f, .95f, .05f, .5f, .5f, .3f, .7f};

  SECTION("Reference") {
    NDArray<float> boxes{{2, 6, 4}, boxesData};
    NDArray<float> scores{{2, 6, 2}, scoresData};
    auto res = NmsMulti(scores, boxes, 6, 0.5);
    res.print(std::cerr);
    REQUIRE_THAT(res.data(),
                 Catch::Equals<uint32_t>({3, 0, 0, 0, 1, 1, 5, 1, 4, 1, 5, 0,
                                          3, 0, 0, 0, 1, 1, 5, 1, 4, 1, 5, 0}));
  }
  SECTION("minPerTile=1") {
    runNmsLarge(1, 2, 1, boxesData, scoresData,
                {3, 0, 1, 5, -1, -1, 3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv, 0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=1 k=2") {
    runNmsLarge(1, 2, 2, boxesData, scoresData,
                {3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0});
  }
  SECTION("minPerTile=2") {
    runNmsLarge(2, 2, 1, boxesData, scoresData,
                {3, 0, 1, 5, -1, -1, 3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv, 0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=2 k=2") {
    runNmsLarge(2, 2, 2, boxesData, scoresData,
                {3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0});
  }
  SECTION("minPerTile=3") {
    runNmsLarge(3, 2, 1, boxesData, scoresData,
                {3, 0, 1, 5, -1, -1, 3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv, 0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=3 k=2") {
    runNmsLarge(3, 2, 2, boxesData, scoresData,
                {3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0});
  }
  SECTION("minPerTile=6") {
    runNmsLarge(6, 2, 1, boxesData, scoresData,
                {3, 0, 1, 5, -1, -1, 3, 0, 1, 5, -1, -1},
                {0, 0, 1, 1, inv, inv, 0, 0, 1, 1, inv, inv});
  }
  SECTION("minPerTile=6 k=2") {
    runNmsLarge(6, 2, 2, boxesData, scoresData,
                {3, 0, 1, 5, 4, 5, 3, 0, 1, 5, 4, 5},
                {0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0});
  }
}
