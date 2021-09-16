// Copyright (c) 2021, Graphcore Ltd, All rights reserved.
#include "ipu_test.hpp"
#include "reference.hpp"
#include "utils.hpp"
// #include <popops/codelets.hpp>
#include "ipu_utils.hpp"
#include "legacynms.hpp"
#include "nms.hpp"
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <random>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

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
  NDArray<uint8_t> keep{{4, 16}, 0};
  auto index = argmax(scores, keep);
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
  SECTION("IPU legacy fp32") {
    Graph graph{target};
    poplar::program::Sequence prog;
    poplar::Tensor scoresIPU =
        graph.addVariable(poplar::FLOAT, {1, 6}, "scores");
    graph.setInitialValue(scoresIPU, poplar::ArrayRef<float>{scoresData});
    poplar::Tensor boxesIPU =
        graph.addVariable(poplar::FLOAT, {1, 6, 4}, "boxes");
    graph.setInitialValue(boxesIPU, poplar::ArrayRef<float>{boxesData});

    auto ipu_res = build_nms(graph, prog, scoresIPU, boxesIPU, 3, 0.5);
    ReadHandle<uint32_t> indices{ipu_res, "indexes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    // std::cerr << "Answer " << to_string(indicesData) << std::endl;
    // REQUIRE_THAT(indicesData, Catch::Approx<uint32_t>({3, 0, 5})); //
    // actually returns {3, 0, 4}
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

    auto ipu_res = nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 3);
    ReadHandle<int32_t> indices{ipu_res, "indexes-read", graph};
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
  std::vector<float> boxesData{0, 0,     1, 1,     0, 0.1f, 1, 1.1f,  0, -0.1f,
                               1, 0.9f,  0, 10,    1, 11,   0, 10.1f, 1, 11.1f,
                               0, 100,   1, 101,   0, 0,    1, 1,     0, 0.1f,
                               1, 1.1f,  0, -0.1f, 1, 0.9f, 0, 10,    1, 11,
                               0, 10.1f, 1, 11.1f, 0, 100,  1, 101};
  std::vector<float> scoresData{.9f, .75f, .6f, .95f, .5f, .3f,
                                .9f, .75f, .6f, .95f, .5f, .3f};
  NDArray<uint32_t> classes{{2, 6}, 1};

  SECTION("Reference") {
    NDArray<float> boxes{{2, 6, 4}, boxesData};
    NDArray<float> scores{{2, 6}, scoresData};
    auto res = Nms(scores, boxes, classes, 3, 0.5);
    res.print(std::cerr);
    REQUIRE_THAT(res.data(), Catch::Equals<uint32_t>({3, 0, 5, 3, 0, 5}));
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

    auto ipu_res = nms(graph, prog, scoresIPU, boxesIPU, classesIPU, 0.5, 3);
    ReadHandle<int32_t> indices{ipu_res, "indexes-read", graph};
    Engine engine{graph,
                  prog,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    const auto &indicesData = indices.read(engine);
    REQUIRE_THAT(indicesData, Catch::Equals<int32_t>({3, 0, 5, 3, 0, 5}));
  }
}
