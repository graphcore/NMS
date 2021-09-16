// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

//
// This example demonstrates how to create a custom operator for PopART, in this
// case an op that will take a tensor and cube all the elements
//
//
// ISSUE : the version can currently only be 9. Need to support onnx version
// information
#include <iostream>
#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <popops/Cast.hpp>

#include "nms.hpp"

// The first thing to do is to provide an identifier that PopART can use later
// to address the operator.
namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier Nms = {"ai.graphcore", "Nms", 1};
} // namespace CustomOperators
} // namespace Onnx

// For training with a custom Op, four classes need to be implemented,
// one for each of:
// {forward, gradient} x {Op, Opx}.
//
// If only inference is required, then two classes need to be implemented:
// {forward} x {Op, Opx}.
//
// The Op is a poplar/hardware agnostic description of the computation.
// the Opx is the poplar implementation of the Op.
//
// We do training in this example, so the four classes implemented are:
//
class NmsOp;
class NmsOpx;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The forward Op
class NmsOp : public popart::Op {
public:
  NmsOp(const popart::OperatorIdentifier &_opid, const float threshold,
        const uint32_t numDetections, const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), threshold_{threshold},
        numDetections_{numDetections} {}

  // same comment as for NmsGradOp, for running shape/type inference
  // "statically"
  virtual void setup() {
    auto scoresInfo = inInfo(0);
    auto boxesInfo = inInfo(1);
    auto classesInfo = inInfo(2);
    assert(scoresInfo.rank() == 2);
    assert(boxesInfo.rank() == 3);
    assert(classesInfo.rank() == 2);
    const uint32_t batchSize = scoresInfo.dim(0);
    const uint32_t N = scoresInfo.dim(1);

    outInfo(0).set(popart::DataType::INT32, {batchSize, numDetections_});
    std::cerr << "Debug NmsOp::setup " << threshold_ << " numDetections "
              << numDetections_ << "]\n";
  }

  std::unique_ptr<Op> clone() const final { return make_unique<NmsOp>(*this); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("threshold", getThreshold());
    os.appendAttribute("numDetections", getNumDetections());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("threshold", getThreshold());
    os.appendAttribute("numDetections", getNumDetections());
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  float getThreshold() const { return threshold_; }
  uint32_t getNumDetections() const { return numDetections_; }

private:
  float threshold_;
  uint32_t numDetections_;
};

// describe the inputs and outputs that are supported by the operation
static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes T2 = {popart::DataType::INT32,
                                             popart::DataType::UINT32};
static popart::OpDefinition::DataTypes T3 = {popart::DataType::INT32};

static popart::OpDefinition
    nmsOpDef({popart::OpDefinition::Inputs(
                  {{"scores", T}, {"boxes", T}, {"classes", T2}}),
              popart::OpDefinition::Outputs({{"indices", T3}}),
              popart::OpDefinition::Attributes({{"threshold", {"*"}},
                                                {"numDetections", {"*"}}})});

static popart::OpCreator<NmsOp> nmsOpCreator(
    popart::OpDefinitions({{Onnx::CustomOperators::Nms, nmsOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      float threshold = info.attributes.getAttribute<popart::Attributes::Float>(
          "threshold", 0.5f);
      uint32_t numDetections =
          info.attributes.getAttribute<popart::Attributes::Int>("numDetections",
                                                                100);
      return std::make_unique<NmsOp>(info.opid, threshold, numDetections,
                                     info.settings);
    },
    true);

// forward Opx (poplar implementation of the forward Op)
class NmsOpx : public popart::popx::Opx {
public:
  NmsOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // not strictly necessary, we check that op is castable to a NmsOp *.
    verifyOp<NmsOp>(op, Onnx::CustomOperators::Nms);
    graph().addCodelets("codelet.cpp"); // add codelets to the graph
  }

  void grow(poplar::program::Sequence &prog) const final {
    // Nms the input. We create a poplar::Tensor of name outId(0)
    std::cerr << "Debug NmsOpx::grow\n";
    auto op = getOp<NmsOp>();
    float threshold = op.getThreshold();
    uint32_t numDetections = op.getNumDetections();
    const auto &scores = getInTensor(0);
    const auto &boxes = getInTensor(1);
    const auto &classes = getInTensor(2);
    auto answer =
        nms(graph(), prog, scores, boxes, classes, threshold, numDetections);
    // for pytorch only ...
    answer = popops::cast(graph(), answer, poplar::INT, prog);
    setOutTensor(0, answer);
  }
};

static popart::popx::OpxCreator<NmsOpx>
    nmsOpxCreator(Onnx::CustomOperators::Nms);
