// Copyright (c) 2021, Graphcore Ltd, All rights reserved.

#pragma once
#include "ipu_utils.hpp"

const uint16_t halfNegInf = 64511;
const float halfNegInff = -65504.0f;

template <typename T>
inline Tensor
addConstantFunction(Graph &graph, Type type, const std::vector<size_t> &shape,
                    const ArrayRef<T> &data, const std::string &name) {
  return graph.addConstant(type, shape, data, name);
}

template <>
inline Tensor addConstantFunction<uint16_t>(Graph &graph, Type type,
                                            const std::vector<size_t> &shape,
                                            const ArrayRef<uint16_t> &data,
                                            const std::string &name) {
  if (type == poplar::HALF)
    return graph.addConstantHalf(type, shape, data.data(), name);
  return graph.addConstant(type, shape, data, name);
}

template <typename T> class Variable {
private:
  std::string name_;
  std::vector<size_t> shape_;
  bool initialized_;
  std::vector<T> data_;
  Type type_;
  bool inOut_{false};
  size_t mapping_;
  // TODO: vector<Tensor> ?
  Tensor addConstant(Graph &graph, const std::vector<T> &vec) const {
    ArrayRef<T> data{vec};
    Tensor v = addConstantFunction(graph, type_, shape_, data, name_);
    graph.setTileMapping(v, mapping_);
    return v;
  }

public:
  Variable(const std::string &name, const std::vector<size_t> &shape,
           const std::vector<T> &data,
           Type type = equivalent_device_type<T>().value, bool inOut = false,
           const size_t mapping = 0)
      : name_{name}, shape_{shape}, initialized_{true}, data_{data},
        type_{type}, inOut_{inOut}, mapping_{mapping} {}
  Variable(const std::string &name, const std::vector<size_t> &shape,
           Type type = equivalent_device_type<T>().value,
           const size_t mapping = 0)
      : name_{name}, shape_{shape},
        initialized_{false}, type_{type}, mapping_{mapping} {
    size_t total = 1;
    for (auto &d : shape_) {
      total *= d;
    }
    data_.resize(total);
  }
  const Type type() const { return type_; }
  const std::vector<T> &data() const { return data_; }
  const std::string dataString() const { return to_string(data_); }
  bool isOutput() const { return !initialized_ || inOut_; }
  std::tuple<std::string, char *, char *> writeHandle() const {
    assert(isOutput());
    return {name() + "-read", (char *)data_.data(),
            (char *)data_.data() + sizeof(T) * data_.size()};
  }
  std::string name() const { return name_; }
  Tensor tensor(Graph &graph) const {
    if (!initialized_) { // variable
      Tensor v = graph.addVariable(type_, shape_, name_);
      graph.setTileMapping(v, mapping_);
      graph.createHostRead(name() + "-read", v);
      return v;
    }
    if (!inOut_) { // constant
      return addConstant(graph, data_);
    }
    // InOut variable
    Tensor v = graph.addVariable(type_, shape_, name_);
    graph.setTileMapping(v, mapping_);
    graph.createHostRead(name() + "-read", v);
    ArrayRef<T> dataRef{data_};
    graph.setInitialValue(v, dataRef);
    return v;
  }
};

class Codelet {
private:
  std::string name_;
  Graph &graph_;
  ComputeSet cs_;
  VertexRef vertex_;
  program::Sequence prog_;
  std::vector<std::pair<std::string, Tensor>> tensors_;
  std::vector<std::tuple<std::string, char *, char *>> readHandles_;

public:
  Codelet(const std::string &name, Graph &graph,
          const std::vector<Type> &types = {}, size_t mapping = 0)
      : name_{name}, graph_{graph} {
    cs_ = graph_.addComputeSet("Test" + name_);
    FloatingPointBehaviour opts{true, true, true, true, true};
    setFloatingPointBehaviour(graph_, prog_, opts);
    std::string vertexFull = name;
    switch (types.size()) {
    case 0:
      break;
    case 1:
      vertexFull = poputil::templateVertex(name, types[0]);
      break;
    case 2:
      vertexFull = poputil::templateVertex(name, types[0], types[1]);
      break;
    default:
      assert(false);
    }
    vertex_ = graph_.addVertex(cs_, vertexFull);
    graph_.setPerfEstimate(vertex_, 20);
    graph_.setTileMapping(vertex_, mapping);
  }
  template <typename T> void connect(const Variable<T> &v) {
    Tensor t = v.tensor(graph_);
    tensors_.push_back({v.name(), t});
    graph_.connect(vertex_[v.name()], t);
    if (v.isOutput())
      readHandles_.push_back(v.writeHandle());
  }
  template <typename T, typename... Args>
  void connect(const Variable<T> &v, Args &&... args) {
    connect(v);
    connect(args...);
  }

  void loadAndRun(Device &device, bool printTensors = false) {
    prog_.add(program::Execute(cs_));
    if (printTensors)
      for (auto &p : tensors_) {
        prog_.add(program::PrintTensor(p.first, p.second));
      }
    Engine engine{graph_,
                  prog_,
                  {{"debug.verify", "true"}, {"debug.runtimeVerify", "true"}}};
    engine.load(device);
    engine.run(0);
    for (auto &p : readHandles_) {
      engine.readTensor(std::get<0>(p), std::get<1>(p), std::get<2>(p));
    }
  }
};
