#include "nms.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Gather.hpp>

namespace {
uint32_t
countRegions(const std::vector<std::vector<poplar::Interval>> &regions) {
  uint32_t res = 0;
  for (const auto &r : regions) {
    for (const auto &i : r) {
      res += i.size();
    }
  }
  return res;
}

std::vector<poplar::Interval>
scaleRegion(const std::vector<poplar::Interval> &region, uint32_t factor) {
  std::vector<poplar::Interval> res;
  for (const auto &i : region) {
    poplar::Interval newInterval{i.begin() * factor, i.end() * factor};
    res.push_back(newInterval);
  }
  return res;
}

void mapTileVertex(
    Graph &graph, const std::unordered_map<std::string, poplar::Tensor> &flat,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    const std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>>
        &aligned,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<poplar::Interval> &regions, uint16_t index,
    uint32_t tileNumber, uint32_t splitSize) {
  auto vertexRegions =
      poputil::splitRegionsBetweenWorkers(graph.getTarget(), regions, 1, 1);
  size_t j = 0;
  for (auto &r : vertexRegions) {
    VertexRef vtx = graph.addVertex(computeSet, vertexName);
    for (auto &p : flat) {
      graph.connect(vtx[p.first], poplar::concat(p.second.slices(r)));
    }
    for (auto &p : full) {
      graph.connect(vtx[p.first], p.second);
    }
    for (auto &p : aligned) {
      uint32_t factor = p.second.second;
      const auto mapping = factor > 1 ? scaleRegion(r, factor) : r;
      graph.connect(vtx[p.first],
                    poplar::concat(p.second.first.slices(mapping)));
    }
    graph.setPerfEstimate(vtx, r.size()); // wrong ...
    graph.setTileMapping(vtx, tileNumber);
    ++j;
  }
}

void mapVertex(Graph &graph,
               const std::unordered_map<std::string, poplar::Tensor> &flat,
               const std::unordered_map<std::string, poplar::Tensor> &full,
               const std::unordered_map<std::string, poplar::Tensor> &aligned,
               poplar::Type elementType, poplar::ComputeSet &computeSet,
               const std::string &vertexName,
               const std::vector<std::vector<poplar::Interval>> &mapping) {
  std::unordered_map<std::string, poplar::Tensor> flatten;
  for (auto &p : flat) {
    flatten.insert({p.first, p.second.flatten()});
  }
  uint32_t mappingSize = countRegions(mapping);
  std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>> aligned_;
  for (auto &p : aligned) {
    size_t tensorSize = p.second.numElements();
    assert(tensorSize >= mappingSize);
    uint32_t factor = tensorSize / mappingSize;
    assert(mappingSize * factor == tensorSize);
    aligned_.insert({p.first, {p.second, factor}});
  }

  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(elementType);
  const auto splitSize =
      std::max<uint32_t>(vectorWidth, target.getAtomicStoreGranularity());
  const auto numTiles = target.getTilesPerIPU();
  size_t index = 0;
  for (size_t i = 0; i < numTiles; ++i) {
    auto &regions = mapping[i];
    if (regions.size() > 0)
      mapTileVertex(graph, flatten, full, aligned_, computeSet, vertexName,
                    regions, index++, i, splitSize);
  }
}

using Mapping = std::vector<std::vector<poplar::Interval>>;

std::vector<Mapping> split_mapping(const Mapping &m, uint32_t partitions,
                                   uint32_t block_size) {
  if (partitions == 1) {
    return {m};
  }
  std::vector<Mapping> res(partitions);
  for (size_t i = 0; i < m.size(); ++i) {
    const std::vector<poplar::Interval> &m_i = m[i];
    const auto regions = poputil::splitRegions(m_i, block_size, partitions);
    for (size_t j = 0; j < regions.size(); ++j) {
      res[j].push_back(regions[j]);
    }
  }
  return res;
}

template <typename T>
void initializeTensor(poplar::Graph &graph, poplar::program::Sequence &program,
                      poplar::Tensor &t, T value) {
  poplar::Tensor v =
      graph.addConstant(t.elementType(), {1}, poplar::ArrayRef<T>({value}));
  graph.setTileMapping(v, 1);
  program.add(poplar::program::Copy(
      v.broadcast(t.numElements(), 0).reshape(t.shape()), t));
}

} // namespace

using namespace popops::expr;

poplar::Tensor compute_area(poplar::Graph &graph, program::Sequence &prog,
                            const poplar::Tensor &boxes,
                            const poplar::DebugContext &dc = {}) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(boxes));
  return popops::map(graph, (_3 - _1) * (_4 - _2),
                     {boxes.slice({0, 0, 0}, {boxes.dim(0), boxes.dim(1), 1}),
                      boxes.slice({0, 0, 1}, {boxes.dim(0), boxes.dim(1), 2}),
                      boxes.slice({0, 0, 2}, {boxes.dim(0), boxes.dim(1), 3}),
                      boxes.slice({0, 0, 3}, {boxes.dim(0), boxes.dim(1), 4})},
                     prog, di);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
initGraph(poplar::Graph &graph, program::Sequence &prog,
          const poplar::Tensor &scores, const poplar::DebugContext &dc) {
  poplar::Tensor scores_copy =
      graph.addVariable(scores.elementType(), scores.shape());
  poputil::mapTensorLinearly(graph, scores_copy, 0, 1);
  prog.add(program::Copy(scores, scores_copy));
  poplar::Tensor keep = graph.addVariable(poplar::BOOL, scores.shape());
  poputil::mapTensorLinearly(graph, keep, 0, 1);
  initializeTensor<int8_t>(graph, prog, keep, 0);
  std::vector<uint32_t> batches;
  std::vector<uint32_t> indices;
  batches.reserve(scores.numElements());
  indices.reserve(scores.numElements());
  size_t num_batches = scores.dim(0);
  size_t num_vals = scores.dim(1);
  for (size_t b = 0; b < num_batches; ++b) {
    for (size_t i = 0; i < num_vals; ++i) {
      batches.push_back(b);
      indices.push_back(i);
    }
  }
  poplar::Tensor indicesT =
      graph.addConstant(poplar::UNSIGNED_INT, scores.shape(),
                        ArrayRef<uint32_t>{indices}, {dc, "indices"});
  poputil::mapTensorLinearly(graph, indicesT, 0, 1);
  poplar::Tensor batchesT =
      graph.addConstant(poplar::UNSIGNED_INT, scores.shape(),
                        ArrayRef<uint32_t>{batches}, {dc, "batches"});
  poputil::mapTensorLinearly(graph, batchesT, 0, 1);
  return {scores_copy, keep, batchesT, indicesT};
}

poplar::Tensor gather(poplar::Graph &graph, program::Sequence &prog,
                      const poplar::Tensor &t, const poplar::Tensor &indices,
                      const poplar::DebugContext &dc) {
  std::vector<poplar::Tensor> slices;
  size_t batchSize = t.dim(0);
  for (size_t b = 0; b < batchSize; ++b) {
    poplar::Tensor subTensor = t[b];
    poplar::Tensor slice_b =
        popops::dynamicSlice(graph, subTensor, indices[b].expand({0}), {0}, {1},
                             prog, {dc, "gather"});
    slices.push_back(slice_b);
  }
  return poplar::concat(slices);
}

poplar::Tensor nms(poplar::Graph &graph, program::Sequence &prog,
                   const poplar::Tensor &scores, const poplar::Tensor &boxes,
                   const poplar::Tensor &classes, float threshold,
                   int num_detections, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, classes, threshold, num_detections));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 2);
  assert(classes.rank() == 2);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections < scores.dim(1));
  poputil::mapTensorLinearly(graph, scores, 0, 1);
  poputil::mapTensorLinearly(graph, classes, 0, 1);
  poputil::mapTensorLinearly(graph, boxes, 0, 4);
  poplar::Tensor scores_copy, keep, batches, indices;
  std::tie(scores_copy, keep, batches, indices) =
      initGraph(graph, prog, scores, di);
  poplar::Tensor areas = compute_area(graph, prog, boxes, di);
  // prog.add(program::PrintTensor("areas", areas));
  poplar::Tensor answer;
  answer = graph.addVariable(poplar::UNSIGNED_INT,
                             {scores.dim(0), size_t(num_detections)},
                             {di, "topIndices"});
  poputil::mapTensorLinearly(graph, answer, 0, 1);
  poplar::Tensor thresholdT =
      graph.addConstant(poplar::FLOAT, {}, threshold, {di, "threshold"});
  graph.setTileMapping(thresholdT, 1);
  poplar::Tensor i = graph.addVariable(poplar::UNSIGNED_INT, {1});
  graph.setTileMapping(i, 1);
  graph.setInitialValue(i, 0);
  poplar::Tensor one =
      graph.addConstant(poplar::UNSIGNED_INT, {}, 1, {di, "one"});
  graph.setTileMapping(one, 1);
  // prog.add(program::PrintTensor("scores", scores_copy));
  // prog.add(program::PrintTensor("boxes", boxes));
  program::Sequence loop;
  poplar::Tensor best_idx =
      popnn::argMax(graph, scores_copy, loop, {di, "argMax"});
  // loop.add(program::PrintTensor("argMax", best_idx));
  poplar::Tensor best_box =
      gather(graph, loop, boxes, best_idx, {di, "boxSlice"});
  // loop.add(program::PrintTensor("bestBox", best_box));
  poplar::Tensor best_area =
      gather(graph, loop, areas, best_idx, {di, "areaSlice"});
  // loop.add(program::PrintTensor("bestArea", best_area));
  poplar::Tensor best_class =
      gather(graph, loop, classes, best_idx, {di, "classSlice"});
  // loop.add(program::PrintTensor("bestClass", best_class));
  popops::dynamicUpdate(graph, answer, best_idx.expand({1}), i, {1},
                        {scores.dim(0)}, loop, {di, "updateAnswer"});
  // loop.add(program::PrintTensor("classes", classes));
  poplar::ComputeSet cs = graph.addComputeSet({di, "updateBest"});
  mapVertex(graph,
            {{"batches", batches},
             {"indices", indices},
             {"scores", scores_copy},
             {"keep", keep}},
            {{"best", best_idx}}, {}, poplar::FLOAT, cs,
            poputil::templateVertex("UpdateBestVertex", scores.elementType()),
            graph.getTileMapping(scores_copy));
  loop.add(program::Execute(cs));
  // loop.add(program::PrintTensor("scores2", scores_copy));
  // loop.add(program::PrintTensor("keep", keep));
  poplar::ComputeSet csNms = graph.addComputeSet({di, "Nms"});
  mapVertex(graph,
            {{"batches", batches},
             {"areas", areas},
             {"scores", scores_copy},
             {"classes", classes},
             {"keep", keep}},
            {{"bestBox", best_box.flatten()},
             {"bestClass", best_class.flatten()},
             {"bestArea", best_area.flatten()},
             {"threshold", thresholdT}},
            {{"boxes", boxes.flatten()}}, poplar::FLOAT, csNms,
            poputil::templateVertex("NmsVertex", scores.elementType(),
                                    classes.elementType()),
            graph.getTileMapping(scores_copy));
  loop.add(program::Execute(csNms));
  // loop.add(program::PrintTensor("scores3", scores_copy));
  popops::addInPlace(graph, i, one, loop, {di, "incrementI"});
  // loop.add(program::PrintTensor("i", i));
  prog.add(program::Repeat(num_detections, loop));
  return answer;
}
