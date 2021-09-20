// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "nms.hpp"
#include <poplar/VariableMappingMethod.hpp>
#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

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

void mapRegionVertex(
    Graph &graph,
    const std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>>
        &aligned,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<poplar::Interval> &regions, size_t index,
    uint32_t tileNumber,
    const std::function<void(VertexRef &, uint32_t, uint32_t)> &callback) {
  VertexRef vtx = graph.addVertex(computeSet, vertexName);
  for (const auto &p : full) {
    graph.connect(vtx[p.first], p.second);
  }
  for (const auto &p : aligned) {
    uint32_t factor = p.second.second;
    const auto mapping = factor > 1 ? scaleRegion(regions, factor) : regions;
    graph.connect(vtx[p.first], poplar::concat(p.second.first.slices(mapping)));
  }
  graph.setPerfEstimate(vtx, regions.size()); // wrong ...
  graph.setTileMapping(vtx, tileNumber);
  callback(vtx, tileNumber, index);
}

void mapTileVertex(
    Graph &graph,
    const std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>>
        &aligned,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<poplar::Interval> &regions, size_t index,
    uint32_t tileNumber, uint32_t numVertices,
    const std::function<void(VertexRef &, uint32_t, uint32_t)> &callback) {
  if (numVertices == 1) {
    return mapRegionVertex(graph, aligned, full, computeSet, vertexName,
                           regions, index, tileNumber, callback);
  }
  std::vector<std::vector<poplar::Interval>> vertexRegions;
  if (numVertices == 0) {
    vertexRegions =
        poputil::splitRegionsBetweenWorkers(graph.getTarget(), regions, 1, 1);
  } else {
    vertexRegions = poputil::splitRegions(regions, 1, numVertices, 1);
  }
  for (auto &r : vertexRegions) {
    mapRegionVertex(graph, aligned, full, computeSet, vertexName, r, index,
                    tileNumber, callback);
  }
}

void mapVertex(
    Graph &graph,
    const std::unordered_map<std::string, poplar::Tensor> &aligned,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<std::vector<poplar::Interval>> &mapping,
    uint32_t numVertices = 0,
    const std::function<void(VertexRef &, uint32_t, uint32_t)> &callback =
        [](VertexRef &, uint32_t, uint32_t) {}) {
  std::unordered_map<std::string, poplar::Tensor> flatten;
  uint32_t mappingSize = countRegions(mapping);
  std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>> aligned_;
  for (const auto &p : aligned) {
    size_t tensorSize = p.second.numElements();
    assert(tensorSize >= mappingSize);
    uint32_t factor = tensorSize / mappingSize;
    assert(mappingSize * factor == tensorSize);
    aligned_.insert({p.first, {p.second, factor}});
  }

  size_t index = 0;
  for (size_t i = 0; i < mapping.size(); ++i) {
    const auto &regions = mapping[i];
    if (!regions.empty()) {
      mapTileVertex(graph, aligned_, full, computeSet, vertexName, regions,
                    index++, i, numVertices, callback);
    }
  }
}

using Mapping = std::vector<std::vector<poplar::Interval>>;
std::vector<size_t> unflattenRegion(const Mapping &mapping, uint32_t tile,
                                    const std::vector<size_t> &shape) {
  const std::vector<Interval> &r = mapping[tile];
  assert(r.size() == 1);
  return poputil::unflattenIndex(shape, r.begin()->begin());
}

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

std::pair<poplar::Tensor, size_t>
createLocalTensor(poplar::Graph &graph, const Mapping &mapping,
                  poplar::Type type, size_t numElem,
                  const poplar::DebugContext &dc) {
  size_t nbTiles = 0;
  for (const auto &r : mapping) {
    if (!r.empty()) {
      ++nbTiles;
    }
  }
  poplar::Tensor res = graph.addVariable(type, {nbTiles, numElem}, dc);
  size_t i = 0;
  for (size_t tile = 0; tile < mapping.size(); ++tile) {
    if (!mapping[tile].empty()) {
      graph.setTileMapping(res[i++], tile);
    }
  }
  return {res, nbTiles};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
initGraphCommon(poplar::Graph &graph, program::Sequence &prog,
                const poplar::Tensor &scores, uint32_t numDetections,
                const poplar::Type &classType, const poplar::DebugContext &dc) {
  const size_t num_batches = scores.dim(0);
  poplar::Tensor lengths = graph.addVariable(
      poplar::INT, {num_batches, (numDetections)}, {dc, "lengths"});
  poputil::mapTensorLinearly(graph, lengths, 1, 1);
  initializeTensor<int32_t>(graph, prog, lengths, int32_t(numDetections));
  poplar::Tensor topBoxes = graph.addVariable(
      scores.elementType(), {num_batches, size_t(numDetections), 4},
      {dc, "topBoxes"});
  poputil::mapTensorLinearly(graph, topBoxes, 1, 4);
  initializeTensor<float>(graph, prog, topBoxes, 0.0);
  poplar::Tensor topScores = graph.addVariable(
      scores.elementType(), {num_batches, size_t(numDetections)},
      {dc, "topScores"});
  poputil::mapTensorLinearly(graph, topScores, 1, 1);
  initializeTensor<float>(graph, prog, topScores, 0.0);
  poplar::Tensor topIndices = graph.addVariable(
      poplar::INT, {num_batches, size_t(numDetections)}, {dc, "topIndices"});
  poputil::mapTensorLinearly(graph, topIndices, 1, 1);
  initializeTensor<int32_t>(graph, prog, topIndices, -1);
  poplar::Tensor topClasses = graph.addVariable(
      classType, {num_batches, size_t(numDetections)}, {dc, "topClasses"});
  poputil::mapTensorLinearly(graph, topClasses, 1, 1);
  initializeTensor<int32_t>(graph, prog, topClasses,
                            std::numeric_limits<int32_t>::max());

  return {topIndices, topScores, topBoxes, topClasses, lengths};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
initGraphMultiLarge(poplar::Graph &graph, program::Sequence &prog,
                    const poplar::Tensor &scores, uint32_t numDetections,
                    poplar::Type indexType, uint32_t K, uint32_t minPerTile,
                    const poplar::DebugContext &dc) {
  const size_t bs = scores.dim(0);
  const size_t N = scores.dim(1);
  poplar::Tensor topks = graph.addVariable(indexType, {bs, N, K}, {dc, "topk"});
  poputil::mapTensorLinearly(graph, topks, minPerTile * K, K);
  poplar::Tensor max, argmax;
  size_t numTiles;
  std::tie(max, numTiles) = createLocalTensor(
      graph, graph.getTileMapping(topks), poplar::FLOAT, 1, {dc, "max"});
  const auto &mapping = graph.getTileMapping(max);
  argmax = graph.addVariable(poplar::UNSIGNED_INT, max.shape(), {dc, "argmax"});
  graph.setTileMapping(argmax, mapping);

  poplar::Tensor topIndices, topScores, topBoxes, topClasses, lengths;
  std::tie(topIndices, topScores, topBoxes, topClasses, lengths) =
      initGraphCommon(graph, prog, scores, numDetections, poplar::INT, dc);

  return {topks,     max,      argmax,     topIndices,
          topScores, topBoxes, topClasses, lengths};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor, poplar::Tensor>
initGraphMulti(poplar::Graph &graph, program::Sequence &prog,
               const poplar::Tensor &scores, uint32_t numDetections,
               bool inPlace, const poplar::DebugContext &dc) {
  const size_t min_per_tile = 1;
  const size_t C = scores.dim(2);
  poplar::Tensor topIndices, topScores, topBoxes, topClasses, lengths;
  std::tie(topIndices, topScores, topBoxes, topClasses, lengths) =
      initGraphCommon(graph, prog, scores, numDetections, poplar::INT, dc);

  if (inPlace)
    return {scores, topIndices, topScores, topBoxes, topClasses, lengths};

  poplar::Tensor scores_copy =
      graph.addVariable(scores.elementType(), scores.shape());
  poputil::mapTensorLinearly(graph, scores_copy, min_per_tile, C);
  prog.add(program::Copy(scores, scores_copy));

  return {scores_copy, topIndices, topScores, topBoxes, topClasses, lengths};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor, poplar::Tensor>
initGraph(poplar::Graph &graph, program::Sequence &prog,
          const poplar::Tensor &scores, const poplar::Tensor &classes,
          uint32_t numDetections, bool inPlace,
          const poplar::DebugContext &dc) {
  const size_t min_per_tile = 1;
  poplar::Tensor topIndices, topScores, topBoxes, topClasses, lengths;
  std::tie(topIndices, topScores, topBoxes, topClasses, lengths) =
      initGraphCommon(graph, prog, scores, numDetections, classes.elementType(),
                      dc);
  if (inPlace) {
    return {scores, topIndices, topScores, topBoxes, topClasses, lengths};
  }
  poplar::Tensor scores_copy =
      graph.addVariable(scores.elementType(), scores.shape());
  poputil::mapTensorLinearly(graph, scores_copy, min_per_tile, 1);
  prog.add(program::Copy(scores, scores_copy));
  return {scores_copy, topIndices, topScores, topBoxes, topClasses, lengths};
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

std::pair<poplar::Tensor, size_t>
createGatherTensor(poplar::Graph &graph, const Mapping &mapping,
                   const poplar::Tensor &output,
                   const poplar::DebugContext &dc) {
  return createLocalTensor(graph, mapping, output.elementType(),
                           output.numElements(), dc);
}

void mapReduceGather(
    poplar::Graph &graph, program::Sequence &prog, const Mapping &mapping,
    const std::vector<size_t> &shape, poplar::Tensor &output,
    poplar::Tensor &outputClass, const std::string &vertexName,
    const poplar::Tensor &index,
    const std::unordered_map<std::string, poplar::Tensor> &aligned,
    bool withClasses, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(output));
  poplar::Tensor gatherTensor;
  size_t firstDim;
  size_t batchSize = shape[0];
  std::tie(gatherTensor, firstDim) =
      createGatherTensor(graph, mapping, output, {di, "gather"});
  gatherTensor = gatherTensor.reshape({firstDim, batchSize, 4});
  initializeTensor(graph, prog, gatherTensor, 0.0);
  poplar::Tensor gatherTensorClass;
  size_t firstDimClass;
  std::tie(gatherTensorClass, firstDimClass) =
      createGatherTensor(graph, mapping, outputClass, {di, "gatherClass"});
  initializeTensor<int>(graph, prog, gatherTensorClass, 0);
  gatherTensorClass = gatherTensorClass.reshape({firstDimClass, batchSize});
  poplar::ComputeSet cs = graph.addComputeSet({di, "mapReduceGather"});
  mapVertex(
      graph, aligned, {}, cs, vertexName, mapping, 1,
      [&](VertexRef &vtx, uint32_t tile, uint32_t index_) {
        std::vector<size_t> indices = unflattenRegion(mapping, tile, shape);
        size_t batch = indices[0], indice = indices[1];
        graph.connect(vtx["index"], index[batch]);
        graph.connect(vtx["output"], gatherTensor[index_][batch]);
        graph.connect(vtx["outputClass"], gatherTensorClass[index_][batch]);
        graph.setInitialValue(vtx["offset"], indice);
      });
  prog.add(program::Execute(cs));
  popops::ReduceParams params(popops::Operation::ADD);
  poplar::Tensor min = popops::reduce(
      graph, gatherTensor.reshape({firstDim, output.numElements()}),
      output.elementType(), {0}, params, prog, {di, "mapReduceGatherReduce"},
      {{"accumType.interTile", "float"}, {"accumType.inVertex", "float"}});
  prog.add(program::Copy(min, output.flatten()));
  if (!withClasses) {
    return;
  }
  if (outputClass.elementType() == poplar::UNSIGNED_INT) {
    gatherTensorClass = gatherTensorClass.reinterpret(poplar::INT);
  }
  poplar::Tensor minClass = popops::reduce(
      graph,
      gatherTensorClass.reshape({firstDimClass, outputClass.numElements()}),
      poplar::INT, {0}, params, prog, {di, "mapReduceGatherReduceClass"});
  if (outputClass.elementType() == poplar::UNSIGNED_INT) {
    minClass = popops::cast(graph, minClass, poplar::UNSIGNED_INT, prog,
                            {di, "castBestClassUInt"});
  }
  prog.add(program::Copy(minClass, outputClass.flatten()));
}

void mapReduceGatherMulti(
    poplar::Graph &graph, program::Sequence &prog, const Mapping &mapping,
    const std::vector<size_t> &shape, poplar::Tensor &output,
    const uint32_t numClasses, const std::string &vertexName,
    const poplar::Tensor &index,
    const std::unordered_map<std::string, poplar::Tensor> &aligned,
    const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(output));
  poplar::Tensor gatherTensor;
  size_t firstDim;
  size_t batchSize = shape[0];
  std::tie(gatherTensor, firstDim) =
      createGatherTensor(graph, mapping, output, {di, "gather"});
  initializeTensor(graph, prog, gatherTensor, 0.0);
  gatherTensor = gatherTensor.reshape({firstDim, batchSize, 4});
  poplar::ComputeSet cs = graph.addComputeSet({di, "mapReduceGather"});
  mapVertex(graph, aligned, {}, cs, vertexName, mapping, 1,
            [&](VertexRef &vtx, uint32_t tile, uint32_t index_) {
              std::vector<size_t> indices =
                  unflattenRegion(mapping, tile, shape);
              size_t batch = indices[0], indice = indices[1];
              graph.connect(vtx["index"], index[batch]);
              graph.connect(vtx["output"], gatherTensor[index_][batch]);
              graph.setInitialValue(vtx["C"], numClasses);
              graph.setInitialValue(vtx["offset"], indice);
            });
  prog.add(program::Execute(cs));
  popops::ReduceParams params(popops::Operation::ADD);
  poplar::Tensor min = popops::reduce(
      graph, gatherTensor.reshape({firstDim, output.numElements()}),
      output.elementType(), {0}, params, prog, {di, "mapReduceGatherReduce"});
  prog.add(program::Copy(min, output.flatten()));
}

std::tuple<program::Execute, poplar::Tensor, poplar::Tensor, poplar::Tensor>
condition(poplar::Graph &graph, program::Sequence &prog,
          const poplar::Tensor &scores, uint32_t numIterations,
          float scoreThreshold, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(numIterations));
  poplar::Tensor bestScores = graph.addVariable(
      scores.elementType(), {scores.dim(0)}, {di, "bestScores"});
  graph.setTileMapping(bestScores, 1);
  poplar::Tensor zeroF =
      graph.addConstant(scores.elementType(), {1}, 0.0, {di, "zeroF"});
  graph.setTileMapping(zeroF, 1);
  prog.add(
      program::Copy(zeroF.broadcast(bestScores.numElements(), 0), bestScores));
  poplar::Tensor iterT = graph.addConstant(
      poplar::UNSIGNED_INT, {}, numIterations, {di, "numIterations"});
  poplar::Tensor predicate = graph.addVariable(poplar::BOOL, {});
  graph.setTileMapping(predicate, 1);
  graph.setTileMapping(iterT, 1);
  poplar::Tensor zero =
      graph.addConstant(poplar::UNSIGNED_INT, {}, 0, {di, "zero"});
  graph.setTileMapping(zero, 1);
  poplar::Tensor i = graph.addVariable(poplar::UNSIGNED_INT, {});
  graph.setTileMapping(i, 1);
  prog.add(program::Copy(zero, i, false, {di, "initializeI"}));
  poplar::ComputeSet cs = graph.addComputeSet({di, "condition"});
  // call vertex
  VertexRef vtx = graph.addVertex(
      cs, poputil::templateVertex("ConditionVertex", bestScores.elementType()));
  graph.connect(vtx["bestScores"], bestScores);
  graph.connect(vtx["condition"], predicate);
  graph.connect(vtx["i"], i);
  graph.connect(vtx["numIter"], iterT);
  graph.setPerfEstimate(vtx, bestScores.numElements() + 1);
  graph.setTileMapping(vtx, 1);
  graph.setInitialValue(vtx["score_threshold"], scoreThreshold);
  return {program::Execute(cs), bestScores, predicate, i};
}

// Just a shorthand to compute ceiling of the quotient (a/b)
inline unsigned quotCeiling(unsigned a, unsigned b) { return (a + b - 1) / b; }

#include "poputil/exceptions.hpp"
using namespace poplar;
using namespace poputil;
using namespace poplar::program;

// Parameters needed to create one ReduceXxxClassGather vertex, for the first
// stage reduction in maxMinArgMaxMin().
struct ClassGatherVertexInfo {
  unsigned tile;       // In which tile to place the vertex
  unsigned row;        // Which row the elements belongs to
  unsigned offsIn;     // Offset (in elements) inside 'row'
  unsigned size;       // Total size (in elements)
  unsigned offsOut;    // Offset where to put the results in the partials
  unsigned workerSize; // Processed by one worker (except possibly the last one)
  unsigned workerNum;  // How many worker (i.e. how many partials)
};

std::string to_string(const ClassGatherVertexInfo &c) {
  return "tile:" + std::to_string(c.tile) + " row:" + std::to_string(c.row) +
         " workerNum:" + std::to_string(c.workerNum) +
         " offsIn:" + std::to_string(c.offsIn) +
         " size:" + std::to_string(c.size) +
         " workerSize:" + std::to_string(c.workerSize) +
         " offsOut:" + std::to_string(c.offsOut);
}

/// Generate the work partition for the first stage reduction of
/// maxMinArgMaxMin. Tries to spread the work uniformly among all IPU tiles.
/// Each tile will have one or more vertices, so that the total number of
/// elements processed per tile is balanced.
/// Also we don't want to assign too little work per worker (and supervisor)
/// and not too much per worker (as the workers use the RPT instruction.)
///
/// \param[in] target         Target where we are running the graph.
/// \param[in] nRows          Number of rows (batches) in the input matrix
/// \param[in] nCols          Number of columns (classes) in the input matrix
///
/// \param[out] partialsPerRow one element per row, specifying how many
//                             partial outputs will be generated for this row
///                            (i.e. how many workers per row)
/// \param[out] vertexInfo     one element per vertex to create, containing all
//                             parameters for the vertex
void argMinMaxSplitFirstReduction(
    const Target &target, unsigned nRows, unsigned nCols,
    std::vector<unsigned> &partialsPerRow,
    std::vector<ClassGatherVertexInfo> &vertexInfo,
    const uint32_t workerMin = 32) {

  const auto numTileWorkers = target.getNumWorkerContexts();
  // Min and max elements to be processed by one worker.
  const unsigned workerMax = target.getRptCountMax();
  // Min elements to be processed by one supervisor vertex
  const unsigned tileMin = numTileWorkers * workerMin;
  const uint64_t totalSize = nRows * nCols; // elements in the matrix
  auto elemsPerTile =
      std::max(tileMin, quotCeiling(totalSize, target.getTilesPerIPU()));
  auto tilesToUse = quotCeiling(totalSize, elemsPerTile);
  // Starting up.
  unsigned row = 0;
  unsigned offsIn = 0;  // offset (from row start) of the vertex input data
  unsigned offsOut = 0; // offset (from row start) of the vertex output partials
  unsigned numPartials = 0; // how many partials for the current row
  // Distribute the elements among tilesToUse tiles.
  for (unsigned tile = 0; tile < tilesToUse; ++tile) {
    const uint64_t elemBegin = (tile * totalSize) / tilesToUse;
    const uint64_t elemEnd = ((tile + 1) * totalSize) / tilesToUse;
    // Total size in this tile.
    uint64_t tileSize = elemEnd - elemBegin;
    // While there are still elements to add to this tile...
    while (tileSize > 0) {
      // Are we finished with this row?
      if (offsIn == nCols) {
        partialsPerRow.push_back(numPartials);
        numPartials = 0;
        row++;
        offsIn = 0;
        offsOut = 0;
      }
      // Try to give one vertex all that is left in tileSize (or whatever
      // is left to the end of the row)
      unsigned vertexSize = std::min<uint64_t>(tileSize, nCols - offsIn);
      unsigned workerSize, numWorkers;
      // Make sure each worker does a minimum of work
      if (vertexSize / numTileWorkers >= workerMin) {
        // Enough work for all 6 workers
        workerSize = quotCeiling(vertexSize, numTileWorkers);
        // but not too much work (RPT counter is limited)
        if (workerSize > workerMax) {
          workerSize = workerMax;
          vertexSize = numTileWorkers * workerSize;
        }
        numWorkers = numTileWorkers;
      } else {
        // Cannot give enough work to all 6 worker
        workerSize = workerMin;
        numWorkers = quotCeiling(vertexSize, workerMin);
      }
      // Store away the parameters for this vertex
      auto tileToMapVertex = target.getTilesPerIPU() - 1 - tile;
      vertexInfo.push_back({tileToMapVertex, row, offsIn, vertexSize, offsOut,
                            workerSize, numWorkers});
      numPartials += numWorkers;
      offsIn += vertexSize;
      offsOut += numWorkers;
      tileSize -= vertexSize;
    }                                    // while (tileSize > 0)
  }                                      // for (tile)
  partialsPerRow.push_back(numPartials); // add last one
  // std::cerr << "NumPartials " << to_string(partialsPerRow) << std::endl;
  // std::cerr << "vertexInfo " << vertexInfo.size() << std::endl;
  // for (const auto &v : vertexInfo) {
  //   std::cerr << to_string(v) << std::endl;
  // }
}

/// Returns the max (or min) values and their indices per row in a 2-D tensor.
///
/// \param[in] graph       the graph for the tensor
/// \param[in] input       the (2-D) tensor to examine
/// \param[in] resultType  type to use for the result elements
/// \param[in] prog        the sequence to add compute sets to
/// \param[in] debugContext as the name says
/// \param[in] max         if True find max, else find min
///
/// \return a pair of 1-D tensor each with as many elements as the number of
///         rows in 'input', with each element in the first tensor being the max
///         (or min) value for that row in 'input' and the second tensor being
///         the index of where the max (or min) value for that row is in
///         'input'.
static std::pair<Tensor, Tensor> maxArgMax(Graph &graph, const Tensor &input,
                                           const Tensor &indices,
                                           Sequence &prog,
                                           const uint32_t workerMin,
                                           const DebugNameAndId &dnai) {
  const std::string lowerCase = "max";
  const std::string capitalized = "Max";
  const std::string layerPrefix = "maxArgMax()/";
  const auto &target = graph.getTarget();
  const auto tilesPerIPU = target.getTilesPerIPU();
  const size_t nRows = input.dim(0);
  const size_t nCols = input.numElements() / nRows;
  const auto inputType = input.elementType();
  // We set the partial values (max/min) to always be 32-bit floats. This works
  // both if the inputs are half or floats, and avoids half-word writes for the
  // partials values. Memory cost is considered negligible as there are few of
  // these partials (2nd stage will have 1/workerMin of initial elements etc).
  const auto partialsType =
      (inputType == HALF || inputType == FLOAT) ? FLOAT : inputType;

  // First stage of reductions (a single compute set).
  // In this stage we use supervisor vertices that will produce as output
  // multiple pairs of partial result, one for each worker processing a chunk of
  // data.
  // The outputs are the max (or min) value for that chunk and the index for the
  // max/min.

  const auto cs = graph.addComputeSet({dnai, layerPrefix + "ReduceClass[0]"});
  std::vector<unsigned> numPartials;
  std::vector<ClassGatherVertexInfo> vertexInfo;
  argMinMaxSplitFirstReduction(target, nRows, nCols, numPartials, vertexInfo,
                               workerMin);
  // How many rows will be fully reduced to a single element by this stage.
  unsigned rowsFullyReduced =
      std::count_if(numPartials.begin(), numPartials.end(),
                    [](unsigned count) { return count == 1; });
  // The partials generated by this first stage, input for second stage. Each
  // row might have a different number of partials.
  std::vector<Tensor> valuePartials(nRows);
  std::vector<Tensor> indexPartials(nRows);
  for (unsigned row = 0; row < nRows; row++) {
    valuePartials[row] = graph.addVariable(
        partialsType, {numPartials[row]},
        {dnai, layerPrefix + "ValuePartials[0][" + std::to_string(row) + "]"});
    indexPartials[row] = graph.addVariable(
        poplar::UNSIGNED_INT, {numPartials[row]},
        {dnai, layerPrefix + "IndexPartials[0][" + std::to_string(row) + "]"});
  }
  const auto vertexGather =
      templateVertex("popnn::Reduce" + capitalized + "ClassSparse", inputType,
                     poplar::UNSIGNED_INT);
  // Create all vertices for first stage.
  for (auto vi : vertexInfo) {
    size_t offset = 0;
    for (uint32_t worker = 0; worker < vi.workerNum; ++worker) {
      size_t wsize = vi.workerSize;
      if (offset + wsize > vi.size) {
        wsize = vi.size - offset;
      }
      size_t full_offset = vi.offsIn + offset;
      const auto v = graph.addVertex(cs, vertexGather);
      auto inputPartials =
          input[vi.row].slice(full_offset, full_offset + wsize);
      auto indicesPartials =
          indices[vi.row].slice(full_offset, full_offset + wsize);
      auto vertexValuePartials = valuePartials[vi.row][vi.offsOut + worker];
      auto vertexIndexPartials = indexPartials[vi.row][vi.offsOut + worker];
      graph.connect(v["activations"], inputPartials);
      graph.connect(v["labels"], indicesPartials);
      graph.connect(v["maxValue"], vertexValuePartials);
      graph.connect(v["maxIndex"], vertexIndexPartials);
      graph.setTileMapping(vertexValuePartials, vi.tile);
      graph.setTileMapping(vertexIndexPartials, vi.tile);
      graph.setTileMapping(v, vi.tile);
      offset += wsize;
    }
  }
  prog.add(Execute(cs, {dnai}));

  // The second and successive stages (each one is one compute set) will divide
  // the partials in batches of 'partialsSize' elements to be processed each by
  // a single worker vertex.
  // For these stages, both the input and the output of each stage are the
  // 1D tensors of max/min (float) values and their corresponding indices.

  unsigned tile = target.getTilesPerIPU() - 1;
  std::size_t reduceIndex = 1; // stage of the reduction
  // How many data element (max) will be processed by one worker vertex.
  const std::size_t partialsSize = workerMin;
  const auto vertexSparse =
      templateVertex("popnn::Reduce" + capitalized + "ClassSparse",
                     partialsType, poplar::UNSIGNED_INT);
  // Do it until we have reduced to a single element (per row) on all rows.
  while (rowsFullyReduced < nRows) {
    const auto stageStr = "[" + std::to_string(reduceIndex) + "]";
    const auto cs =
        graph.addComputeSet({dnai, layerPrefix + "ReduceClass" + stageStr});
    for (std::size_t row = 0; row < nRows; ++row) {
      // if rows was already reduced, nothing to do
      if (numPartials[row] > 1) {
        const std::string suffix =
            "Partials" + stageStr + "[" + std::to_string(row) + "]";
        unsigned nextNumPartials = quotCeiling(numPartials[row], partialsSize);
        // New partials for this row (output from this stage)
        auto nextValuePartials =
            graph.addVariable(partialsType, {nextNumPartials},
                              {dnai, layerPrefix + "Value" + suffix});
        auto nextIndexPartials =
            graph.addVariable(poplar::UNSIGNED_INT, {nextNumPartials},
                              {dnai, layerPrefix + "Index" + suffix});
        // All vertices for this row
        for (size_t i = 0, offs = 0; offs < numPartials[row];
             i++, offs += partialsSize) {
          const auto v = graph.addVertex(cs, vertexSparse);
          const auto size = std::min(numPartials[row] - offs, partialsSize);
          // Input values/indices for this vertex
          auto splitValuePartials = valuePartials[row].slice(offs, offs + size);
          auto splitIndexPartials = indexPartials[row].slice(offs, offs + size);
          graph.connect(v["activations"], splitValuePartials);
          graph.connect(v["labels"], splitIndexPartials);
          graph.connect(v[lowerCase + "Value"], nextValuePartials[i]);
          graph.connect(v[lowerCase + "Index"], nextIndexPartials[i]);
          graph.setTileMapping(nextValuePartials[i], tile);
          graph.setTileMapping(nextIndexPartials[i], tile);
          graph.setTileMapping(v, tile);
          tile = (tilesPerIPU + tile - 1) % tilesPerIPU;
        } // for (i,offs)
        // the outputs just generated become the inputs of next stage
        valuePartials[row] = nextValuePartials;
        indexPartials[row] = nextIndexPartials;
        numPartials[row] = nextNumPartials;
        if (nextNumPartials == 1) {
          rowsFullyReduced++;
        }
      } // row was not reduced yet
    }   // for (nRows)
    prog.add(Execute(cs, {dnai}));
    reduceIndex++;
  } // while (rowsFullyReduced < nRows)
  return std::make_pair(concat(valuePartials), concat(indexPartials));
}

std::pair<poplar::Tensor, poplar::Tensor>
localMaxAndArgMax(Graph &graph, const Tensor &input, const Tensor &indices,
                  poplar::Type resultType, Sequence &prog,
                  const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  poplar::Tensor values, indices_;
  std::tie(values, indices_) = maxArgMax(graph, input, indices, prog, 16, {di});

  if (values.elementType() != resultType) {
    values = popops::cast(graph, values, resultType, prog, {di});
  }
  di.addOutputs(
      {{"max", toProfileValue(values)}, {"argMax", toProfileValue(indices_)}});
  return std::make_pair(values, indices_);
}

void connectSliceVertex(poplar::Graph &graph, ComputeSet &cs,
                        const poplar::Tensor &input,
                        const poplar::Tensor &index,
                        const poplar::Tensor &output, uint32_t numClasses = 1) {
  const auto vertexName = poputil::templateVertex(
      "SliceVertex", input.elementType(), index.elementType());
  VertexRef vtx = graph.addVertex(cs, vertexName);
  size_t tile = getTile(graph, input);
  graph.connect(vtx["input"], input);
  graph.connect(vtx["index"], index);
  graph.connect(vtx["output"], output);
  graph.setInitialValue(vtx["C"], numClasses);
  graph.setTileMapping(vtx, tile);
  graph.setPerfEstimate(vtx, 1);
}

struct NMSContext {
  // references
  poplar::Graph &graph_;
  // returned tensors
  poplar::Tensor topIndices, topScores, topBoxes, topClasses, lengths;
  // temporary tensors
  poplar::Tensor scoresCopy, indices;
  poplar::Tensor bestBox, bestClass;
  poplar::Tensor bestScoresCond, predicate, i;
  program::Program cond;
  // constants
  poplar::Tensor one;
  // options
  float iouThreshold_, scoreThreshold_, sigma_;
  int numDetections_;
  bool useClasses_;
  bool useGather_;
  poplar::Tensor boxesGather, classesGather;
  size_t batchSize;
  size_t N;

  NMSContext(poplar::Graph &graph, float iouThreshold, float scoreThreshold,
             float sigma, int numDetections, const poplar::DebugContext &dc,
             bool withClasses = true, bool useGather = false)
      : graph_{graph}, iouThreshold_{iouThreshold},
        scoreThreshold_{scoreThreshold}, sigma_{sigma},
        numDetections_{numDetections}, useClasses_{withClasses},
        useGather_{useGather} {
    one = graph.addConstant(poplar::UNSIGNED_INT, {}, 1, {dc, "one"});
    graph.setTileMapping(one, 1);
  }
  void init(program::Sequence &prog, const poplar::Tensor &scores,
            const poplar::Tensor &boxes, const poplar::Tensor &classes,
            bool inPlace, const poplar::DebugContext &dc) {
    batchSize = scores.dim(0);
    N = scores.dim(1);
    std::tie(scoresCopy, topIndices, topScores, topBoxes, topClasses, lengths) =
        initGraph(graph_, prog, scores, classes, numDetections_, inPlace, dc);
    std::tie(cond, bestScoresCond, predicate, i) =
        condition(graph_, prog, scores, numDetections_, scoreThreshold_, dc);
    bestBox = graph_.addVariable(boxes.elementType(), {batchSize, 4},
                                 poplar::VariableMappingMethod::LINEAR);
    if (useClasses_) {
      bestClass = graph_.addVariable(classes.elementType(), {batchSize},
                                     poplar::VariableMappingMethod::LINEAR);
    } else {
      bestClass = graph_.addConstant(classes.elementType(), {batchSize}, 0,
                                     "dummyBestClass");
      poputil::mapTensorLinearly(graph_, bestClass);
    }
    if (useGather_) {
      boxesGather = graph_.addVariable(topBoxes.elementType(),
                                       {batchSize, 4, N}, {dc, "boxesGather"});
      poputil::mapTensorLinearlyWithOffset(graph_, boxesGather, 1, N, 0, false);
      prog.add(
          program::Copy(boxes, boxesGather.dimShufflePartial({1, 2}, {2, 1})));
      size_t offset = batchSize * 4;
      if (useClasses_) {
        offset += batchSize;
        classesGather = graph_.addVariable(
            classes.elementType(), {batchSize, N}, {dc, "classesGather"});
        poputil::mapTensorLinearlyWithOffset(graph_, classesGather, 1, N,
                                             offset, false);
        prog.add(program::Copy(classes, classesGather));
      }
    }
  }

  void gatherBoxes(program::Sequence &loop, const poplar::Tensor &best_idx,
                   const poplar::Tensor &boxes, const poplar::Tensor &classes,
                   const poplar::DebugContext &dc) {
    if (useGather_) {
      poplar::ComputeSet cs = graph_.addComputeSet({dc, "gather"});
      for (size_t b = 0; b < batchSize; ++b) {
        // gather boxes
        connectSliceVertex(graph_, cs, boxesGather[b][0], best_idx[b],
                           bestBox[b][0]);
        connectSliceVertex(graph_, cs, boxesGather[b][1], best_idx[b],
                           bestBox[b][1]);
        connectSliceVertex(graph_, cs, boxesGather[b][2], best_idx[b],
                           bestBox[b][2]);
        connectSliceVertex(graph_, cs, boxesGather[b][3], best_idx[b],
                           bestBox[b][3]);
        if (useClasses_) {
          connectSliceVertex(graph_, cs, classesGather[b], best_idx[b],
                             bestClass[b]);
        }
      }
      loop.add(program::Execute(cs));
    } else {
      mapReduceGather(
          graph_, loop, graph_.getTileMapping(scoresCopy), scoresCopy.shape(),
          bestBox, bestClass,
          poputil::templateVertex("GatherVertex", scoresCopy.elementType(),
                                  bestClass.elementType()),
          best_idx,
          {{"classes", classes.flatten()}, {"boxes", boxes.flatten()}},
          useClasses_, {dc, "mapReduce"});
    }
  }

  void updateState(program::Sequence &loop, const poplar::Tensor &best_idx,
                   const poplar::Tensor &best_score,
                   const poplar::DebugContext &dc) {
    const auto mapping = graph_.getTileMapping(scoresCopy);
    poplar::ComputeSet cs = graph_.addComputeSet({dc, "updateBest"});
    mapVertex(
        graph_, {{"scores", scoresCopy.flatten()}}, {}, cs,
        poputil::templateVertex("UpdateBestVertex", scoresCopy.elementType()),
        mapping, 1, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
          std::vector<size_t> indices =
              unflattenRegion(mapping, tile, scoresCopy.shape());
          size_t batch = indices[0], indice = indices[1];
          graph_.connect(vtx["best"], best_idx[batch]);
          graph_.setInitialValue(vtx["offset"], indice);
          graph_.setInitialValue(vtx["C"], 1);
        });
    loop.add(program::Execute(cs));

    poplar::ComputeSet csTop = graph_.addComputeSet({dc, "updateAnswer"});
    mapVertex(graph_,
              {{"lengths", lengths.flatten()},
               {"top_indices", topIndices.flatten()},
               {"top_scores", topScores.flatten()},
               {"top_boxes", topBoxes.flatten()},
               {"top_classes", topClasses.flatten()}},
              {{"i", i}}, csTop,
              poputil::templateVertex("UpdateAnswerVertex",
                                      scoresCopy.elementType(),
                                      bestClass.elementType()),
              graph_.getTileMapping(topIndices.flatten()), 1,
              [&](VertexRef &vtx, uint32_t tile, uint32_t) {
                const std::vector<size_t> indices =
                    unflattenRegion(graph_.getTileMapping(topIndices.flatten()),
                                    tile, topIndices.shape());
                const size_t batch = indices[0];
                graph_.setInitialValue(vtx["score_threshold"], scoreThreshold_);
                graph_.setInitialValue(vtx["K"], numDetections_);
                graph_.setInitialValue(vtx["offset"], indices[1]);
                graph_.connect(vtx["best_indices"], best_idx[batch]);
                graph_.connect(vtx["best_scores"], best_score[batch]);
                graph_.connect(vtx["best_boxes"], bestBox[batch]);
                graph_.connect(vtx["best_classes"], bestClass[batch]);
              });
    loop.add(program::Execute(csTop));
  }
  void nms(program::Sequence &loop, const poplar::Tensor &boxes,
           const poplar::Tensor &classes, const poplar::DebugContext &dc) {
    poplar::ComputeSet csNms = graph_.addComputeSet({dc, "Nms"});
    const auto mapping = graph_.getTileMapping(scoresCopy);
    mapVertex(graph_,
              {{"scores", scoresCopy.flatten()},
               {"classes", classes.flatten()},
               {"boxes", boxes.flatten()}},
              {}, csNms,
              poputil::templateVertex("NmsVertex", scoresCopy.elementType(),
                                      bestClass.elementType()),
              mapping, 0, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
                std::vector<size_t> indices =
                    unflattenRegion(mapping, tile, scoresCopy.shape());
                size_t batch = indices[0];
                graph_.connect(vtx["bestClass"], bestClass[batch]);
                graph_.connect(vtx["bestBox"], bestBox[batch]);
                graph_.setInitialValue(vtx["sigma"], sigma_);
                graph_.setInitialValue(vtx["threshold"], iouThreshold_);
                graph_.setInitialValue(vtx["score_threshold"], scoreThreshold_);
              });
    loop.add(program::Execute(csNms));
  }
};

// class-less version
// for now with a dummy classes tensor.
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, float threshold, int num_detections,
    float score_threshold, float sigma, bool useGather, bool inPlace,
    const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, threshold, num_detections, inPlace));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 2);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t min_per_tile = 1;
  poputil::mapTensorLinearly(graph, scores, min_per_tile, 1);
  poplar::Tensor classes = graph.addConstant(poplar::UNSIGNED_INT,
                                             scores.shape(), 0, "dummyClasses");
  poputil::mapTensorLinearly(graph, classes, min_per_tile, 1);
  poputil::mapTensorLinearly(graph, boxes, min_per_tile, 4);
  NMSContext context{graph,          threshold, score_threshold, sigma,
                     num_detections, di,        false,           useGather};
  context.init(prog, scores, boxes, classes, inPlace, di);

  program::Sequence loop;
  poplar::Tensor best_score, best_idx;
  std::tie(best_score, best_idx) =
      popnn::maxAndArgMax(graph, context.scoresCopy, loop, {di, "maxArgMax"});
  loop.add(program::Copy(best_score, context.bestScoresCond));

  context.gatherBoxes(loop, best_idx, boxes, classes, di);
  context.updateState(loop, best_idx, best_score, di);
  context.nms(loop, boxes, classes, di);
  popops::addInPlace(graph, context.i, context.one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(context.cond, context.predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ =
      popops::reduce(graph, context.lengths, poplar::INT, {1}, params, prog,
                     {di, "reduceLengths"});
  return {context.topIndices, context.topScores, context.topBoxes, lengths_};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, const poplar::Tensor &classes, float threshold,
    int num_detections, float score_threshold, float sigma, bool useGather,
    bool inPlace, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, classes, threshold, num_detections, inPlace));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 2);
  assert(classes.rank() == 2);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t min_per_tile = 1;
  poputil::mapTensorLinearly(graph, scores, min_per_tile, 1);
  poputil::mapTensorLinearly(graph, classes, min_per_tile, 1);
  poputil::mapTensorLinearly(graph, boxes, min_per_tile, 4);
  NMSContext context{graph, threshold, score_threshold, sigma, num_detections,
                     di,    true,      useGather};
  context.init(prog, scores, boxes, classes, inPlace, di);

  program::Sequence loop;
  poplar::Tensor best_score, best_idx;
  std::tie(best_score, best_idx) =
      popnn::maxAndArgMax(graph, context.scoresCopy, loop, {di, "maxArgMax"});
  loop.add(program::Copy(best_score, context.bestScoresCond));

  context.gatherBoxes(loop, best_idx, boxes, classes, di);
  context.updateState(loop, best_idx, best_score, di);
  context.nms(loop, boxes, classes, di);

  popops::addInPlace(graph, context.i, context.one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(context.cond, context.predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ =
      popops::reduce(graph, context.lengths, poplar::INT, {1}, params, prog,
                     {di, "reduceLengths"});

  return {context.topIndices, context.topScores, context.topBoxes,
          context.topClasses, lengths_};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nmsMulti(poplar::Graph &graph, program::Sequence &prog,
         const poplar::Tensor &scores, const poplar::Tensor &boxes,
         float threshold, int num_detections, float score_threshold,
         float sigma, bool useGather, bool inPlace,
         const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, threshold, num_detections, inPlace));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 3);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t min_per_tile = 1;
  const size_t bs = scores.dim(0);
  const size_t N = scores.dim(1);
  const size_t C = scores.dim(2);
  poputil::mapTensorLinearly(graph, scores, min_per_tile, C);
  poputil::mapTensorLinearly(graph, boxes, min_per_tile, 4);
  poplar::Tensor dummy = graph.addVariable(boxes.elementType(), {bs, N});
  poputil::mapTensorLinearly(graph, dummy, min_per_tile, 1);
  const auto mapping = graph.getTileMapping(dummy);
  const auto shape = dummy.shape();
  poplar::Tensor boxesGather;
  poplar::Tensor scores_copy, topIndices, topScores, topBoxes, topClasses,
      lengths;
  std::tie(scores_copy, topIndices, topScores, topBoxes, topClasses, lengths) =
      initGraphMulti(graph, prog, scores, num_detections, inPlace, di);
  if (useGather) {
    boxesGather = graph.addVariable(topBoxes.elementType(), {bs, 4, N},
                                    {di, "boxesGather"});
    poputil::mapTensorLinearlyWithOffset(graph, boxesGather, 1, N, 0, false);
    prog.add(
        program::Copy(boxes, boxesGather.dimShufflePartial({1, 2}, {2, 1})));
  }

  poplar::Tensor one =
      graph.addConstant(poplar::UNSIGNED_INT, {}, 1, {di, "one"});
  graph.setTileMapping(one, 1);
  program::Program cond;
  poplar::Tensor bestScores, predicate, i;
  std::tie(cond, bestScores, predicate, i) =
      condition(graph, prog, scores, num_detections, score_threshold, di);
  program::Sequence loop;
  poplar::Tensor best_score, best_idx;
  std::tie(best_score, best_idx) = popnn::maxAndArgMax(
      graph, scores_copy.reshape({bs, N * C}), loop, {di, "maxArgMax"});
  loop.add(program::Copy(best_score, bestScores));
  poplar::Tensor best_box =
      graph.addVariable(boxes.elementType(), {scores.dim(0), 4},
                        poplar::VariableMappingMethod::LINEAR);
  if (useGather) {
    poplar::ComputeSet csGather = graph.addComputeSet({dc, "gather"});
    for (size_t b = 0; b < bs; ++b) {
      // gather boxes
      connectSliceVertex(graph, csGather, boxesGather[b][0], best_idx[b],
                         best_box[b][0], C);
      connectSliceVertex(graph, csGather, boxesGather[b][1], best_idx[b],
                         best_box[b][1], C);
      connectSliceVertex(graph, csGather, boxesGather[b][2], best_idx[b],
                         best_box[b][2], C);
      connectSliceVertex(graph, csGather, boxesGather[b][3], best_idx[b],
                         best_box[b][3], C);
    }
    loop.add(program::Execute(csGather));
  } else {
    mapReduceGatherMulti(
        graph, loop, graph.getTileMapping(dummy), dummy.shape(), best_box, C,
        poputil::templateVertex("GatherMultiVertex", scores.elementType(),
                                best_idx.elementType()),
        best_idx, {{"boxes", boxes.flatten()}}, {di, "mapReduceBoxes"});
  }
  poplar::ComputeSet cs = graph.addComputeSet({di, "updateBest"});
  mapVertex(graph, {{"scores", scores_copy.flatten()}}, {}, cs,
            poputil::templateVertex("UpdateBestVertex", scores.elementType()),
            mapping, 1, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
              std::vector<size_t> indices =
                  unflattenRegion(mapping, tile, shape);
              size_t batch = indices[0], indice = indices[1];
              graph.connect(vtx["best"], best_idx[batch]);
              graph.setInitialValue(vtx["offset"], indice);
              graph.setInitialValue(vtx["C"], C);
            });
  loop.add(program::Execute(cs));

  poplar::ComputeSet csTop = graph.addComputeSet({di, "updateAnswer"});
  mapVertex(graph,
            {{"top_indices", topIndices.flatten()},
             {"lengths", lengths.flatten()},
             {"top_scores", topScores.flatten()},
             {"top_boxes", topBoxes.flatten()},
             {"top_classes", topClasses.flatten()}},
            {{"i", i}}, csTop,
            poputil::templateVertex("UpdateAnswerMultiVertex",
                                    scores.elementType(),
                                    best_idx.elementType()),
            graph.getTileMapping(topIndices.flatten()), 1,
            [&](VertexRef &vtx, uint32_t tile, uint32_t) {
              const std::vector<size_t> indices =
                  unflattenRegion(graph.getTileMapping(topIndices.flatten()),
                                  tile, topIndices.shape());
              const size_t batch = indices[0];
              graph.setInitialValue(vtx["score_threshold"], score_threshold);
              graph.setInitialValue(vtx["K"], num_detections);
              graph.setInitialValue(vtx["C"], C);
              graph.setInitialValue(vtx["offset"], indices[1]);
              graph.connect(vtx["best_indices"], best_idx[batch]);
              graph.connect(vtx["best_scores"], best_score[batch]);
              graph.connect(vtx["best_boxes"], best_box[batch]);
            });
  loop.add(program::Execute(csTop));

  poplar::ComputeSet csNms = graph.addComputeSet({di, "Nms"});
  mapVertex(
      graph, {{"boxes", boxes.flatten()}, {"scores", scores_copy.flatten()}},
      {}, csNms,
      poputil::templateVertex("NmsMultiVertex", scores.elementType()), mapping,
      0, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
        std::vector<size_t> indices = unflattenRegion(mapping, tile, shape);
        size_t batch = indices[0];
        graph.connect(vtx["bestBox"], best_box[batch]);
        graph.connect(vtx["bestIndices"], best_idx[batch]);
        graph.setInitialValue(vtx["sigma"], sigma);
        graph.setInitialValue(vtx["threshold"], threshold);
        graph.setInitialValue(vtx["C"], C);
        graph.setInitialValue(vtx["score_threshold"], score_threshold);
      });
  loop.add(program::Execute(csNms));

  popops::addInPlace(graph, i, one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(cond, predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ = popops::reduce(graph, lengths, poplar::INT, {1},
                                           params, prog, {di, "reduceLengths"});

  return {topIndices, topScores, topBoxes, topClasses, lengths_};
}

struct TileContext {
  uint32_t tileNumber, index, N;
  Interval region;
  poplar::Tensor tree;
};

uint32_t nearestPower2(uint32_t v) {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  ++v;
  return v;
}

struct NMSLargeContext {
  // references
  poplar::Graph &graph_;
  // returned tensors
  poplar::Tensor topIndices, topScores, topBoxes, topClasses, lengths;
  // temporary tensors
  poplar::Tensor topks, max, argmax;
  poplar::Tensor bestBox;
  poplar::Tensor bestScoresCond, predicate, i;
  program::Program cond;
  // constants
  poplar::Tensor one;
  // options
  float iouThreshold_, scoreThreshold_;
  int numDetections_;
  poplar::Tensor boxesGather, classesGather;
  size_t batchSize, N, C, K;
  bool useGather_;
  std::vector<TileContext> tiles;
  Mapping baseMapping;
  std::vector<size_t> baseShape;
  bool needTrees;

  NMSLargeContext(poplar::Graph &graph, float iouThreshold,
                  float scoreThreshold, int numDetections,
                  const poplar::DebugContext &dc, uint32_t topk = 1,
                  bool useGather = false)
      : graph_{graph}, iouThreshold_{iouThreshold},
        scoreThreshold_{scoreThreshold}, numDetections_{numDetections}, K{topk},
        useGather_{useGather} {
    one = graph.addConstant(poplar::UNSIGNED_INT, {}, 1, {dc, "one"});
    graph.setTileMapping(one, 1);
  }

  bool initTiles(const poplar::DebugContext &dc) {
    bool needTrees = false;
    for (size_t i = 0; i < baseMapping.size(); ++i) {
      if (baseMapping[i].size() > 0) {
        TileContext tc;
        assert(baseMapping[i].size() == 1);
        const auto region = baseMapping[i].front();
        uint32_t localN = region.size();
        tc.tileNumber = i;
        tc.N = localN;
        tc.region = region;
        tc.index = tiles.size();
        if (localN > 1) {
          needTrees = true;
          tc.tree = graph_.addVariable(topks.elementType(),
                                       {nearestPower2(localN) - 1},
                                       {dc, "tree_" + std::to_string(i)});
          graph_.setTileMapping(tc.tree, i);
        }
        tiles.push_back(tc);
      }
    }
    return needTrees;
  }

  void init(program::Sequence &prog, const poplar::Tensor &scores,
            const poplar::Tensor &boxes, poplar::Type indexType,
            uint32_t minPerTile, const poplar::DebugContext &dc) {
    batchSize = scores.dim(0);
    N = scores.dim(1);
    C = scores.dim(2);
    poputil::mapTensorLinearly(graph_, scores, 1, C);
    poputil::mapTensorLinearly(graph_, boxes, 1, 4);
    baseMapping =
        poputil::calcLinearTileMapping(graph_, {batchSize, N}, minPerTile, 1);
    baseShape = {batchSize, N, 1};
    std::tie(topks, max, argmax, topIndices, topScores, topBoxes, topClasses,
             lengths) =
        initGraphMultiLarge(graph_, prog, scores, numDetections_, indexType, K,
                            minPerTile, dc);
    needTrees = initTiles(dc);
    std::tie(cond, bestScoresCond, predicate, i) =
        condition(graph_, prog, scores, numDetections_, scoreThreshold_, dc);
    bestBox = graph_.addVariable(boxes.elementType(), {batchSize, 4},
                                 poplar::VariableMappingMethod::LINEAR);
    if (useGather_) {
      boxesGather = graph_.addVariable(topBoxes.elementType(),
                                       {batchSize, 4, N}, {dc, "boxesGather"});
      poputil::mapTensorLinearlyWithOffset(graph_, boxesGather, 1, N, 0, false);
      prog.add(
          program::Copy(boxes, boxesGather.dimShufflePartial({1, 2}, {2, 1})));
    }
  }

  void prepare(program::Sequence &prog, const poplar::Tensor &scores,
               const poplar::DebugContext &dc) {
    // prog.add(program::PrintTensor("scores", scores));
    {
      poplar::ComputeSet cs = graph_.addComputeSet({dc, "topk"});
      mapVertex(
          graph_, {{"input", scores.flatten()}, {"topk", topks.flatten()}}, {},
          cs,
          poputil::templateVertex("TopKVertex", scores.elementType(),
                                  topks.elementType()),
          baseMapping, 0, [&](VertexRef &vtx, uint32_t tile, uint32_t index) {
            graph_.setInitialValue(vtx["score_threshold"], scoreThreshold_);
            graph_.setInitialValue(vtx["C"], C);
            graph_.setInitialValue(vtx["K"], K);
          });
      prog.add(program::Execute(cs));
    }
    // prog.add(program::PrintTensor("post topk", topks));
    if (needTrees) {
      poplar::ComputeSet cs = graph_.addComputeSet({dc, "buildTree"});
      for (const TileContext &t : tiles) {
        if (t.N > 1) {
          VertexRef vtx = graph_.addVertex(
              cs, poputil::templateVertex("BuildWinnerTreeVertex",
                                          scores.elementType(),
                                          topks.elementType()));
          graph_.setTileMapping(vtx, t.tileNumber);
          graph_.setInitialValue(vtx["C"], C);
          graph_.setInitialValue(vtx["K"], K);
          graph_.connect(vtx["tree"], t.tree);
          graph_.connect(
              vtx["input"],
              scores.flatten().slice(t.region.begin() * C, t.region.end() * C));
          graph_.connect(
              vtx["topk"],
              topks.flatten().slice(t.region.begin() * K, t.region.end() * K));
          graph_.setPerfEstimate(vtx, t.N);
        }
      }
      prog.add(program::Execute(cs));
      // for (const TileContext &t : tiles) {
      //   if (t.N > 1)
      //     prog.add(program::PrintTensor(
      //         "initial tree_" + std::to_string(t.tileNumber), t.tree));
      // }
    }
  }

  void maxargmax(program::Sequence &loop, const poplar::Tensor &scores,
                 const poplar::DebugContext &dc) {
    poplar::ComputeSet cs = graph_.addComputeSet({dc, "maxArgmax"});
    for (const TileContext &t : tiles) {
      VertexRef vtx;
      std::vector<size_t> indices =
          poputil::unflattenIndex(baseShape, t.region.begin());
      if (t.N > 1) {
        vtx = graph_.addVertex(
            cs, poputil::templateVertex("MaxTreeVertex", scores.elementType(),
                                        topks.elementType()));
        graph_.connect(vtx["tree"], t.tree);
        graph_.setInitialValue(vtx["C"], C);
      } else {
        assert(t.N == 1);
        vtx = graph_.addVertex(
            cs, poputil::templateVertex("MaxTopkVertex", scores.elementType(),
                                        topks.elementType()));
      }
      graph_.setTileMapping(vtx, t.tileNumber);
      graph_.setInitialValue(vtx["K"], K);
      graph_.connect(vtx["input"], scores.flatten().slice(t.region.begin() * C,
                                                          t.region.end() * C));
      graph_.connect(vtx["topk"], topks.flatten().slice(t.region.begin() * K,
                                                        t.region.end() * K));
      graph_.setPerfEstimate(vtx, t.N);
      graph_.connect(vtx["max"], max[t.index][0]);
      graph_.connect(vtx["argmax"], argmax[t.index][0]);
      graph_.setInitialValue(vtx["offset"], indices[1] * C);
    }
    loop.add(program::Execute(cs));
    // for (const TileContext &t : tiles) {
    //   if (t.N > 1)
    //     loop.add(program::PrintTensor("tree_" +
    //     std::to_string(t.tileNumber),
    //                                   t.tree));
    // }
  }

  void reducemax(program::Sequence &loop, poplar::Tensor &bestScores,
                 poplar::Tensor &bestIdx, const poplar::DebugContext &dc) {
    std::vector<std::vector<poplar::Tensor>> batchMax(batchSize),
        batchIndices(batchSize);

    for (const auto &t : tiles) {
      std::vector<size_t> indices =
          poputil::unflattenIndex(baseShape, t.region.begin());
      size_t batch = indices[0];
      batchIndices[batch].push_back(argmax[t.index].reshape({1, 1}));
      batchMax[batch].push_back(max[t.index].reshape({1, 1}));
    }
    std::vector<poplar::Tensor> maxTensors, indTensors;
    for (uint32_t b = 0; b < batchSize; ++b) {
      maxTensors.push_back(poplar::concat(batchMax[b], 1));
      indTensors.push_back(poplar::concat(batchIndices[b], 1));
    }
    poplar::Tensor maxTensor = poplar::concat(maxTensors, 0);
    poplar::Tensor indTensor = poplar::concat(indTensors, 0);
    poplar::Tensor max_, argmax_;
    std::tie(max_, argmax_) = localMaxAndArgMax(
        graph_, maxTensor, indTensor, bestScores.elementType(), loop, dc);
    loop.add(program::Copy(max_, bestScores));
    loop.add(program::Copy(argmax_, bestIdx));
  }

  void gatherBoxes(program::Sequence &loop, const poplar::Tensor &best_idx,
                   const poplar::Tensor &boxes,
                   const poplar::DebugContext &dc) {
    if (useGather_) {
      poplar::ComputeSet cs = graph_.addComputeSet({dc, "gather"});
      for (size_t b = 0; b < batchSize; ++b) {
        // gather boxes
        connectSliceVertex(graph_, cs, boxesGather[b][0], best_idx[b],
                           bestBox[b][0]);
        connectSliceVertex(graph_, cs, boxesGather[b][1], best_idx[b],
                           bestBox[b][1]);
        connectSliceVertex(graph_, cs, boxesGather[b][2], best_idx[b],
                           bestBox[b][2]);
        connectSliceVertex(graph_, cs, boxesGather[b][3], best_idx[b],
                           bestBox[b][3]);
      }
      loop.add(program::Execute(cs));
    } else {
      mapReduceGatherMulti(
          graph_, loop, baseMapping, baseShape, bestBox, C,
          poputil::templateVertex("GatherMultiVertex", boxes.elementType(),
                                  argmax.elementType()),
          best_idx, {{"boxes", boxes.flatten()}}, {dc, "mapReduceBoxes"});
    }
  }
  void updateState(program::Sequence &loop, const poplar::Tensor &best_idx,
                   const poplar::Tensor &best_score,
                   const poplar::DebugContext &dc) {
    poplar::ComputeSet cs = graph_.addComputeSet({dc, "updateState"});
    mapVertex(
        graph_, {{"topk", topks.flatten()}}, {}, cs,
        poputil::templateVertex("UpdateBestTreeVertex", topks.elementType()),
        baseMapping, 1, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
          std::vector<size_t> indices =
              unflattenRegion(baseMapping, tile, baseShape);
          size_t batch = indices[0], indice = indices[1];
          graph_.connect(vtx["best"], best_idx[batch]);
          graph_.setInitialValue(vtx["offset"], indice);
          graph_.setInitialValue(vtx["C"], C);
          graph_.setInitialValue(vtx["K"], K);
        });

    mapVertex(graph_,
              {{"top_indices", topIndices.flatten()},
               {"lengths", lengths.flatten()},
               {"top_scores", topScores.flatten()},
               {"top_boxes", topBoxes.flatten()},
               {"top_classes", topClasses.flatten()}},
              {{"i", i}}, cs,
              poputil::templateVertex("UpdateAnswerMultiVertex",
                                      best_score.elementType(),
                                      argmax.elementType()),
              graph_.getTileMapping(topIndices.flatten()), 1,
              [&](VertexRef &vtx, uint32_t tile, uint32_t) {
                const std::vector<size_t> indices =
                    unflattenRegion(graph_.getTileMapping(topIndices.flatten()),
                                    tile, topIndices.shape());
                const size_t batch = indices[0];
                graph_.setInitialValue(vtx["score_threshold"], scoreThreshold_);
                graph_.setInitialValue(vtx["K"], numDetections_);
                graph_.setInitialValue(vtx["C"], C);
                graph_.setInitialValue(vtx["offset"], indices[1]);
                graph_.connect(vtx["best_indices"], best_idx[batch]);
                graph_.connect(vtx["best_scores"], best_score[batch]);
                graph_.connect(vtx["best_boxes"], bestBox[batch]);
              });

    loop.add(program::Execute(cs));
  }
  void nms(program::Sequence &loop, const poplar::Tensor &boxes,
           const poplar::Tensor &bestIdx, const poplar::DebugContext &dc) {
    poplar::ComputeSet cs = graph_.addComputeSet({dc, "Nms"});
    mapVertex(
        graph_, {{"topk", topks.flatten()}, {"boxes", boxes.flatten()}}, {}, cs,
        poputil::templateVertex("NmsTreeVertex", boxes.elementType(),
                                topks.elementType()),
        baseMapping, 0, [&](VertexRef &vtx, uint32_t tile, uint32_t index) {
          std::vector<size_t> indices =
              unflattenRegion(baseMapping, tile, baseShape);
          const size_t batch = indices[0];
          graph_.connect(vtx["bestIdx"], bestIdx[batch]);
          graph_.setInitialValue(vtx["C"], C);
          graph_.setInitialValue(vtx["K"], K);
          graph_.connect(vtx["bestBox"], bestBox[batch]);
          graph_.setInitialValue(vtx["threshold"], iouThreshold_);
        });
    loop.add(program::Execute(cs));
  }
};

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nmsMultiLarge(poplar::Graph &graph, program::Sequence &prog,
              const poplar::Tensor &scores, const poplar::Tensor &boxes,
              float threshold, int num_detections, float score_threshold,
              bool useGather, uint32_t topk, const poplar::DebugContext &dc,
              uint32_t minPerTile) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, threshold, num_detections));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 3);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t C = scores.dim(2);
  poputil::mapTensorLinearly(graph, scores, minPerTile, C);
  poputil::mapTensorLinearly(graph, boxes, minPerTile, 4);
  NMSLargeContext context{graph, threshold, score_threshold, num_detections,
                          di,    topk,      useGather};
  context.init(prog, scores, boxes, poplar::UNSIGNED_SHORT, minPerTile, di);
  context.prepare(prog, scores, di);
  program::Sequence loop;
  context.maxargmax(loop, scores, di);
  // loop.add(program::PrintTensor("max", context.max));
  // loop.add(program::PrintTensor("argmax", context.argmax));
  poplar::Tensor bestIdx = graph.addVariable(
      context.argmax.elementType(), {context.batchSize}, {di, "bestIndex"});
  graph.setTileMapping(bestIdx, 1);

  context.reducemax(loop, context.bestScoresCond, bestIdx, di);
  // loop.add(program::PrintTensor("bestscores", context.bestScoresCond));
  // loop.add(program::PrintTensor("bestidx", bestIdx));
  context.gatherBoxes(loop, bestIdx, boxes, di);
  // loop.add(program::PrintTensor("boxes", boxes));
  // loop.add(program::PrintTensor("bestbox", context.bestBox));
  context.updateState(loop, bestIdx, context.bestScoresCond, di);
  // loop.add(program::PrintTensor("pre nms topk", context.topks));
  context.nms(loop, boxes, bestIdx, di);
  // loop.add(program::PrintTensor("post nms topk", context.topks));

  popops::addInPlace(graph, context.i, context.one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(context.cond, context.predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ =
      popops::reduce(graph, context.lengths, poplar::INT, {1}, params, prog,
                     {di, "reduceLengths"});

  return {context.topIndices, context.topScores, context.topBoxes,
          context.topClasses, lengths_};
}
