#include "legacynms.hpp"
#include "ipu_utils.hpp"
#include "poputil/DebugInfo.hpp"
#include <poplar/Program.hpp>
#include <popnn/Loss.hpp>
#include <popnn/codelets.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>

using namespace poplar;

program::Program create_nms_core_program(Graph &graph, std::string debugMsg,
                                         Tensor &Keep, Tensor &Score,
                                         Tensor &Box, Tensor &Finish,
                                         Tensor &Box_i, Tensor &Score_2D_idx,
                                         float threshold, int bs, int vlength) {
  std::cerr << "Debug create_nms_core_program\n";
  program::Sequence prog;
  ComputeSet Update_CS =
      graph.addComputeSet(poputil::templateVertex(debugMsg + "Update_CS"));

  int tile = 0;
  int numTiles = graph.getTarget().getNumTiles();

  for (int sample = 0; sample < bs; sample++) {
    for (unsigned int idx = 0; idx < vlength; idx++) {
      auto v = graph.addVertex(
          Update_CS,
          poputil::templateVertex("NmsCoreVertex", Box.elementType()),
          {{"idx", idx},
           {"nms_thresh", threshold},
           {"sorted_index", Score_2D_idx[sample]},
           {"keep_r", Keep[sample][idx]},
           {"score_r", Score[sample][idx]},
           {"box_r", Box[sample][idx]},
           {"box_i", Box_i[sample]},
           {"finish_r", Finish[sample]}});

      graph.setTileMapping(v, tile);
      graph.setTileMapping(Keep[sample][idx], tile);
      graph.setTileMapping(Score[sample][idx], tile);
      graph.setTileMapping(Box[sample][idx], tile);
      graph.setTileMapping(Box_i[sample], tile);
      graph.setTileMapping(Finish[sample], tile);
      graph.setPerfEstimate(v, Box_i[sample].numElements());
      if (tile + 1 >= numTiles)
        tile = 0;
      else
        tile++;
    }
  }
  prog.add(program::Execute(Update_CS));
  auto Score_2D = Score.reshape({bs, vlength});
  return prog;
}

program::Program
fetch_set_result_program(Graph &graph, std::string debugMsg, Tensor &Score,
                         Tensor &Box, Tensor &Keep, Tensor &ITensor,
                         Tensor &Box_i, Tensor &result, Tensor &resultbox,
                         Tensor &value_index, Tensor &Score_2D_idx, int bs,
                         int vlength, int numDetections) {
  std::cerr << "Debug fetch_set_result_program\n";
  program::Sequence prog;
  ComputeSet Calc_NumNonZero_CS = graph.addComputeSet(
      poputil::templateVertex(debugMsg + "Calc_NumNonZeros_CS"));
  ComputeSet Set_Result_CS =
      graph.addComputeSet(poputil::templateVertex(debugMsg + "SetIthKeep_CS"));
  ComputeSet Result_Index_CS =
      graph.addComputeSet(poputil::templateVertex(debugMsg + "ResultIndex_CS"));
  ComputeSet Fetch_ielements_CS = graph.addComputeSet(
      poputil::templateVertex(debugMsg + "Fetch_ielements_CS"));
  // graph.setTileMapping(Scores_i, 1);
  graph.setTileMapping(Box_i, 1);

  int numTiles = graph.getTarget().getNumTiles();
  // numTiles = 1216;
  // numTiles = 4;
  // reshape cause pass3 codelets require 2D tensors
  Tensor Score_2D =
      Score.reshape({(long unsigned int)bs, (long unsigned int)vlength});
  Tensor Box_2D =
      Box.reshape({(long unsigned int)bs, (long unsigned int)vlength * 4});
  int numWorkers = graph.getTarget().getNumWorkerContexts() *
                   graph.getTarget().getNumTiles();

  int L = vlength;
  numWorkers = 1216;
  int numRowsPerWorker = (L + numWorkers - 1) / numWorkers;
  int numVertices = L / numRowsPerWorker + 1;
  int tile = 0;

  // Complete the tile mapping for iTensor (otherwise there is compilation
  // error)
  for (int sample = 0; sample < bs; sample++)
    graph.setTileMapping(ITensor[sample], 9);

  tile = 0;
  for (int sample = 0; sample < bs; sample++)
    for (int idx = 0; idx < vlength; idx++) {
      graph.setTileMapping(Score[sample][idx], tile);
      graph.setTileMapping(Box[sample][idx], tile);

      if (tile + 1 >= numTiles)
        tile = 0;
      else
        tile++;
    }

  poplar::Tensor Box_dist_Tensor =
      graph.addVariable(Box.elementType(), {numVertices, bs, 4}, "Box_dist_T");

  for (int i = 0; i < numVertices; i++) {
    int rowStart = i * numRowsPerWorker * 4;
    int rowEnd = std::min(L * 4, rowStart + numRowsPerWorker * 4);
    poplar::Tensor workerBox = Box_2D.slice(rowStart, rowEnd, 1);
    graph.setTileMapping(workerBox, tile);
    graph.setTileMapping(Box_dist_Tensor[i], tile);
    poplar::VertexRef v3 = graph.addVertex(
        Fetch_ielements_CS,
        poputil::templateVertex("PartialFetchBoxVertex", Box.elementType()),
        {{"in_row_start", rowStart},
         {"in_row_end", rowEnd},
         {"in_tensor", workerBox},
         {"sorted_index", Score_2D_idx[0]},
         {"length", rowEnd - rowStart},
         {"out_val", Box_dist_Tensor[i]},
         {"batch_size", bs}});
    graph.setTileMapping(v3, tile);
    graph.setPerfEstimate(v3, workerBox.numElements());
    if (tile + 1 >= numTiles)
      tile = 0;
    else
      tile++;
  }
  prog.add(program::Execute(Fetch_ielements_CS)); // 4 Vertices
  Box_i = popops::reduce(graph, Box_dist_Tensor, Box.elementType(), {0},
                         popops::Operation::ADD, prog,
                         debugMsg + "FetchVertexOutValBox");

  for (int sample = 0; sample < bs; sample++)
    graph.setTileMapping(Box_i[sample], 6);

  resultbox = resultbox.reshape({bs, 4 * numDetections});
  for (int i = 0; i < numDetections; i++) {
    int rowStart = i;
    int rowEnd = std::min(L, rowStart + 1);
    // std::cout << numVertices << " for worker i:" << i << ", in tileId:" <<
    // tile << ", rowStart:" << rowStart << ", rowEnd:" << rowEnd << std::endl;

    poplar::Tensor worker_ResultBox =
        resultbox.slice(rowStart * 4, rowEnd * 4, 1);
    poplar::Tensor worker_result_idx = result.slice(rowStart, rowEnd, 1);
    graph.setTileMapping(worker_ResultBox, tile);
    graph.setTileMapping(worker_result_idx, tile);
    poplar::VertexRef v3 = graph.addVertex(
        Set_Result_CS,
        poputil::templateVertex("PartialSetIthKeepVertex", Box.elementType()),
        {{"in_row_start", rowStart},
         {"in_row_end", rowEnd},
         {"result", worker_result_idx},
         {"resultbox", worker_ResultBox},
         {"sorted_index", Score_2D_idx},
         {"box_i", Box_i},
         {"batch_size", bs},
         {"index", value_index}});
    graph.setTileMapping(v3, tile);
    graph.setPerfEstimate(v3, bs * 10);
    if (tile + 1 >= numTiles)
      tile = 0;
    else
      tile++;
  }
  graph.setTileMapping(value_index, 3);
  prog.add(program::Execute(Set_Result_CS)); // 1 Vertex)
  poplar::VertexRef value_index_add_one =
      graph.addVertex(Result_Index_CS, poputil::templateVertex("addOneVertex"),
                      {{"batch_size", bs}, {"var", value_index}});
  graph.setTileMapping(value_index_add_one, tile);
  graph.setPerfEstimate(value_index_add_one, bs);
  prog.add(program::Execute(Result_Index_CS));
  graph.setTileMapping(value_index, 3);
  return prog;
}

program::Program init_iter_round_program(Graph &graph, std::string debugMsg,
                                         Tensor &Keep, Tensor &Score,
                                         Tensor NumOnes, Tensor NumNonZeros,
                                         Tensor iTensor, Tensor Finish,
                                         Tensor &Score_2D_idx, int bs,
                                         int vlength) {
  std::cerr << "Debug init_iter_round_program\n";
  program::Sequence prog;
  ComputeSet Counting_CS =
      graph.addComputeSet(poputil::templateVertex(debugMsg + "Counting_CS"));
  ComputeSet Criterion_CS =
      graph.addComputeSet(poputil::templateVertex(debugMsg + "Criterion_CS"));

  // unsigned int numWorkers = graph.getTarget().getNumWorkerContexts() *
  // graph.getTarget().getNumTiles();
  int numTiles = graph.getTarget().getNumTiles();
  int numWorkers = numTiles;
  unsigned int L = vlength;

  unsigned int numRowsPerWorker = (L + numWorkers - 1) / numWorkers;
  unsigned int numVertices = L / numRowsPerWorker + 1;
  unsigned int tile = 0;

  Tensor Score_2D =
      Score.reshape({(long unsigned int)bs, (long unsigned int)vlength});

  poplar::Tensor Ones_dist_Tensor =
      graph.addVariable(INT, {numVertices, bs}, "Ones_dist_T");
  poplar::Tensor NZs_dist_Tensor =
      graph.addVariable(INT, {numVertices, bs}, "NZs_dist_T");
  graph.setTileMapping(NumOnes, 10);
  graph.setTileMapping(NumNonZeros, 1);
  for (size_t i = 0; i < numVertices; i++) {
    unsigned int rowStart = i * numRowsPerWorker;
    unsigned int rowEnd = std::min(L, rowStart + numRowsPerWorker);
    unsigned int length = rowEnd - rowStart;
    poplar::Tensor workerScore = Score_2D.slice(rowStart, rowEnd, 1);
    graph.setTileMapping(workerScore, tile);
    graph.setTileMapping(Ones_dist_Tensor[i], tile);
    graph.setTileMapping(NZs_dist_Tensor[i], tile);

    auto v1 = graph.addVertex(
        Counting_CS,
        poputil::templateVertex("CalcNzVertex", Score.elementType()),
        {{"InputVector", workerScore},
         {"length", length},
         {"batch_size", bs},
         {"v_num_nonzero", NZs_dist_Tensor[i]}});
    graph.setTileMapping(v1, tile);
    graph.setPerfEstimate(v1, workerScore.numElements());
    if (tile + 1 >= numTiles)
      tile = 0;
    else
      tile++;
  }
  prog.add(program::Execute(Counting_CS));
  NumNonZeros = popops::reduce(graph, NZs_dist_Tensor, poplar::INT, {0},
                               popops::Operation::ADD, prog,
                               debugMsg + "CalcNzVertex_nzs");
  graph.setTileMapping(Score_2D_idx, 1);
  Score_2D_idx = popnn::argMax(graph, Score_2D, prog, debugMsg + "argmax");
  Score_2D_idx = Score_2D_idx.reshape({bs, 1});

  auto v2 = graph.addVertex(
      Criterion_CS,
      poputil::templateVertex("UpdateStateVertex", Score.elementType()),
      {
          {"num_nonzeros_in_scores", NumNonZeros},
          {"batch_size", bs},
          {"iTensor", iTensor},
          {"finish", Finish},
      });
  graph.setTileMapping(v2, 4);
  graph.setPerfEstimate(v2, iTensor.numElements());

  tile = 0;
  for (int sample = 0; sample < bs; sample++)
    for (int idx = 0; idx < vlength; idx++) {
      graph.setTileMapping(Score[sample][idx], tile);

      if (tile + 1 >= numTiles)
        tile = 0;
      else
        tile++;
    }
  graph.setTileMapping(iTensor, 3);
  graph.setTileMapping(Finish, 8);

  prog.add(program::Execute(Criterion_CS));
  return prog;
}

void init_params(Graph &graph, std::string debugMsg, program::Sequence &prog,
                 poplar::Tensor &value_index, poplar::Tensor &keep, int bs,
                 int vlength) {
  std::cerr << "Debug init_params\n";
  const auto initializePhaseCS =
      graph.addComputeSet(debugMsg + "initializePhaseCS");
  unsigned int numWorkers = graph.getTarget().getNumWorkerContexts() *
                            graph.getTarget().getNumTiles();
  int numTiles = graph.getTarget().getNumTiles();
  unsigned int L = vlength * bs;
  numWorkers = 1216;
  unsigned int numRowsPerWorker = (L + numWorkers - 1) / numWorkers;
  unsigned int numVertices = L / numRowsPerWorker + 1;
  unsigned int tile = 0;
  keep = keep.reshape({bs * vlength});
  for (int i = 0; i < numVertices; ++i) {
    unsigned int rowStart = i * numRowsPerWorker;
    unsigned int rowEnd = std::min(L, rowStart + numRowsPerWorker);
    poplar::Tensor workerKeep = keep.slice(rowStart, rowEnd);
    graph.setTileMapping(workerKeep, tile);
    poplar::VertexRef fillTrueVertex = graph.addVertex(
        initializePhaseCS, poputil::templateVertex("fillTrueVertex"),
        {
            {"keep", workerKeep} // Input
        });
    graph.setTileMapping(fillTrueVertex, tile);
    graph.setPerfEstimate(fillTrueVertex, value_index.numElements());
    if (tile + 1 >= numTiles)
      tile = 0;
    else
      tile++;
  }
  keep = keep.reshape({bs, vlength, 1});
  poplar::VertexRef fillZeroVertex = graph.addVertex(
      initializePhaseCS, poputil::templateVertex("fillZeroVertex"),
      {
          {"var", value_index} // InOut
      });
  graph.setTileMapping(fillZeroVertex, 2);
  graph.setPerfEstimate(fillZeroVertex, value_index.numElements());
  prog.add(poplar::program::Execute(initializePhaseCS));
}

// the forward process
poplar::Tensor build_nms(poplar::Graph &graph, poplar::program::Sequence &prog,
                         const poplar::Tensor &scores,
                         const poplar::Tensor &boxes, uint32_t numDetections,
                         float threshold) {
  // long unsigned int N = op.;   // N = 5 * top_n, assuming top_n = 100
  poplar::DebugContext dc{"legacynms"};
  graph.addCodelets("nms_lower_codelet.cpp");
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  std::string debugMsg = "nmsOpx::grow";
  poplar::Tensor Score_Tensor = scores; // {B, L}
  poplar::Tensor Box_Tensor = boxes;    // {B, L, 4}

  auto score_shape = Score_Tensor.shape();
  long unsigned int Bs = score_shape[0];
  long unsigned int N = score_shape[1];
  Score_Tensor = Score_Tensor.reshape({Bs, score_shape[1], 1});

  poplar::Tensor Score_2D_idx =
      graph.addVariable(UNSIGNED_INT, {Bs, 1}, {dc, "score_2d_idx"});
  poplar::Tensor Keep_Tensor =
      graph.addVariable(FLOAT, {Bs, N, 1}, {dc, "Keep_T"});
  poplar::Tensor result_Tensor =
      graph.addVariable(UNSIGNED_INT, {Bs, numDetections}, {dc, "Result_T"});
  poplar::Tensor resultbox_Tensor_tmp =
      graph.addVariable(Box_Tensor.elementType(), {Bs, numDetections, 4},
                        {dc, "ResultBox_T_tmp"});
  graph.setTileMapping(resultbox_Tensor_tmp, 0);
  poplar::Tensor resultbox_Tensor = graph.addVariable(
      Box_Tensor.elementType(), {Bs, numDetections, 4}, {dc, "ResultBox_T"});
  poplar::Tensor value_index =
      graph.addVariable(INT, {Bs}, {dc, "value_index_T"});
  poplar::Tensor Finish_Tensor = graph.addVariable(INT, {Bs}, {dc, "Finish_T"});
  poplar::Tensor Box_i_Tensor =
      graph.addVariable(Box_Tensor.elementType(), {Bs, 4}, "Box_i_T");

  poplar::Tensor iTensor = graph.addVariable(INT, {Bs}, "iTensor");
  poplar::Tensor NumOne_Tensor = graph.addVariable(INT, {Bs}, "NumOnes_T");
  poplar::Tensor NumNonZeros_Tensor =
      graph.addVariable(INT, {Bs}, "NumNonZeros_T");

  poplar::program::Sequence core_programs;

  init_params(graph, debugMsg, prog, value_index, Keep_Tensor, Bs, N);

  auto iter_init_prog = init_iter_round_program(
      graph, debugMsg, Keep_Tensor, Score_Tensor, NumOne_Tensor,
      NumNonZeros_Tensor, iTensor, Finish_Tensor, Score_2D_idx, Bs, N);

  auto fetch_set_prog = fetch_set_result_program(
      graph, debugMsg, Score_Tensor, Box_Tensor, Keep_Tensor, iTensor,
      Box_i_Tensor, result_Tensor, resultbox_Tensor, value_index, Score_2D_idx,
      Bs, N, numDetections);

  auto nms_core_prog = create_nms_core_program(
      graph, debugMsg, Keep_Tensor, Score_Tensor, Box_Tensor, Finish_Tensor,
      Box_i_Tensor, Score_2D_idx, threshold, Bs, N);

  core_programs.add(iter_init_prog);
  core_programs.add(fetch_set_prog);
  core_programs.add(nms_core_prog);

  prog.add(program::Repeat(numDetections, core_programs));

  return result_Tensor;
}
