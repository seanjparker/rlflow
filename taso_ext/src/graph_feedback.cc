#include "taso/ops.h"
#include "taso/substitution.h"
#include "graph_feedback.h"
#include <iostream>
#include <limits>

using namespace taso;
using namespace std;
using namespace xflowrl;



RLOptimizer::RLOptimizer(Graph* graph) {
  this->graph = graph;
}


void RLOptimizer::set_graph(Graph* graph) {
  this->graph = graph;
}

Graph* RLOptimizer::get_graph() {
  return this->graph;
}

bool RLOptimizer::reset() {
  Graph* graph = this->graph;
  Model* model = graph->model;

  //printf("Resetting Xfers...\n");
  // TODO: Memory leak due to xfers vector!
  xfers = std::vector<GraphXfer*>();

  for (int i = 1; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
      xfers.push_back(GraphXfer::create_conv_relu(graph->model, i, i, pad_mode));
      xfers.push_back(GraphXfer::create_conv_batch(graph->model, i, i, pad_mode));
      xfers.push_back(GraphXfer::create_conv_mul(graph->model, i, i, pad_mode));
      //xfers.push_back(GraphXfer::create_conv_add(graph->model, i, i, pad_mode));
    }
  xfers.push_back(GraphXfer::create_enlarge_merge_convs(graph->model, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_enlarge_merge_convs(graph->model, AC_MODE_RELU));
  xfers.push_back(GraphXfer::create_merge_group_convs(graph->model, 1, 1, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_merge_group_convs(graph->model, 1, 1, AC_MODE_RELU));
  xfers.push_back(GraphXfer::create_merge_group_convs(graph->model, 2, 2, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_merge_group_convs(graph->model, 2, 2, AC_MODE_RELU));

  //xfers.push_back(create_avg_pool_conv(graph->model));
  //xfers.push_back(create_two_pools(graph->model));
  //xfers.push_back(create_merge_seperable_convs(graph->model));
  char* taso_path = getenv("TASO_HOME");
  if (taso_path == NULL) {
    fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
           "Please set TASO_HOME to the home directory of TASO source code.\n");
    assert(false);
  }
  std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";
  GraphXfer::load_graph_xfer_from_pb_file(graph->model, xfers, graph_subst_file);
  //xfers.push_back(create_fuse_conv_batch_xfer(graph->model));
  //xfers.push_back(create_fuse_conv_relu_xfer(graph->model));
  //xfers.push_back(create_merge_conv_xfer(graph->model));
  //xfers.push_back(create_exclusive_concat_xfer(graph->model));
  //xfers.push_back(create_enlarge_conv_xfer(graph->model));
  //xfers.push_back(create_resnet_merge_xfer(graph->model));

  //printf("Resetting stats...\n");

  bestGraph = graph;
  bestCost = graph->total_cost();

  hashmap.clear();
  hashmap.insert(graph->hash());
}

int RLOptimizer::get_num_xfers() {
  return static_cast<int>(xfers.size());
}

// state
std::vector<int> RLOptimizer::get_available_xfers() {
  auto available_xfers = new std::vector<int>();
  for (int i = 0; i <= xfers.size(); i++) {
    available_xfers->push_back(1);
  }
  return *available_xfers;
}

std::vector<int> RLOptimizer::get_available_locations() {
  hashmap.clear();
  hashmap.insert(graph->hash());

  auto available_xfers = new std::vector<int>();

  // clear xfer graphs and maps
  xfer_graphs = std::vector<std::vector<Graph*>>();
  xfer_inputs = std::vector<std::vector<std::vector<Op>>>();
  xfer_outputs = std::vector<std::vector<std::vector<Op>>>();

  int maxNumOps = graph->inEdges.size() * 2;

  // loop through all xfers
  for (int i = 0; i < xfers.size(); i++) {
    std::vector<Graph*> this_xfer_graphs;
    std::vector<std::vector<Op>> this_xfer_inputs;
    std::vector<std::vector<Op>> this_xfer_outputs;

    get_xfer_locations(xfers[i], 0, graph, this_xfer_graphs, this_xfer_inputs, this_xfer_outputs, hashmap, maxNumOps);

    int num_locations = (int)this_xfer_graphs.size();
    //assert((int)this_xfer_inputs.size() == num_locations);
    //assert((int)this_xfer_outputs.size() == num_locations);

    xfer_graphs.push_back(this_xfer_graphs);
    xfer_inputs.push_back(this_xfer_inputs);
    xfer_outputs.push_back(this_xfer_outputs);

    available_xfers->push_back(num_locations);
  }
  return *available_xfers;
}

std::vector<std::vector<Graph*>> RLOptimizer::get_xfer_graphs() {
  return xfer_graphs;
}

std::vector<std::vector<std::vector<Op>>> RLOptimizer::get_xfer_inputs() {
  return xfer_inputs;
}

std::vector<std::vector<std::vector<Op>>> RLOptimizer::get_xfer_outputs() {
  return xfer_outputs;
}

void RLOptimizer::get_xfer_locations(
                    GraphXfer* xfer,
                    int depth, Graph* graph,
                    std::vector<Graph*>& graphs,
                    std::vector<std::vector<Op>>& this_xfer_inputs,
                    std::vector<std::vector<Op>>& this_xfer_outputs,
                    std::set<size_t>& hashmap, int maxNumOps)
{
  /*

  */
  if (depth >= (int)xfer->srcOps.size()) {
    // this is run once all srcOps have been mapped
    // Create dst operators
    bool pass = true;
    std::vector<OpX*>::const_iterator dstIt;
    for (dstIt = xfer->dstOps.begin(); dstIt != xfer->dstOps.end(); dstIt++)
      if (pass) {
        OpX* dstOp = *dstIt;
        pass = (pass & xfer->create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass) {
        return;
    }
    // Check that output tensors with external edges are mapped
    std::map<Op, OpX*, OpCompare>::const_iterator opIt;
    for (opIt = xfer->mappedOps.begin(); opIt != xfer->mappedOps.end(); opIt++) {
      // loop through all mapped ops Op -> OpX
      const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        // loop through all output edges of the mapped ops
        if (xfer->mappedOps.find(it->dstOp) == xfer->mappedOps.end()) {
          // only check this if the dstOp is not in the mapped Ops ("dstOp is external", i.e. not in the Xfer)
          // dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt->second;
          srcTen.idx = it->srcIdx;
          if (xfer->mappedOutputs.find(srcTen) == xfer->mappedOutputs.end()) {
            pass = false;
            return;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    Graph* newGraph = xfer->create_new_graph(graph);
    // Check that the new graph should not have any loop
    if (newGraph->has_loop()) {
      delete newGraph;
      return;
    }
    // TODO: remove me for better performance
    assert(newGraph->check_correctness());
    if ((int)newGraph->inEdges.size() < maxNumOps) {
      if (hashmap.find(newGraph->hash()) == hashmap.end()) {

        hashmap.insert(newGraph->hash());
        graphs.push_back(newGraph);

        std::vector<Op> input_ops;
        for (std::vector<OpX*>::iterator srcOpsIt = xfer->srcOps.begin(); srcOpsIt != xfer->srcOps.end(); srcOpsIt++) {
          OpX* srcOp = *srcOpsIt;

          input_ops.push_back(srcOp->mapOp);
        }
        this_xfer_inputs.push_back(input_ops);


        std::vector<Op> output_ops;
        for (std::vector<OpX*>::iterator dstOpsIt = xfer->dstOps.begin(); dstOpsIt != xfer->dstOps.end(); dstOpsIt++) {
          OpX* dstOp = *dstOpsIt;

          output_ops.push_back(dstOp->mapOp);
        }
        this_xfer_outputs.push_back(output_ops);

      }
    } else {
      delete newGraph;
    }
  } else {
    // This is called as long as depth < srcOps.size(), so all srcOps must be accounted for
    OpX* srcOp = xfer->srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      // Here we iterate over all Ops in the graph (we don't care for the edges right now)
      //printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
      if (xfer->can_match(srcOp, it->first, graph)
      // if the srcOpX matches the Op
      && (xfer->mappedOps.find(it->first) == xfer->mappedOps.end())) {
        // and the Op has not been mapped yet
        Op op = it->first;
        // Check mapOutput
        xfer->match(srcOp, op, graph);
        get_xfer_locations(xfer, depth + 1, graph, graphs, this_xfer_inputs, this_xfer_outputs, hashmap, maxNumOps);
        xfer->unmatch(srcOp, op, graph);
      }
    }
  }
}


float RLOptimizer::get_op_runtime(size_t guid) {
  Op op = this->graph->find_op_or_fail(guid);
  if (op.ptr == NULL)
      return 0.0f;
  return op.ptr->runtime;
}

float RLOptimizer::get_op_runtime_for_graph(Graph* graph, size_t guid) {
  Op op = graph->find_op_or_fail(guid);
  if (op.ptr == NULL)
      return 0.0f;
  return op.ptr->runtime;
}


// action
Graph* RLOptimizer::apply_xfer(int xfer_id, int location_id) {
  if (xfer_id < 0 || xfer_id >= xfer_graphs.size()) {
    printf("Invalid xfer ID: %u\n", xfer_id);
    return NULL;
  }

  std::vector<Graph*> &this_xfer_graphs = xfer_graphs[xfer_id];

  if (location_id < 0 || location_id >= this_xfer_graphs.size()) {
    printf("Invalid location ID: %u\n", location_id);
    return NULL;
  }

  this->graph = this_xfer_graphs[location_id];
  return this->graph;
}

// reward
float RLOptimizer::get_cost() {
  return this->graph->total_cost();
}

float RLOptimizer::get_measured_runtime(Graph* graph)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  std::vector<OpBase*> opBaseList;
  for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = graph->outEdges[op];
    std::set<Edge, EdgeCompare> inList = graph->inEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    assert(inList.size() > 0);
    OpBase* opPtr = NULL;
    // Step 1: prepare inputs
    Tensor inputs[MAX_NUM_INPUTS];
    if ((op.ptr->type == OP_INPUT) || (op.ptr->type == OP_WEIGHT)) {
      assert(inList.size() == 1);
      //Edge e = *inList.begin();
      //assert(e.srcOp.ptr == NULL); // NoOp's input must not be any Op
      Tensor t = op.ptr->inputs[0];
      size_t size = sizeof(DATATYPE);
      for (int j = 0; j < t.numDim; j++)
        size *= t.dim[j];
      if (op.ptr->type == OP_INPUT) {
        assert(t.data_ptr == NULL);
        t.data_ptr = (DATATYPE*) graph->model->allocate_memory(size);
      } else {
        assert(t.data_ptr != NULL);
      }
      inputs[0] = t;
    } else {
      for (it2 = inList.begin(); it2 != inList.end(); it2++) {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == it2->srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[it2->dstIdx].data_ptr == NULL); // No duplicated dstIdxes
        inputs[it2->dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
#ifdef DEADCODE
    // Step 1: prepare inputs
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      Edge e = *it2;
      if (e.srcOp.guid == GUID_INPUT) {
        Tensor t = op.ptr->inputs[e.dstIdx];
        t.ptr = (DATATYPE*) model->allocate_memory(sizeof(DATATYPE) * t.size());
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = t;
      } else if (e.srcOp.guid = GUID_WEIGHT) {
        Tensor t = op.ptr->inputs[e.dstIdx];
        t.ptr = (DATATYPE*) model->allocate_memory(sizeof(DATATYPE) * t.size());
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = t;
      } else {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == e.srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
#endif
    // Step 2: create Ops
    switch (op.ptr->type) {
      case OP_CONV2D:
      {
        //Conv2D* conv = (Conv2D*) op.ptr;
        Conv2D* conv = static_cast<Conv2D*>(op.ptr);
        assert(inList.size() == 2);
        printf("Padding: %d\n", conv->padding);
        opPtr = new Conv2D(graph->model, inputs[0], inputs[1],
                           conv->strideH, conv->strideW,
                           conv->padding, conv->activation);
#ifdef USE_CUDNN
        ((Conv2D*)opPtr)->fwdAlgo = conv->fwdAlgo;
#endif
        break;
      }
      case OP_MATMUL:
      {
        Matmul* matmul = (Matmul*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Matmul(graph->model, inputs[0], inputs[1], matmul->activation);
        break;
      }
      case OP_RESHAPE:
      {
        Reshape* reshape = (Reshape*) op.ptr;
        assert(inList.size() == 1);
        std::vector<int> shape;
        for (int i = 0; i < reshape->outputs[0].numDim; i++)
          shape.push_back(reshape->outputs[0].dim[i]);
        opPtr = new Reshape(graph->model, inputs[0], shape);
        break;
      }
      case OP_TRANSPOSE:
      {
        Transpose* transpose = (Transpose*) op.ptr;
        assert(inList.size() == 1);
        int ndim = inputs[0].numDim, permIdx = transpose->permIdx;
        std::vector<int> permVec;
        int permArray[MAX_DIM];
        for (int i = ndim - 1; i >= 0; i--) {
          permArray[i] = permIdx % ndim;
          permIdx = permIdx / ndim;
        }
        assert(permIdx == 0);
        for (int i = 0; i < ndim; i++)
          for (int j = i + 1; j < ndim; j++)
            assert(permArray[i] != permArray[j]);
        for (int i = 0; i < ndim; i++)
          permVec.push_back(permArray[i]);
        opPtr = new Transpose(graph->model, inputs[0], permVec, transpose->shuffle);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_MUL:
      {
        //Element* element = (Element*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Element(graph->model, op.ptr->type, inputs[0], inputs[1]);
        break;
      }
      case OP_ENLARGE:
      {
        //Enlarge* enlarge = (Enlarge*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Enlarge(graph->model, inputs[0], inputs[1]);
        break;
      }
      case OP_MERGE_GCONV:
      {
        MergeGConv* merge = (MergeGConv*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new MergeGConv(graph->model, inputs[0], merge->count);
        break;
      }
      case OP_POOL2D_MAX:
      case OP_POOL2D_AVG:
      {
        Pool2D* pool = (Pool2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Pool2D(graph->model, inputs[0], inputs[1], pool->type,
                           pool->kernelH, pool->kernelW,
                           pool->strideH, pool->strideW,
                           pool->padding, pool->activation);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      {
        Activation* act = (Activation*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Activation(graph->model, inputs[0], act->type, act->inPlace);
        break;
      }
      case OP_BATCHNORM:
      {
        BatchNorm* batchnorm = (BatchNorm*) op.ptr;
        assert(inList.size() == 5);
        opPtr = new BatchNorm(graph->model, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], batchnorm->epsilon);
        break;
      }
      case OP_SPLIT:
      {
        Split* split = (Split*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Split(graph->model, inputs[0], split->axis, split->sizes);
        break;
      }
      case OP_INPUT:
      case OP_WEIGHT:
      case OP_DROPOUT:
      {
        assert(inList.size() == 1);
        opPtr = new NoOp(graph->model, inputs[0], op.ptr->type);
        break;
      }
      case OP_CONCAT:
      {
        Concat* concat = (Concat*) op.ptr;
        opPtr = new Concat(graph->model, concat->axis, inList.size(), inputs, concat->needCopy);
        break;
      }
      default:
        printf("op.type = %d\n", op.ptr->type);
        assert(false);
    }
    // Step 3: map new Op
    opPtr->map();
    opBaseList.push_back(opPtr);
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      //printf("myOp(%zu) dstOp(%zu) dstType(%d) dstTodos(%d)\n",
      //    it2->srcOp.guid, it2->dstOp.guid,
      //    it2->dstOp.ptr->type, todos[it2->dstOp]);
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
#ifdef VERBOSE_PRINTS
  for (int i =0; i < opList.size(); i++) {
    printf("opList[%d]: guid(%zu) type(%d)\n", i, opList[i].guid,
           opList[i].ptr->type);
  }
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    printf("op: guid(%zu) type(%d)\n", it->first.guid, it->first.ptr->type);
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    int cnt = 0;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      printf("    inEdge[%d]: srcOp(%zu) srcIdx(%d) dstOp(%zu) dstIdx(%d)\n", cnt++, it2->srcOp.guid, it2->srcIdx, it2->dstOp.guid, it2->dstIdx);
    }
  }
#endif

  assert(opList.size() == graph->inEdges.size());
  assert(opList.size() == opBaseList.size());

  float result = graph->model->measure_oplist_runtime(opBaseList);
  // Now free GPU memory from the opList
  for (int i = 0; i < opBaseList.size(); i++) {
    OpBase* opBase = opBaseList[i];
    opBase->unmap();
    delete opBaseList[i];
    // free(opBase);
    opBase = nullptr;
  }

  return result;
}