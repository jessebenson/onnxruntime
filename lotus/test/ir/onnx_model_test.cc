#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <memory>
#include "core/platform/env.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "gtest/gtest.h"
#include "test/ir/node_helper.h"

using namespace Lotus;
using namespace onnx;
namespace LotusIR {
namespace Test {
// Tests that Resolve() properly clears the state of topological sorted nodes,
// inputs, outputs and valueInfo.
// Assumes the graph passed in has been previously resolved.
static void TestResolve(LotusIR::Graph* p_graph) {
  const std::vector<LotusIR::NodeIndex>* nodes;
  EXPECT_TRUE(p_graph->GetNodesInTopologicalOrder(&nodes).IsOK());
  auto nodes_before = *nodes;
  auto& inputs_before = p_graph->GetInputs();
  auto& outputs_before = p_graph->GetOutputs();
  auto& value_info_before = p_graph->GetValueInfo();

  // Touch the graph to force Resolve() to recompute.
#ifdef _WIN32
  NodeTestHelper::MutableDefinitions(*p_graph->GetNode(0)).input_arg_count;
#else
  NodeTestHelper::MutableDefinitions(*p_graph->GetNode(0));
#endif
  EXPECT_TRUE(p_graph->Resolve().IsOK());

  const std::vector<LotusIR::NodeIndex>* nodes_after;
  EXPECT_TRUE(p_graph->GetNodesInTopologicalOrder(&nodes_after).IsOK());
  auto& inputs_after = p_graph->GetInputs();
  auto& outputs_after = p_graph->GetOutputs();
  auto& value_info_after = p_graph->GetValueInfo();

  // Multiple calls to Resolve() should not alter the sorted nodes,
  // inputs, outputs and valueInfo. The internal state should be
  // cleared.
  EXPECT_EQ(nodes_before, *nodes_after);
  EXPECT_EQ(inputs_before, inputs_after);
  EXPECT_EQ(outputs_before, outputs_after);
  EXPECT_EQ(value_info_before, value_info_after);
}

TEST(ONNXModelsTest, squeeze_net) {
  // NOTE: this requires the current directory to be where LotusIR_UT.exe is located
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("./testdata/squeezenet/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
#ifdef _WIN32
  // wstring version
  std::shared_ptr<Model> model2;
  ASSERT_TRUE(Model::Load(L"./testdata/squeezenet/model.onnx", model2).IsOK());
  TestResolve(model2->MainGraph());
#endif
}

TEST(ONNXModelsTest, non_existing_model) {
  // NOTE: this requires the current directory to be where LotusIR_UT.exe is located
  std::shared_ptr<Model> model;
  Common::Status st = Model::Load("./testdata/non_existing_model_XXXXXX/model.onnx", model);
  ASSERT_FALSE(st.IsOK());
  ASSERT_EQ(st.Code(), Common::NO_SUCHFILE);
#ifdef _WIN32
  // wstring version
  std::shared_ptr<Model> model2;
  ASSERT_FALSE(Model::Load(L"./testdata/non_existing_model_XXXXXX/model.onnx", model2).IsOK());
  ASSERT_EQ(st.Code(), Common::NO_SUCHFILE);
#endif
}

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
TEST(ONNXModelsTest1, bvlc_alexnet_1) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  ASSERT_TRUE(Env::Default().FileOpenRd("../models/test_bvlc_alexnet/model.onnx", &fd).IsOK());
  ASSERT_TRUE(fd > 0);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
  ModelProto model_proto;
  bool result = model_proto.ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  EXPECT_TRUE(result);
  ASSERT_TRUE(Env::Default().FileClose(fd).IsOK());

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_bvlc_alexnet/model.onnx", model).IsOK());

  // Check the graph input/output/value_info should have the same size as specified in the model file.
  EXPECT_EQ(model_proto.graph().value_info_size(), model->MainGraph()->GetValueInfo().size());
  EXPECT_EQ(model_proto.graph().input_size(), model->MainGraph()->GetInputs().size() + model->MainGraph()->GetAllInitializedTensors().size());
  EXPECT_EQ(model_proto.graph().output_size(), model->MainGraph()->GetOutputs().size());
  TestResolve(model->MainGraph());
}

class ONNXModelsTest : public ::testing::TestWithParam<const char*> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
 public:
  std::string GetModelFileName() const {
    std::ostringstream oss;
    oss << "../models/test_" << GetParam() << "/model.onnx";
    return oss.str();
  }
};

TEST_P(ONNXModelsTest, LoadFromFile) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(GetModelFileName(), model).IsOK());
  TestResolve(model->MainGraph());
}

TEST_P(ONNXModelsTest, LoadFromProtobuf) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  auto st = Env::Default().FileOpenRd(GetModelFileName(), &fd);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  ASSERT_TRUE(fd > 0);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
  std::unique_ptr<ModelProto> model_proto = std::make_unique<ModelProto>();
  bool result = model_proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  ASSERT_TRUE(result);
  ASSERT_TRUE(Env::Default().FileClose(fd).IsOK());
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(std::move(model_proto), model).IsOK());
  TestResolve(model->MainGraph());
}

INSTANTIATE_TEST_CASE_P(ONNXModelsTests,
                        ONNXModelsTest,
                        ::testing::Values("bvlc_alexnet", "bvlc_googlenet", "bvlc_reference_caffenet", "bvlc_reference_rcnn_ilsvrc13", "densenet121", "emotion_ferplus", "inception_v1", "inception_v2", "mnist", "resnet50", "shufflenet", "squeezenet", "tiny_yolov2", "vgg19", "zfnet"));

#endif
}  // namespace Test
}  // namespace LotusIR
