// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/framework/test_utils.h"
#include "core/session/inference_session.h"

#include <chrono>
#include <sstream>

namespace onnxruntime {
namespace test {

TEST(TensorOpTest, Pad_Performance) {
  // Create graph with single Pad node
  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries;
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  auto model = std::make_unique<onnxruntime::Model>("test", false, ModelMetaData(), custom_schema_registries, domain_to_version);
  onnxruntime::Graph& graph = model->MainGraph();

  ONNX_NAMESPACE::TypeProto in_type;
  in_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto in_shape = in_type.mutable_tensor_type()->mutable_shape();
  for (int64_t dim : { 1, 224, 224, 3 })
    in_shape->add_dim()->set_dim_value(dim);

  ONNX_NAMESPACE::TypeProto out_type;
  out_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto out_shape = out_type.mutable_tensor_type()->mutable_shape();
  for (int64_t dim : { 1, 230, 230, 3 })
    out_shape->add_dim()->set_dim_value(dim);

  NodeArg in_arg = { "in", &in_type };
  NodeArg out_arg = { "out", &out_type };
  auto& node = graph.AddNode("node1", "Pad", "Pad", { &in_arg} , { &out_arg }, nullptr, onnxruntime::kOnnxDomain);
  node.AddAttribute("pads", std::vector<int64_t>{0, 3, 3, 0, 0, 3, 3, 0});
  node.AddAttribute("value", 0.0f);

  EXPECT_TRUE(graph.Resolve().IsOK());
  std::stringstream s1;
  model->ToProto().SerializeToOstream(&s1);

  // load the model
  SessionOptions so;
  so.session_logid = "Pad";
  so.enable_profiling = true;
  InferenceSession session_object{so};
  EXPECT_TRUE(session_object.RegisterExecutionProvider(DefaultCpuExecutionProvider()).IsOK());
  EXPECT_TRUE(session_object.Load(s1).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);

  // prepare inputs
  MLValue ml_value;
  CreateMLValue<float>(allocator, {1,224,224,3}, std::vector<float>(1*224*224*3, 0.0f), &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("out");
  std::vector<MLValue> fetches;

  RunOptions run_options;

  // measure Pad performance
  uint64_t elapsed = 0;
  for (size_t i = 0; i < 1000; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    common::Status status = session_object.Run(run_options, feeds, output_names, &fetches);
    auto end = std::chrono::high_resolution_clock::now();
    elapsed += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    EXPECT_TRUE(status.IsOK());
    auto& rtensor = fetches.front().Get<Tensor>();
    TensorShape expected_shape({1, 230, 230, 3});
    ASSERT_EQ(expected_shape, rtensor.Shape());
  }

  std::cout << "Pad elapsed: " << (elapsed / 1000) << "us" << std::endl;
  std::cout << session_object.EndProfiling() << std::endl;
}

TEST(TensorOpTest, Pad_Spec_Example) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{0, 2, 0, 0});
  test.AddAttribute("value", 0.0f);
  test.AddInput<float>("data", {3, 2}, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
  test.AddOutput<float>("output", {3, 4}, {0.0f, 0.0f, 1.0f, 1.2f, 0.0f, 0.0f, 2.3f, 3.4f, 0.0f, 0.0f, 4.5f, 5.7f});
  test.Run();
}

TEST(TensorOpTest, Pad_Constant_1D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("output", {5}, {1234.0f, 1.0f, 2.0f, 1234.0f, 1234.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_Constant_1D_Zero) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("output", {2}, {1.0f, 2.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_Constant_2D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2, 1, 2});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2, 2},
                       {11.0f, 21.0f,
                        12.0f, 22.0f});
  test.AddOutput<float>("output", {4, 6},
                        {1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 11.0f, 21.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 12.0f, 22.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_Constant_2D_negative) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2, 1, -1});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f});
  test.AddOutput<float>("output", {4, 4},
                        {1234.0f, 1234.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 11.0f, 21.0f,
                         1234.0f, 1234.0f, 12.0f, 22.0f,
                         1234.0f, 1234.0f, 1234.0f, 1234.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_3D_complex) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 0, 0, -1, 0, 0});
  test.AddAttribute("value", 0.0f);
  test.AddInput<float>("data", {2, 2, 2},
                       {111.0f, 112.0f,
                        121.0f, 122.0f,

                        211.0f, 212.0f,
                        221.0f, 222.0f});
  test.AddOutput<float>("output", {2, 2, 2},
                        {0.0f, 0.0f,
                         0.0f, 0.0f,

                         111.0f, 112.0f,
                         121.0f, 122.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_Edge_2D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2});
  test.AddAttribute("mode", "edge");
  test.AddInput<float>("data", {2, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f});
  test.AddOutput<float>("output", {6, 7},
                        {11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_Edge_3D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2, 2, 1, 2, 2});
  test.AddAttribute("mode", "edge");
  test.AddInput<float>("data", {1, 2, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f});
  test.AddOutput<float>("output", {3, 6, 7},
                        {11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,

                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,

                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f});

  test.Run();
}

TEST(TensorOpTest, Pad_Reflect_2D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2});
  test.AddAttribute("mode", "reflect");
  test.AddInput<float>("data", {3, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f,
                        13.0f, 23.0f, 33.0f});
  test.AddOutput<float>("output", {7, 7},
                        {33.0f, 23.0f, 13.0f, 23.0f, 33.0f, 23.0f, 13.0f,
                         32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                         31.0f, 21.0f, 11.0f, 21.0f, 31.0f, 21.0f, 11.0f,
                         32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                         33.0f, 23.0f, 13.0f, 23.0f, 33.0f, 23.0f, 13.0f,
                         32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                         31.0f, 21.0f, 11.0f, 21.0f, 31.0f, 21.0f, 11.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
