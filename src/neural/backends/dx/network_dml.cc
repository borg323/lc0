/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_conf.h"
#ifdef USE_DML
#include "neural/backends/dx/network_dx.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "onnxruntime_cxx_api.h"
#include "utils/bf16_utils.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
namespace lczero {
namespace dx_dml_backend {
using dx_backend::DxContext;
static constexpr int kNumOutputPolicy = 1858;
class DmlDxNetwork;
struct DmlInputsOutputs {
  explicit DmlInputsOutputs(DmlDxNetwork* network);
  ~DmlInputsOutputs() {
    free(input_tensor_data_);
    for (void* ptr : output_tensors_data_) {
      free(ptr);
    }
  }
  void* input_tensor_data_;
  std::vector<void*> output_tensors_data_;
  std::vector<size_t> output_tensors_step_;
  std::vector<float> wdl_output_data_;
  Ort::MemoryInfo memory_info_{nullptr};
};
template <typename DataType>
class DmlDxComputation final : public NetworkComputation {
 public:
  explicit DmlDxComputation(DmlDxNetwork* network);
  ~DmlDxComputation() override;
  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override;
  void ComputeBlocking() override;
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  Ort::IoBinding PrepareInputs(int start, int batch_size, int step);
  DmlDxNetwork* network_;
  size_t input_size_ = 0;
  std::vector<InputPlanes> raw_input_;
  std::unique_ptr<DmlInputsOutputs> inputs_outputs_;
};
class DmlDxNetwork final : public Network {
 public:
  DmlDxNetwork(const WeightsFile& file, const OptionsDict& options,
               bool cpu_wdl);
  std::unique_ptr<NetworkComputation> NewComputation() override {
    if (fp16_) {
      return std::make_unique<DmlDxComputation<Ort::Float16_t>>(this);
    } else if (bf16_) {
      return std::make_unique<DmlDxComputation<Ort::BFloat16_t>>(this);
    } else {
      return std::make_unique<DmlDxComputation<float>>(this);
    }
  }
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  int GetMiniBatchSize() const override {
    return batch_size_ == -1 ? Network::GetMiniBatchSize()
                             : batch_size_ * steps_;
  }
  int GetPreferredBatchStep() const override {
    return batch_size_ == -1 ? min_batch_size_ : batch_size_;
  }
  bool IsCpu() const override { return false; }
  Ort::SessionOptions GetOptions(int threads, int batch_size, int optimize);
  std::unique_ptr<DmlInputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<DmlInputsOutputs>(this);
    }
    std::unique_ptr<DmlInputsOutputs> resource =
        std::move(free_inputs_outputs_.front());
    free_inputs_outputs_.pop_front();
    return resource;
  }
  void ReleaseInputsOutputs(std::unique_ptr<DmlInputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }
  Ort::Env onnx_env_;
  int steps_;
  std::vector<Ort::Session> session_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  int policy_head_ = -1;
  int wdl_head_ = -1;
  int value_head_ = -1;
  int mlh_head_ = -1;
  NetworkCapabilities capabilities_;
  bool fp16_;
  bool bf16_;
  bool cpu_wdl_;
  int batch_size_;
  int min_batch_size_;
  int gpu_;
  static constexpr int max_batch_size_ = 1024;
  std::mutex lock_;
  DxContext dx_context_;

 private:
  std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<DmlInputsOutputs>> free_inputs_outputs_;
};
DmlInputsOutputs::DmlInputsOutputs(DmlDxNetwork* network) {
  const int max_batch_size = network->max_batch_size_;
  const int value_head = network->value_head_;
  const int wdl_head = network->wdl_head_;
  const int policy_head = network->policy_head_;
  const int mlh_head = network->mlh_head_;
  const int data_size = (network->fp16_ || network->bf16_) ? 2 : 4;
  int outputs_size =
      std::max({value_head, wdl_head, policy_head, mlh_head}) + 1;
  output_tensors_data_.resize(outputs_size);
  output_tensors_step_.resize(outputs_size);
  if (wdl_head != -1) {
    wdl_output_data_.resize(3 * max_batch_size);
  }
  output_tensors_step_[policy_head] = kNumOutputPolicy;
  if (wdl_head != -1) {
    output_tensors_step_[wdl_head] = 3;
  }
  if (value_head != -1) {
    output_tensors_step_[value_head] = 1;
  }
  if (mlh_head != -1) {
    output_tensors_step_[mlh_head] = 1;
  }
  // NOTE: This backend owns a DX12 context and is structured for D3D12-backed
  // ORT IoBinding interop. The ORT version/API used in this repository path
  // does not expose a stable D3D12 tensor binding surface in-tree, so we use
  // the safe buildable CPU tensor fallback here while preserving DML EP usage.
  input_tensor_data_ =
      malloc(max_batch_size * kInputPlanes * 8 * 8 * data_size);
  for (int i = 0; i < outputs_size; i++) {
    output_tensors_data_[i] =
        malloc(max_batch_size * output_tensors_step_[i] * data_size);
  }
  memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
}
void AsDataType(float x, float* y) { *y = x; }
void AsDataType(float x, Ort::Float16_t* y) {
  uint16_t tmp = FP32toFP16(x);
  std::memcpy(reinterpret_cast<uint16_t*>(y), &tmp, sizeof(uint16_t));
}
void AsDataType(float x, Ort::BFloat16_t* y) {
  uint16_t tmp = FP32toBF16(x);
  std::memcpy(reinterpret_cast<uint16_t*>(y), &tmp, sizeof(uint16_t));
}
template <typename DataType>
DmlDxComputation<DataType>::DmlDxComputation(DmlDxNetwork* network)
    : network_(network) {
  inputs_outputs_ = network_->GetInputsOutputs();
}
template <typename DataType>
DmlDxComputation<DataType>::~DmlDxComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}
template <typename DataType>
void DmlDxComputation<DataType>::AddInput(InputPlanes&& input) {
  if (input_size_ >= network_->max_batch_size_) {
    throw Exception("NN input exceeds max batch size of " +
                    std::to_string(network_->max_batch_size_) + ".");
  }
  raw_input_.emplace_back(std::move(input));
  input_size_++;
}
template <typename DataType>
int DmlDxComputation<DataType>::GetBatchSize() const {
  return input_size_;
}
float AsFloat(float x) { return x; }
float AsFloat(Ort::Float16_t x) {
  uint16_t tmp;
  std::memcpy(&tmp, reinterpret_cast<uint16_t*>(&x), sizeof(uint16_t));
  return FP16toFP32(tmp);
}
float AsFloat(Ort::BFloat16_t x) {
  uint16_t tmp;
  std::memcpy(&tmp, reinterpret_cast<uint16_t*>(&x), sizeof(uint16_t));
  return BF16toFP32(tmp);
}
template <typename DataType>
float DmlDxComputation<DataType>::GetQVal(int sample) const {
  if (network_->wdl_head_ != -1) {
    return inputs_outputs_->wdl_output_data_[sample * 3 + 0] -
           inputs_outputs_->wdl_output_data_[sample * 3 + 2];
  }
  DataType* data = static_cast<DataType*>(
      inputs_outputs_->output_tensors_data_[network_->value_head_]);
  return AsFloat(data[sample]);
}
template <typename DataType>
float DmlDxComputation<DataType>::GetDVal(int sample) const {
  if (network_->wdl_head_ == -1) return 0.0f;
  return inputs_outputs_->wdl_output_data_[sample * 3 + 1];
}
template <typename DataType>
float DmlDxComputation<DataType>::GetPVal(int sample, int move_id) const {
  DataType* data = static_cast<DataType*>(
      inputs_outputs_->output_tensors_data_[network_->policy_head_]);
  return AsFloat(data[sample * kNumOutputPolicy + move_id]);
}
template <typename DataType>
float DmlDxComputation<DataType>::GetMVal(int sample) const {
  if (network_->mlh_head_ == -1) return 0.0f;
  DataType* data = static_cast<DataType*>(
      inputs_outputs_->output_tensors_data_[network_->mlh_head_]);
  return AsFloat(data[sample]);
}
template <typename DataType>
Ort::IoBinding DmlDxComputation<DataType>::PrepareInputs(int start,
                                                         int batch_size,
                                                         int step) {
  DataType* iter = static_cast<DataType*>(inputs_outputs_->input_tensor_data_);
  iter += start * kInputPlanes * 8 * 8;
  std::memset(static_cast<void*>(iter), 0,
              batch_size * kInputPlanes * 8 * 8 * sizeof(DataType));
  const int end = std::min(start + batch_size, static_cast<int>(input_size_));
  for (int i = start; i < end; i++) {
    for (const auto& plane : raw_input_[i]) {
      DataType value;
      AsDataType(plane.value, &value);
      for (auto bit : IterateBits(plane.mask)) {
        *(iter + bit) = value;
      }
      iter += 64;
    }
  }
  Ort::IoBinding binding{network_->session_[step - 1]};
  for (size_t i = 0; i < inputs_outputs_->output_tensors_step_.size(); i++) {
    int size = inputs_outputs_->output_tensors_step_[i];
    int64_t dims[] = {batch_size, size};
    binding.BindOutput(
        network_->outputs_[i].c_str(),
        Ort::Value::CreateTensor<DataType>(
            inputs_outputs_->memory_info_,
            static_cast<DataType*>(inputs_outputs_->output_tensors_data_[i]) +
                start * size,
            size * batch_size, dims, 2));
  }
  int64_t dims[] = {batch_size, kInputPlanes, 8, 8};
  binding.BindInput(
      network_->inputs_[0].c_str(),
      Ort::Value::CreateTensor<DataType>(
          inputs_outputs_->memory_info_,
          static_cast<DataType*>(inputs_outputs_->input_tensor_data_) +
              start * kInputPlanes * 8 * 8,
          batch_size * kInputPlanes * 8 * 8, dims, 4));
  return binding;
}
template <typename DataType>
void DmlDxComputation<DataType>::ComputeBlocking() {
  if (GetBatchSize() == 0) return;
  int batch_size = network_->batch_size_;
  if (batch_size < 0) {
    batch_size =
        std::max(static_cast<int>(input_size_), network_->min_batch_size_);
  }
  std::lock_guard<std::mutex> lock(network_->lock_);
  for (size_t i = 0; i < input_size_;) {
    int step = (input_size_ - i + batch_size - 1) / batch_size;
    if (step > network_->steps_) step = network_->steps_;
    int batch = batch_size * step;
    auto binding = PrepareInputs(i, batch, step);
    binding.SynchronizeInputs();
    network_->session_[step - 1].Run(Ort::RunOptions{}, binding);
    binding.SynchronizeOutputs();
    i += batch;
  }
  if (network_->wdl_head_ != -1) {
    const DataType* data = static_cast<DataType*>(
        inputs_outputs_->output_tensors_data_[network_->wdl_head_]);
    for (size_t i = 0; i < input_size_; i++) {
      float w = AsFloat(data[i * 3 + 0]);
      float d = AsFloat(data[i * 3 + 1]);
      float l = AsFloat(data[i * 3 + 2]);
      if (network_->cpu_wdl_) {
        float m = std::max({w, d, l});
        w = std::exp(w - m);
        d = std::exp(d - m);
        l = std::exp(l - m);
        float sum = w + d + l;
        w /= sum;
        d /= sum;
        l /= sum;
      }
      inputs_outputs_->wdl_output_data_[3 * i + 0] = w;
      inputs_outputs_->wdl_output_data_[3 * i + 1] = d;
      inputs_outputs_->wdl_output_data_[3 * i + 2] = l;
    }
  }
}
Ort::SessionOptions DmlDxNetwork::GetOptions(int threads, int batch_size,
                                             int optimize) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  GraphOptimizationLevel level = GraphOptimizationLevel::ORT_DISABLE_ALL;
  switch (optimize) {
    case 0:
      level = GraphOptimizationLevel::ORT_DISABLE_ALL;
      break;
    case 1:
      level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
      break;
    case 2:
      level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
      break;
    default:
      level = GraphOptimizationLevel::ORT_ENABLE_ALL;
      break;
  }
  options.SetGraphOptimizationLevel(level);
  if (batch_size > 0) {
    Ort::ThrowOnError(
        OrtGetApiBase()
            ->GetApi(ORT_API_VERSION)
            ->AddFreeDimensionOverrideByName(options, "batch", batch_size));
  }
  std::unordered_map<std::string, std::string> dml_options;
  dml_options["device_id"] = std::to_string(gpu_);
  options.AppendExecutionProvider("DML", dml_options);
  return options;
}
DmlDxNetwork::DmlDxNetwork(const WeightsFile& file, const OptionsDict& opts,
                           bool cpu_wdl)
    : onnx_env_(ORT_LOGGING_LEVEL_WARNING, "lc0"),
      capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()},
      fp16_(file.onnx_model().data_type() == pblczero::OnnxModel::FLOAT16),
      bf16_(file.onnx_model().data_type() == pblczero::OnnxModel::BFLOAT16),
      cpu_wdl_(cpu_wdl),
      dx_context_(opts) {
  onnx_env_.DisableTelemetryEvents();
  gpu_ = opts.GetOrDefault<int>("gpu", 0);
  int threads = opts.GetOrDefault<int>("threads", 0);
  int optimize = opts.GetOrDefault<int>("optimize", 3);
  int default_batch = 16;
  int default_steps = 4;
  int default_min_batch = 1;
  batch_size_ = opts.GetOrDefault<int>("batch", default_batch);
  steps_ = opts.GetOrDefault<int>("steps", default_steps);
  min_batch_size_ = opts.GetOrDefault<int>("min_batch", default_min_batch);
  if (batch_size_ <= 0) {
    batch_size_ = -1;
    steps_ = 1;
  }
  if (batch_size_ * steps_ > max_batch_size_) {
    batch_size_ = max_batch_size_ / steps_;
  }
  const auto& md = file.onnx_model();
  if (!md.has_input_planes()) {
    throw Exception("NN doesn't have input planes defined.");
  }
  inputs_.emplace_back(md.input_planes());
  if (!md.has_output_policy()) {
    throw Exception("NN doesn't have policy head defined.");
  }
  policy_head_ = outputs_.size();
  outputs_.emplace_back(md.output_policy());
  if (md.has_output_wdl()) {
    wdl_head_ = outputs_.size();
    outputs_.emplace_back(md.output_wdl());
  } else if (md.has_output_value()) {
    value_head_ = outputs_.size();
    outputs_.emplace_back(md.output_value());
  } else {
    throw Exception("NN doesn't have value head.");
  }
  if (md.has_output_mlh()) {
    mlh_head_ = outputs_.size();
    outputs_.emplace_back(md.output_mlh());
  }
  for (int step = 1; step <= steps_; step++) {
    session_.emplace_back(onnx_env_, md.model().data(), md.model().size(),
                          GetOptions(threads, batch_size_ * step, optimize));
  }
}
std::unique_ptr<Network> MakeDmlDxNetwork(const std::optional<WeightsFile>& w,
                                          const OptionsDict& opts) {
  if (!w) throw Exception("The DML DX12 backend requires a network file.");
  if (w->has_onnx_model()) {
    return std::make_unique<DmlDxNetwork>(*w, opts, false);
  }
  WeightsToOnnxConverterOptions converter_options;
  converter_options.ir = opts.GetOrDefault<int>("ir", -1);
  converter_options.alt_mish = opts.GetOrDefault<bool>("alt_mish", false);
  converter_options.alt_layernorm = opts.GetOrDefault<bool>(
      "alt_layernorm", w->format().network_format().ffn_activation() ==
                               pblczero::NetworkFormat::ACTIVATION_RELU_2
                           ? true
                           : false);
  converter_options.no_shape = opts.GetOrDefault<bool>("no_shape", false);
  converter_options.policy_head =
      opts.GetOrDefault<std::string>("policy_head", "vanilla");
  converter_options.value_head =
      opts.GetOrDefault<std::string>("value_head", "winner");
  converter_options.no_wdl_softmax = true;
  converter_options.real_mish = false;
  std::string datatype;
  if (opts.Exists<std::string>("datatype")) {
    datatype = opts.Get<std::string>("datatype");
  } else {
    bool fp16 = opts.GetOrDefault<bool>("fp16", true);
    datatype = fp16 ? "f16" : "f32";
  }
  converter_options.data_type =
      WeightsToOnnxConverterOptions::StringToDataType(datatype);
  converter_options.opset = opts.GetOrDefault<int>(
      "opset", converter_options.data_type ==
                       WeightsToOnnxConverterOptions::DataType::kBFloat16
                   ? 22
                   : 17);
  auto converted = ConvertWeightsToOnnx(*w, converter_options);
  return std::make_unique<DmlDxNetwork>(converted, opts, true);
}
REGISTER_NETWORK("dml-dx12", MakeDmlDxNetwork, 121)
}  // namespace dx_dml_backend
}  // namespace lczero
#endif  // USE_DML
