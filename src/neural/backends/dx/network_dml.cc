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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "onnx_conf.h"
#ifdef USE_DML

#include "dml_provider_factory.h"
#include "neural/backends/dx/network_dx.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "onnxruntime_cxx_api.h"
#include "utils/bf16_utils.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"

namespace lczero {
namespace dx_dml_backend {
using dx_backend::DXAlloc;
using dx_backend::DxContext;
using dx_backend::DxError;

namespace {

static constexpr int kNumOutputPolicy = 1858;

void ReleaseAlloc(DXAlloc& alloc) {
  if (alloc.resource) {
    alloc.resource->Release();
    alloc.resource = nullptr;
  }
}

const OrtDmlApi* GetDmlApi() {
  const void* provider_api = nullptr;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)
                        ->GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                  &provider_api));
  return static_cast<const OrtDmlApi*>(provider_api);
}

using DmlCreateDeviceFn = HRESULT(WINAPI*)(IUnknown* d3d12_device,
                                           DML_CREATE_DEVICE_FLAGS flags,
                                           REFIID riid, void** dml_device);

DmlCreateDeviceFn GetDmlCreateDeviceFn() {
  static DmlCreateDeviceFn fn = []() -> DmlCreateDeviceFn {
    HMODULE module = GetModuleHandleW(L"DirectML.dll");
    if (!module) module = LoadLibraryW(L"DirectML.dll");
    if (!module) {
      throw Exception("Failed to load DirectML.dll for dml-dx12 backend.");
    }
    auto proc = reinterpret_cast<DmlCreateDeviceFn>(
        GetProcAddress(module, "DMLCreateDevice"));
    if (!proc) {
      throw Exception("Failed to resolve DMLCreateDevice from DirectML.dll.");
    }
    return proc;
  }();
  return fn;
}

IDMLDevice* CreateDmlDevice(ID3D12Device* device) {
  IDMLDevice* dml_device = nullptr;
  ReportDxErrors(GetDmlCreateDeviceFn()(
      device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml_device)));
  return dml_device;
}

template <typename DataType>
void CopyOutputChunk(void* dst, const void* src, size_t offset_elements,
                     size_t count_elements) {
  std::memcpy(static_cast<char*>(dst) + offset_elements * sizeof(DataType), src,
              count_elements * sizeof(DataType));
}

void IgnoreOrtStatus(OrtStatus* status) {
  if (status) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }
}

}  // namespace

class DmlDxNetwork;

struct DmlInputsOutputs {
  explicit DmlInputsOutputs(DmlDxNetwork* network);
  ~DmlInputsOutputs();

  const OrtDmlApi* dml_api_ = nullptr;

  DXAlloc input_masks_mem_gpu_{};
  DXAlloc input_val_mem_gpu_{};
  DXAlloc input_tensor_gpu_{};
  uint64_t* input_masks_mem_ = nullptr;
  float* input_val_mem_ = nullptr;
  void* input_ort_allocation_ = nullptr;

  std::vector<DXAlloc> output_tensors_gpu_;
  std::vector<void*> output_tensors_gpu_mapped_;
  std::vector<void*> output_tensors_data_;
  std::vector<void*> output_tensors_ort_allocation_;
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
  void CopyOutputs(int start, int batch_size);

  DmlDxNetwork* network_;
  size_t input_size_ = 0;
  std::unique_ptr<DmlInputsOutputs> inputs_outputs_;
};

class DmlDxNetwork final : public Network {
 public:
  DmlDxNetwork(const WeightsFile& file, const OptionsDict& options,
               bool cpu_wdl);
  ~DmlDxNetwork() override;

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
  const OrtDmlApi* dml_api_ = nullptr;
  IDMLDevice* dml_device_ = nullptr;
  bool command_list_needs_reset_ = false;

 private:
  std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<DmlInputsOutputs>> free_inputs_outputs_;
};

DmlInputsOutputs::DmlInputsOutputs(DmlDxNetwork* network)
    : dml_api_(network->dml_api_) {
  const int max_batch_size = network->max_batch_size_;
  const int value_head = network->value_head_;
  const int wdl_head = network->wdl_head_;
  const int policy_head = network->policy_head_;
  const int mlh_head = network->mlh_head_;
  const int data_size = (network->fp16_ || network->bf16_) ? 2 : 4;
  const int outputs_size =
      std::max({value_head, wdl_head, policy_head, mlh_head}) + 1;

  output_tensors_data_.resize(outputs_size);
  output_tensors_step_.resize(outputs_size);
  output_tensors_ort_allocation_.resize(outputs_size);
  output_tensors_gpu_.resize(outputs_size);
  output_tensors_gpu_mapped_.resize(outputs_size);

  if (wdl_head != -1) {
    wdl_output_data_.resize(3 * max_batch_size);
  }

  output_tensors_step_[policy_head] = kNumOutputPolicy;
  if (wdl_head != -1) output_tensors_step_[wdl_head] = 3;
  if (value_head != -1) output_tensors_step_[value_head] = 1;
  if (mlh_head != -1) output_tensors_step_[mlh_head] = 1;

  network->dx_context_.CreateAlloc(max_batch_size * kInputPlanes *
                                       sizeof(uint64_t),
                                   D3D12_HEAP_TYPE_UPLOAD, input_masks_mem_gpu_,
                                   false);
  network->dx_context_.CreateAlloc(max_batch_size * kInputPlanes *
                                       sizeof(float),
                                   D3D12_HEAP_TYPE_UPLOAD, input_val_mem_gpu_,
                                   false);
  network->dx_context_.CreateAlloc(
      max_batch_size * kInputPlanes * 8 * 8 * data_size,
      D3D12_HEAP_TYPE_DEFAULT, input_tensor_gpu_,
      network->fp16_ || network->bf16_);

  ReportDxErrors(input_masks_mem_gpu_.resource->Map(
      0, nullptr, reinterpret_cast<void**>(&input_masks_mem_)));
  ReportDxErrors(input_val_mem_gpu_.resource->Map(
      0, nullptr, reinterpret_cast<void**>(&input_val_mem_)));
  Ort::ThrowOnError(dml_api_->CreateGPUAllocationFromD3DResource(
      input_tensor_gpu_.resource, &input_ort_allocation_));

  for (int i = 0; i < outputs_size; i++) {
    if (output_tensors_step_[i] == 0) continue;
    output_tensors_data_[i] =
        malloc(max_batch_size * output_tensors_step_[i] * data_size);
    network->dx_context_.CreateAlloc(
        max_batch_size * output_tensors_step_[i] * data_size,
        D3D12_HEAP_TYPE_CUSTOM, output_tensors_gpu_[i],
        network->fp16_ || network->bf16_);
    ReportDxErrors(output_tensors_gpu_[i].resource->Map(
        0, nullptr, &output_tensors_gpu_mapped_[i]));
    Ort::ThrowOnError(dml_api_->CreateGPUAllocationFromD3DResource(
        output_tensors_gpu_[i].resource, &output_tensors_ort_allocation_[i]));
  }

  memory_info_ = Ort::MemoryInfo{"DML", OrtDeviceAllocator, network->gpu_,
                                 OrtMemTypeDefault};
}

DmlInputsOutputs::~DmlInputsOutputs() {
  if (input_masks_mem_gpu_.resource && input_masks_mem_) {
    input_masks_mem_gpu_.resource->Unmap(0, nullptr);
  }
  if (input_val_mem_gpu_.resource && input_val_mem_) {
    input_val_mem_gpu_.resource->Unmap(0, nullptr);
  }
  for (size_t i = 0; i < output_tensors_gpu_.size(); i++) {
    if (output_tensors_gpu_[i].resource && output_tensors_gpu_mapped_[i]) {
      output_tensors_gpu_[i].resource->Unmap(0, nullptr);
    }
  }
  if (dml_api_) {
    if (input_ort_allocation_) {
      IgnoreOrtStatus(dml_api_->FreeGPUAllocation(input_ort_allocation_));
    }
    for (void* allocation : output_tensors_ort_allocation_) {
      if (allocation) {
        IgnoreOrtStatus(dml_api_->FreeGPUAllocation(allocation));
      }
    }
  }
  ReleaseAlloc(input_tensor_gpu_);
  ReleaseAlloc(input_masks_mem_gpu_);
  ReleaseAlloc(input_val_mem_gpu_);
  for (auto& alloc : output_tensors_gpu_) ReleaseAlloc(alloc);
  for (void* ptr : output_tensors_data_) free(ptr);
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
  const size_t offset = input_size_ * kInputPlanes;
  size_t plane_index = 0;
  for (const auto& plane : input) {
    inputs_outputs_->input_masks_mem_[offset + plane_index] = plane.mask;
    inputs_outputs_->input_val_mem_[offset + plane_index] = plane.value;
    plane_index++;
  }
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
void DmlDxComputation<DataType>::CopyOutputs(int start, int batch_size) {
  for (size_t i = 0; i < inputs_outputs_->output_tensors_step_.size(); i++) {
    const size_t stride = inputs_outputs_->output_tensors_step_[i];
    if (stride == 0) continue;
    CopyOutputChunk<DataType>(inputs_outputs_->output_tensors_data_[i],
                              inputs_outputs_->output_tensors_gpu_mapped_[i],
                              static_cast<size_t>(start) * stride,
                              static_cast<size_t>(batch_size) * stride);
  }
}

template <typename DataType>
void DmlDxComputation<DataType>::ComputeBlocking() {
  if (GetBatchSize() == 0) return;

  int batch_size = network_->batch_size_;
  if (batch_size < 0) {
    batch_size =
        std::max(static_cast<int>(input_size_), network_->min_batch_size_);
  }

  for (size_t start = 0; start < input_size_;) {
    int step = (input_size_ - start + batch_size - 1) / batch_size;
    if (step > network_->steps_) step = network_->steps_;
    int batch = batch_size * step;
    int actual_batch = std::min(batch, static_cast<int>(input_size_ - start));

    Ort::IoBinding binding{network_->session_[step - 1]};
    for (size_t i = 0; i < inputs_outputs_->output_tensors_step_.size(); i++) {
      const int size = inputs_outputs_->output_tensors_step_[i];
      if (size == 0) continue;
      const int64_t dims[] = {batch, size};
      auto* output = reinterpret_cast<DataType*>(
          inputs_outputs_->output_tensors_ort_allocation_[i]);
      binding.BindOutput(
          network_->outputs_[i].c_str(),
          Ort::Value::CreateTensor<DataType>(inputs_outputs_->memory_info_,
                                             output, size * batch, dims, 2));
    }

    const int64_t dims[] = {batch, kInputPlanes, 8, 8};
    auto* input =
        reinterpret_cast<DataType*>(inputs_outputs_->input_ort_allocation_);
    binding.BindInput(network_->inputs_[0].c_str(),
                      Ort::Value::CreateTensor<DataType>(
                          inputs_outputs_->memory_info_, input,
                          batch * kInputPlanes * 8 * 8, dims, 4));

    inputs_outputs_->input_masks_mem_gpu_.offset = start * sizeof(DataType);
    inputs_outputs_->input_val_mem_gpu_.offset = start * sizeof(DataType);
    {
      std::lock_guard<std::mutex> lock(network_->lock_);

      network_->dx_context_.ResetCL(nullptr, nullptr,
                                    network_->command_list_needs_reset_);
      network_->dx_context_.getShaderWrapper()->ExpandPlanes(
          network_->dx_context_.getCommandList(),
          inputs_outputs_->input_tensor_gpu_,
          inputs_outputs_->input_masks_mem_gpu_,
          inputs_outputs_->input_val_mem_gpu_, batch, network_->fp16_,
          network_->bf16_);
      network_->dx_context_.UavBarrier();
      network_->dx_context_.FlushCL();
      network_->command_list_needs_reset_ = true;

      binding.SynchronizeInputs();
      network_->session_[step - 1].Run(Ort::RunOptions{}, binding);
      binding.SynchronizeOutputs();
    }
    CopyOutputs(static_cast<int>(start), actual_batch);
    start += batch;
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

  Ort::ThrowOnError(dml_api_->SessionOptionsAppendExecutionProvider_DML1(
      options, dml_device_, dx_context_.getCommandQueue()));
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
  dml_api_ = GetDmlApi();
  dml_device_ = CreateDmlDevice(dx_context_.getDevice());

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

DmlDxNetwork::~DmlDxNetwork() {
  if (dml_device_) dml_device_->Release();
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
