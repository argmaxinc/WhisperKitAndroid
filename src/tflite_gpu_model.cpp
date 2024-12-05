//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include <tflite_gpu_model.hpp>
#include <filesystem>   // C++ 17 or later
#include "tensorflow/lite/optional_debug_tools.h"

TFLiteGPU::TFLiteGPU(const std::string& name)
:TFLiteModel(name)
{
}

TFLiteGPU::~TFLiteGPU() {
    uninitialize();
}

bool TFLiteGPU::initialize(
    std::string model_path, 
    std::string lib_dir,
    std::string cache_dir,
    int backend, 
    bool debug)
{
    set_dirs(model_path, lib_dir, cache_dir);

    if (!create_interpreter_delegate(model_path)) {
        LOGE("Failed with create_interpreter_delegate..\n");
        return false; 
    }
    if (!allocate_tensors()) {
        LOGE("Failed with allocate_tensors..\n");
        return false;
    }

    modify_graph_delegate();

    if(debug){
        LOGI("\n========== %s delegation info ==========\n", _model_name.c_str());
        tflite::PrintInterpreterState(_interpreter.get());
    }
    return true;
}

void TFLiteGPU::uninitialize() {
    if (_delegate != nullptr) {
        TfLiteGpuDelegateV2Delete(_delegate);
        _delegate = nullptr;
    }

    TFLiteModel::uninitialize();
}

bool TFLiteGPU::create_interpreter_delegate(std::string model_path) 
{
    _model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (_model.get() == nullptr) 
        return false; 

    tflite::ops::builtin::BuiltinOpResolver tflite_resolver;
    tflite::InterpreterBuilder builder(*_model, tflite_resolver);
    TFLITE_FUNCTION_CHECK(builder(&_interpreter))

    TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
    gpu_options.serialization_dir = _cache_dir.c_str();
    gpu_options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    gpu_options.max_delegated_partitions = 3;
    _delegate = TfLiteGpuDelegateV2Create(&gpu_options);

    if (_delegate == nullptr) 
        return false; 

    const auto processor_count = std::thread::hardware_concurrency();
    _interpreter->SetNumThreads(processor_count-1);

    return true;
}
