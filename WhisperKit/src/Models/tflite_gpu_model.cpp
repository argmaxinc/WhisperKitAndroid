//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#if GPU_DELEGATE
#include "tflite_gpu_model.hpp"
#include <filesystem>   // C++ 17 or later
#include "tensorflow/lite/optional_debug_tools.h"
#include "ml_drift_delegate/ml_drift_cl.h"

using namespace std;

TFLiteGPU::TFLiteGPU(const string& name)
:TFLiteModel(name)
{
}

TFLiteGPU::~TFLiteGPU() {
    uninitialize();
}

bool TFLiteGPU::initialize(
    string model_path, 
    string lib_dir,
    string cache_dir,
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
    if (!_delegates.empty()) {
        for (int i = 0; i < _delegates.size(); ++i) {
            if (_delegates[i] == nullptr) {
                continue;
            }
            if (_delegate_types[i]
                == BackendType::WHISPERKIT_BACKEND_EXPERIMENTAL) {
                TfLiteDeleteMlDriftClDelegate(_delegates[i]);
            }
        }
    }
    _delegates.clear();
    _delegate_types.clear();

    if (_interpreter.get() != nullptr) {
        _interpreter->Cancel();
        _interpreter.reset(nullptr);
    }
}

bool TFLiteGPU::create_interpreter_delegate(string model_path) 
{
    _model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (_model.get() == nullptr) 
        return false; 

    tflite::ops::builtin::BuiltinOpResolver tflite_resolver;
    tflite::InterpreterBuilder builder(*_model, tflite_resolver);
    TFLITE_FUNCTION_CHECK(builder(&_interpreter))

    _interpreter->SetAllowFp16PrecisionForFp32(true);

    //  other options (priority, precision loss allowed)
    //  also causes 2x impact.
    MlDriftClDelegateOptions* mldrift_options
        = MlDriftClDelegateDefaultOptions();
    TfLiteDelegate* delegate = TfLiteCreateMlDriftClDelegate(mldrift_options);
    if (delegate == nullptr) {
        LOGI("Failed to create GPU delegate\n");
    } else {
        _delegates.push_back(delegate);
        _delegate_types.push_back(
            BackendType::WHISPERKIT_BACKEND_EXPERIMENTAL);
    }

    const auto processor_count = thread::hardware_concurrency();
    _interpreter->SetNumThreads(processor_count);

    return true;
}
#endif
