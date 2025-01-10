//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#if QNN_DELEGATE
#include "tflite_qnn_model.hpp"
#include <filesystem>   // C++ 17 or later
#include "tensorflow/lite/optional_debug_tools.h"

using namespace std;

TFLiteQNN::TFLiteQNN(const string& name)
:TFLiteModel(name)
{
}

TFLiteQNN::~TFLiteQNN() {
    uninitialize();
}

bool TFLiteQNN::initialize(
    string model_path, 
    string lib_dir,
    string cache_dir,
    int backend, 
    bool debug)
{
    set_dirs(model_path, lib_dir, cache_dir);

    _options = TfLiteQnnDelegateOptionsDefault();
    _options.backend_type = kUndefinedBackend;

    switch(backend){
        case kHtpBackend:
            _options.backend_type = kHtpBackend;
            _options.htp_options.precision = kHtpFp16;
            _options.htp_options.performance_mode = kHtpHighPerformance; 
                //kHtpSustainedHighPerformance; 
            _options.htp_options.useConvHmx = true;
            break;
        case kGpuBackend:
            _options.backend_type = kGpuBackend;
            _options.gpu_options.precision = kGpuFp16;
            _options.gpu_options.performance_mode = kGpuHigh;
            break;
        default:
            LOGI("%s: delegate to TFLite GPU backend..\n", _model_name.c_str());
            _options.backend_type = kUndefinedBackend;
            break;
    }

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

void TFLiteQNN::uninitialize() {
    if (_delegate != nullptr) {
        if (_options.backend_type == kUndefinedBackend)
            TfLiteGpuDelegateV2Delete(_delegate);
        else
            TfLiteQnnDelegateDelete(_delegate);
        _delegate = nullptr;
    }

    TFLiteModel::uninitialize();
}

bool TFLiteQNN::create_interpreter_delegate(string model_path) 
{
    _model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (_model.get() == nullptr) 
        return false; 

    if (_options.backend_type == kUndefinedBackend){
        tflite::ops::builtin::BuiltinOpResolver tflite_resolver;
        tflite::InterpreterBuilder builder(*_model, tflite_resolver);
        TFLITE_FUNCTION_CHECK(builder(&_interpreter))

        TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
        gpu_options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
        gpu_options.serialization_dir = _cache_dir.c_str();
        gpu_options.model_token = "model_token";;
        _delegate = TfLiteGpuDelegateV2Create(&gpu_options);
    } else {
        _options.skel_library_dir = _lib_dir.c_str();
        _options.cache_dir = _cache_dir.c_str();
        _options.model_token = _model_token.c_str();

        tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates qnn_resolver;
        tflite::InterpreterBuilder builder(*_model, qnn_resolver);
        TFLITE_FUNCTION_CHECK(builder(&_interpreter))
        _delegate = TfLiteQnnDelegateCreate(&_options);
    }

    if (_delegate == nullptr) 
        return false; 

    const auto processor_count = thread::hardware_concurrency();
    _interpreter->SetNumThreads(processor_count);

    return true;
}
#endif