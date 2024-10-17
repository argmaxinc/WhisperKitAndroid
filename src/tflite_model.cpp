//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include <tflite_model.hpp>
// C++ 17 or later
#include <filesystem>

TFLiteModel::TFLiteModel(const string& name) {
    _delegate = nullptr;
    _model_name = name;
}

TFLiteModel::~TFLiteModel() { uninitialize(); }

bool TFLiteModel::initialize(string model_path, string lib_dir, string cache_dir, int backend) {
    _options = TfLiteQnnDelegateOptionsDefault();

    switch (backend) {
        case kHtpBackend:
            _options.backend_type = kHtpBackend;
            _options.htp_options.precision = kHtpFp16;
            _options.htp_options.performance_mode = kHtpHighPerformance;
            // kHtpSustainedHighPerformance;
            _options.htp_options.useConvHmx = true;
            break;
        case kGpuBackend:
            _options.backend_type = kGpuBackend;
            _options.gpu_options.precision = kGpuFp16;
            _options.gpu_options.performance_mode = kGpuHigh;
            break;
        default:
            LOGI("Falling back to default TFLite GPU backend..");
            _options.backend_type = kUndefinedBackend;
            break;
    }

    set_dirs(model_path, lib_dir, cache_dir);
    if (!create_interpreter_delegate(model_path)) {
        return false;
    }
    if (!allocate_tensors()) {
        return false;
    }
    modify_graph_delegate();
    return true;
}

void TFLiteModel::uninitialize() {
    if (_delegate != nullptr) {
        if (_options.backend_type == kUndefinedBackend)
            TfLiteGpuDelegateV2Delete(_delegate);
        else
            TfLiteQnnDelegateDelete(_delegate);
        _delegate = nullptr;
    }
    if (_interpreter.get() != nullptr) {
        LOGI("Deleted interpreter & delegate for %s\n", _model_name.c_str());
        _interpreter->Cancel();
        _interpreter.reset(nullptr);
    }
}

bool TFLiteModel::allocate_tensors() {
    TFLITE_FUNCTION_CHECK(_interpreter->AllocateTensors())

    return true;
}

void TFLiteModel::modify_graph_delegate() {
    // Replace the original delegate with the new one.
    _interpreter->ModifyGraphWithDelegate(_delegate);
}

bool TFLiteModel::create_interpreter_delegate(string model_path) {
    if (_options.backend_type != kUndefinedBackend) {
        _options.skel_library_dir = _lib_dir.c_str();
        _options.cache_dir = _cache_dir.c_str();
        _options.model_token = _model_token.c_str();
    }

    _model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (_options.backend_type == kUndefinedBackend) {
        tflite::ops::builtin::BuiltinOpResolver tflite_resolver;
        tflite::InterpreterBuilder builder(*_model, tflite_resolver);
        TFLITE_FUNCTION_CHECK(builder(&_interpreter))
    } else {
        tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates qnn_resolver;
        tflite::InterpreterBuilder builder(*_model, qnn_resolver);
        TFLITE_FUNCTION_CHECK(builder(&_interpreter))
    }

    if (_options.backend_type == kUndefinedBackend)
        _delegate = TfLiteGpuDelegateV2Create(nullptr);
    else
        _delegate = TfLiteQnnDelegateCreate(&_options);

    const auto processor_count = thread::hardware_concurrency();
    _interpreter->SetNumThreads(processor_count / 2);

    return true;
}

void TFLiteModel::read_input_file(string input_file, int idx) {
    get_input_ptrs();
    ifstream fin(input_file, ios::binary);
    auto ptr = _input_ptrs[idx];

    auto data_size = fin.tellg();
    if (data_size > ptr.second) {
        data_size = ptr.second;
    }

    fin.seekg(0, fin.beg);
    fin.read(reinterpret_cast<char*>(ptr.first), data_size);
    fin.close();
}

void TFLiteModel::read_input_data(char* input_data, int idx) {
    get_input_ptrs();
    auto ptr = _input_ptrs[idx];

    memcpy(reinterpret_cast<char*>(ptr.first), input_data, ptr.second);
}

vector<pair<char*, int>> TFLiteModel::get_input_ptrs() {
    if (!_input_ptrs.empty()) {
        return _input_ptrs;
    }

    for (int idx = 0; idx < _interpreter->inputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->inputs()[idx]);
        void* input_ptr;
        switch (tensor->type) {
            case kTfLiteFloat32:
                input_ptr = _interpreter->typed_input_tensor<float>(idx);
                break;
            case kTfLiteInt32:
                input_ptr = _interpreter->typed_input_tensor<int>(idx);
                break;
            default:
                fprintf(stderr, "Error: unsupported tensor type\n");
                exit(-1);
        }
        _input_ptrs.push_back(make_pair(reinterpret_cast<char*>(input_ptr), tensor->bytes));
    }
    return _input_ptrs;
}

vector<pair<char*, int>> TFLiteModel::get_output_ptrs() {
    if (!_output_ptrs.empty()) {
        return _output_ptrs;
    }

    for (int idx = 0; idx < _interpreter->outputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->outputs()[idx]);
        void* output_ptr;
        switch (tensor->type) {
            case kTfLiteFloat32:
                output_ptr = _interpreter->typed_output_tensor<float>(idx);
                break;
            case kTfLiteInt32:
                output_ptr = _interpreter->typed_output_tensor<int>(idx);
                break;
            default:
                fprintf(stderr, "Error: unsupported tensor type\n");
                exit(-1);
        }
        _output_ptrs.push_back(make_pair(reinterpret_cast<char*>(output_ptr), tensor->bytes));
    }
    return _output_ptrs;
}

void TFLiteModel::invoke(bool measure_time) {
    chrono::time_point<chrono::high_resolution_clock> before_exec;
    if (measure_time) {
        before_exec = chrono::high_resolution_clock::now();
    }

    _interpreter->Invoke();

    if (measure_time) {
        auto after_exec = chrono::high_resolution_clock::now();
        float interval_infs =
            chrono::duration_cast<std::chrono::microseconds>(after_exec - before_exec).count() / 1000.0;
        _latencies.push_back(interval_infs);
    }
}

void TFLiteModel::print_tensor_dims() {
    LOGI("=== tensors of %s ===\n", _model_name.c_str());
    LOGI("** input size: %zu\n", _interpreter->inputs().size());
    for (int idx = 0; idx < _interpreter->inputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->inputs()[idx]);
        LOGI("  name: %s\n", tensor->name);
        LOGI("  bytes: %zu\n", tensor->bytes);
        LOGI("  type: %d\n", tensor->type);
        LOGI("  input tensor dims %d : (", tensor->dims->size);

        for (int i = 0; i < tensor->dims->size; i++) {
            LOGI("%d", tensor->dims->data[i]);
            if (i < tensor->dims->size - 1) {
                LOGI(", ");
            } else {
                LOGI(")");
            }
        }
        LOGI("\n\n");
    }

    LOGI("** output size: %zu\n", _interpreter->outputs().size());
    for (int idx = 0; idx < _interpreter->outputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->outputs()[idx]);
        LOGI("  name: %s\n", tensor->name);
        LOGI("  bytes: %zu\n", tensor->bytes);
        LOGI("  type: %d\n", tensor->type);
        LOGI("  input tensor dims %d : (", tensor->dims->size);

        for (int i = 0; i < tensor->dims->size; i++) {
            LOGI("%d", tensor->dims->data[i]);
            if (i < tensor->dims->size - 1) {
                LOGI(", ");
            } else {
                LOGI(")");
            }
        }
        LOGI("\n\n");
    }
    LOGI("==================================\n");
}

void TFLiteModel::set_dirs(string filename, string lib_dir, string cache_dir) {
    vector<string> model_sizes = {"tiny", "base", "small"};
    _lib_dir = lib_dir;
    // NOTE: for Android, fs /sdcard does not support flock() operation,
    // so we need to use /data/* such as /data/local/tmp/cache
    _cache_dir = cache_dir;

    if (!filesystem::exists(_cache_dir)) {
        LOGI("Creating cache directory: %s\n", _cache_dir.c_str());
        filesystem::create_directory(_cache_dir);
    }

    for (auto& size : model_sizes) {
        size_t found = filename.find(size);
        if (found != std::string::npos) {
            _model_token = _model_name + "_" + size;
            return;
        }
    }
    _model_token = _model_name;
}

void TFLiteModel::save_tensor(string filename, char* tensor, int size) {
    fstream fout;
    fout.open(filename, fstream::out);
    fout.write(tensor, size);
    fout.close();
}

unique_ptr<json> TFLiteModel::get_latency_json() {
    auto perfjson = make_unique<json>();

    int i = 0;
    for (auto& latency : _latencies) {
        (*perfjson)[to_string(i++)] = ceil(latency * 100.0) / 100.0;
    }
    auto avg = reduce(_latencies.begin(), _latencies.end()) / _latencies.size();
    (*perfjson)["avg"] = ceil(avg * 100.0) / 100.0;
    return perfjson;
}
