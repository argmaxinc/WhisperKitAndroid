//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include <tflite_model.hpp>
#include <filesystem>   // C++ 17 or later

using namespace std;

TFLiteModel::TFLiteModel(const string& name)
{
    _delegate = nullptr;
    _model_name = name;
}

TFLiteModel::~TFLiteModel() {
    uninitialize();
}

bool TFLiteModel::initialize(
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

    return true;
}

void TFLiteModel::uninitialize() {
    if (_interpreter.get() != nullptr) {
        //LOGI("Deleted interpreter & delegate for %s\n", _model_name.c_str());
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

bool TFLiteModel::create_interpreter_delegate(string model_path) 
{
    _model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (_model.get() == nullptr) 
        return false; 

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*_model, resolver);
    TFLITE_FUNCTION_CHECK(builder(&_interpreter))

    const auto processor_count = thread::hardware_concurrency();
    _interpreter->SetNumThreads(processor_count/2);

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

    for(int idx = 0; idx < _interpreter->inputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->inputs()[idx]);
        void * input_ptr;
        switch(tensor->type){
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

    for(int idx = 0; idx < _interpreter->outputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->outputs()[idx]);
        void * output_ptr;
        switch(tensor->type){
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
    if(measure_time) {
        before_exec = chrono::high_resolution_clock::now();
    }

    _interpreter->Invoke();

    if(measure_time) {
        auto after_exec = chrono::high_resolution_clock::now();
        float interval_infs =
            chrono::duration_cast<std::chrono::microseconds>(
                after_exec - before_exec).count() / 1000.0;
        _latencies.push_back(interval_infs);
    }
}

void TFLiteModel::print_tensor_dims(){
    LOGI("=== tensors of %s ===\n", _model_name.c_str());
    LOGI("** input size: %zu\n", _interpreter->inputs().size());
    for(int idx = 0; idx < _interpreter->inputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->inputs()[idx]);
        LOGI("  name: %s\n", tensor->name);
        LOGI("  bytes: %zu\n", tensor->bytes);
        LOGI("  type: %d\n", tensor->type);
        LOGI("  input tensor dims %d : (", tensor->dims->size);

        for(int i = 0; i < tensor->dims->size; i++) {
            LOGI("%d", tensor->dims->data[i]);
            if(i < tensor->dims->size - 1) {
                LOGI(", ");
            } else {
                LOGI(")");
            }
        }
        LOGI("\n\n"); 
    }

    LOGI("** output size: %zu\n", _interpreter->outputs().size());
    for(int idx = 0; idx < _interpreter->outputs().size(); idx++) {
        auto* tensor = _interpreter->tensor(_interpreter->outputs()[idx]);
        LOGI("  name: %s\n", tensor->name);
        LOGI("  bytes: %zu\n", tensor->bytes);
        LOGI("  type: %d\n", tensor->type);
        LOGI("  input tensor dims %d : (", tensor->dims->size);

        for(int i = 0; i < tensor->dims->size; i++) {
            LOGI("%d", tensor->dims->data[i]);
            if(i < tensor->dims->size - 1) {
                LOGI(", ");
            } else {
                LOGI(")");
            }
        }
        LOGI("\n\n"); 
    }
    LOGI("==================================\n");
}

void TFLiteModel::set_dirs(
    string filename, 
    string lib_dir,
    string cache_dir
) {
    vector<string> model_sizes = {"tiny", "base", "small"};
    _lib_dir = lib_dir;
    // NOTE: for Android, fs /sdcard does not support flock() operation,
    // so we need to use /data/* such as /data/local/tmp/cache
    _cache_dir = cache_dir;

    if (!filesystem::exists(_cache_dir)) {
        LOGI("Creating cache directory: %s\n", _cache_dir.c_str());
        filesystem::create_directory(_cache_dir);
    }

    for(auto& size : model_sizes){
        size_t found = filename.find(size);
        if (found!=std::string::npos){
            _model_token = _model_name + "_" + size;
            return;
        }
    }
    _model_token = _model_name;
}

void TFLiteModel::save_tensor(string filename, char* tensor, int size)
{
    fstream fout;
    fout.open(filename, fstream::out);
    fout.write(tensor, size);
    fout.close();
}

unique_ptr<json> TFLiteModel::get_latency_json() {
    auto perfjson = make_unique<json>();

    (*perfjson)["inf"] = _latencies.size();
    auto avg = get_latency_avg();
    (*perfjson)["avg"] = avg; 

    vector<float> diff(_latencies.size());
    transform(_latencies.begin(), _latencies.end(), diff.begin(), [avg](double x) { return x - avg; });
    float sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    auto stdev = sqrt(sq_sum / _latencies.size());
    (*perfjson)["std"] = ceil(stdev * 100.0) / 100.0;

    auto med = get_latency_median();
    (*perfjson)["med"] = ceil(med * 100.0) / 100.0;

    return perfjson;
}

float TFLiteModel::get_latency_sum() {
    auto sum = accumulate(_latencies.begin(), _latencies.end(), 0) / 1000.0;
    return sum;
}

float TFLiteModel::get_latency_avg() {
    auto avg = reduce(_latencies.begin(), _latencies.end()) 
                / _latencies.size();
    return ((int)(avg * 100.0) / 100.0);
}

float TFLiteModel::get_latency_median() {
    if(_latencies.empty()) {
        return 0.0f;
    }

    const auto middleItr = _latencies.begin() + _latencies.size() / 2;
    nth_element(_latencies.begin(), middleItr, _latencies.end());
    if (_latencies.size() % 2 == 0) {
        const auto leftMiddleItr = max_element(_latencies.begin(), middleItr);
        return (*leftMiddleItr + *middleItr) / 2;
    } else {
        return *middleItr;
    }
}
