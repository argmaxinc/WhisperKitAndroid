//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <assert.h>
#include <memory>
#include <numeric>
#include <filesystem>
#include <sys/stat.h>
#include <nlohmann/json.hpp>
#include <tflite_msg.hpp>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/core/c/common.h"

#define TFLITE_FUNCTION_CHECK(x)                          \
    if ((x) != kTfLiteOk) {                               \
        fprintf(stderr, "Error at %s:%d\n", __FUNCTION__, __LINE__); \
        return false;                                         \
    }

using namespace std;
using json = nlohmann::json;

class TFLiteModel {
public:
    TFLiteModel(const string& name);
    virtual ~TFLiteModel();

    bool initialize(
        string model_path, 
        string lib_dir,
        string cache_path, 
        int backend, 
        bool debug=false
    );
    void uninitialize();
    virtual void invoke(bool measure_time=false);

    mutex* get_mutex() { return &_mutex; }
    void read_input_file(string input_file, int idx);
    void read_input_data(char* input_data, int idx);
    vector<pair<char*, int>> get_input_ptrs();
    vector<pair<char*, int>> get_output_ptrs();

    void print_tensor_dims();
    unique_ptr<json> get_latency_json();
    float get_latency_median();
    float get_latency_sum();
    float get_latency_avg();
    int get_inference_num()     { return _latencies.size(); }

    static void save_tensor(string filename, char* tensor, int size);

protected: 
    mutex _mutex;
    unique_ptr<tflite::FlatBufferModel> _model;
    unique_ptr<tflite::Interpreter> _interpreter;
    TfLiteDelegate* _delegate = nullptr;
    string _model_name;
    string _lib_dir; 
    string _cache_dir; 
    string _model_token;
    vector<float> _latencies;

    vector<pair<char*, int>> _input_ptrs;
    vector<pair<char*, int>> _output_ptrs;    

    bool create_interpreter_delegate(string model_path);
    bool allocate_tensors();
    void modify_graph_delegate();
    void set_dirs(string filename, string lib_dir, string cache_dir);
};
