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

using json = nlohmann::json;

class TFLiteModel {
public:
    TFLiteModel(const std::string& name);
    virtual ~TFLiteModel();

    bool initialize(
        std::string model_path, 
        std::string lib_dir,
        std::string cache_path, 
        int backend, 
        bool debug=false
    );
    void uninitialize();
    virtual void invoke(bool measure_time=false);

    std::mutex* get_mutex() { return &_mutex; }
    void read_input_file(std::string input_file, int idx);
    void read_input_data(char* input_data, int idx);
    std::vector<std::pair<char*, int>> get_input_ptrs();
    std::vector<std::pair<char*, int>> get_output_ptrs();

    void print_tensor_dims();
    std::unique_ptr<json> get_latency_json();
    float get_latency_median();
    float get_latency_sum();
    float get_latency_avg();
    int get_inference_num()     { return _latencies.size(); }

    static void save_tensor(std::string filename, char* tensor, int size);

protected: 
    std::mutex _mutex;
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    TfLiteDelegate* _delegate = nullptr;
    std::string _model_name;
    std::string _lib_dir; 
    std::string _cache_dir; 
    std::string _model_token;
    std::vector<float> _latencies;

    std::vector<std::pair<char*, int>> _input_ptrs;
    std::vector<std::pair<char*, int>> _output_ptrs;    

    bool create_interpreter_delegate(std::string model_path);
    bool allocate_tensors();
    void modify_graph_delegate();
    void set_dirs(std::string filename, std::string lib_dir, std::string cache_dir);
};
