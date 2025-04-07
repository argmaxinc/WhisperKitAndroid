//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include "tflite_model.hpp"
#include <filesystem>   // C++ 17 or later

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"

#define TFLITE_SCHEMA_VERSION 3


using namespace std;

TFLiteModel::TFLiteModel(const string& name)
{
    _delegate = nullptr;
    _model_name = name;
}

TFLiteModel::~TFLiteModel() {
    uninitialize();
}

bool TFLiteModel::buildSimpleVADModel() {


    /* Structure:

    -- create op codes composing the model;
    -- define the input tensors (here: input frames, and the energy threshold as a bias)
    -- define the intermediate tensors between the nodes
    -- create the output tensor
    -- assemble the graph, connecting the nodes by the tensor indices
    -- build the graph, assign the model to an interpreter
    */
    
    const std::vector<int32_t> input_shape = {150, 1600};
    const std::vector<int32_t> scalar_shape = {}; // For RMSE and bias

    auto op_code_square = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_SQUARE);
    auto op_code_mean = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_MEAN);
    auto op_code_sqrt = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_SQRT);
    auto op_code_sub = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_SUB);
    
    auto input_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(input_shape), tflite::TensorType_FLOAT32,
        0, _builder.CreateString("input_frames"));
    auto bias_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(scalar_shape), tflite::TensorType_FLOAT32, 
        0, _builder.CreateString("energy_threshold"));

    auto squared_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(input_shape), tflite::TensorType_FLOAT32);
    auto reduced_mean_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(scalar_shape), tflite::TensorType_FLOAT32);
    auto rmse_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(scalar_shape), tflite::TensorType_FLOAT32);

    std::vector<int32_t> mean_axis_data = {1};
    auto mean_axis_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(mean_axis_data.data()), mean_axis_data.size() * sizeof(int32_t));
    auto mean_axis_buffer = tflite::CreateBuffer(_builder, mean_axis_buffer_data);
    auto mean_axis_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,                    // type
        1,                                   // buffer index
        _builder.CreateString("mean_axis"),   // name
        /*quantization=*/0,                  // optional quantization
        /*is_variable=*/false,              // is_variable
        /*sparsity=*/0,                     // optional sparsity
        /*shape_signature=*/0               // optional shape signature
    );

    auto output_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(scalar_shape), tflite::TensorType_FLOAT32,
        0, _builder.CreateString("output_0"));

    // Operators
    auto square_op = tflite::CreateOperator(_builder, 0, _builder.CreateVector<int32_t>({0}), _builder.CreateVector<int32_t>({2}));
    auto mean_op = tflite::CreateOperator(_builder, 1, _builder.CreateVector<int32_t>({2, 3}), _builder.CreateVector<int32_t>({4}));
    auto sqrt_op = tflite::CreateOperator(_builder, 2, _builder.CreateVector<int32_t>({4}), _builder.CreateVector<int32_t>({5}));
    auto sub_op = tflite::CreateOperator(_builder, 3, _builder.CreateVector<int32_t>({5, 1}), _builder.CreateVector<int32_t>({6}));

    // Create graph
    auto graph = tflite::CreateSubGraph(_builder, 
        _builder.CreateVector({input_tensor, bias_tensor, squared_tensor, mean_axis_tensor, reduced_mean_tensor, rmse_tensor, output_tensor}),
        _builder.CreateVector<int32_t>({0, 1}),
        _builder.CreateVector<int32_t>({6}),
        _builder.CreateVector({square_op, mean_op, sqrt_op, sub_op})
    );
    

    auto buffer = tflite::CreateBuffer(_builder, _builder.CreateVector({}));


    auto model = tflite::CreateModel(_builder, TFLITE_SCHEMA_VERSION,
        _builder.CreateVector({op_code_square, op_code_mean, op_code_sqrt, op_code_sub}),
        _builder.CreateVector({graph}),
        _builder.CreateString("RMSE Model"),
        _builder.CreateVector({buffer, mean_axis_buffer}));
    
    _builder.Finish(model, tflite::ModelIdentifier());

    // Wrap in FlatBufferModel object
    _model =
        tflite::FlatBufferModel::BuildFromBuffer(
            reinterpret_cast<const char*>(_builder.GetBufferPointer()), _builder.GetSize());

    if (!_model) {
        std::cerr << "Failed to build TFLite model in memory!" << std::endl;
        return false;
    }

    // Build an interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder interpreter_builder(*_model, resolver);
    interpreter_builder(&_interpreter);

    if (!_interpreter) {
        std::cerr << "Failed to create TFLite interpreter!" << std::endl;
        return false;
    }

    // Allocate memory for tensors (see other default initializer method, we put here instead of initializeModelInMemory)
    if (_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return false;
    }
    return true;

}

bool TFLiteModel::initializeModelInMemory(
    WhisperKit::InMemoryModel::ModelType model_type,
    bool debug
) {

    switch(model_type) {
        case WhisperKit::InMemoryModel::ModelType::kSimpleVADModel:
            return buildSimpleVADModel();
        case WhisperKit::InMemoryModel::ModelType::kSimplePostProcessingModel:
            return buildPostProcModel();
        default:
            LOGE("Unsupported model type for in-memory model\n");
            return false;
    }

    return false;
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

bool TFLiteModel::buildPostProcModel() {

    /* Structure:

    -- create op codes composing the model;
    -- define the input tensor (here: logits)
    -- define the intermediate tensors between the nodes and output tensors
    -- assemble the graph, connecting the nodes by the tensor indices
    -- build the graph, assign the model to an interpreter
    */
    constexpr int LOGITS_SIZE = 51864;
    constexpr int TOKEN_TIMESTAMP_BEGIN = 50363;
    constexpr int TOKEN_NO_SPEECH = 50361;

    
    const std::vector<int32_t> input_shape = {LOGITS_SIZE};
    const std::vector<int32_t> text_slice_shape = {TOKEN_TIMESTAMP_BEGIN};
    const std::vector<int32_t> timestamp_slice_shape = {LOGITS_SIZE-TOKEN_TIMESTAMP_BEGIN};
    const std::vector<int32_t> unary_shape = {1};
    const std::vector<int32_t> output_shape = {3};

    // Op Codes
    auto op_code_log_softmax = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_LOG_SOFTMAX);
    auto op_code_slice = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_SLICE);
    auto op_code_exp = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_EXP);
    auto op_code_sum = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_SUM);
    auto op_code_log = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_LOG);
    auto op_code_reduce_max = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_REDUCE_MAX);
    auto op_code_sub = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_SUB);
    auto op_code_add = tflite::CreateOperatorCode(_builder, tflite::BuiltinOperator_ADD);
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    
    // IO tensors
    auto input_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(input_shape), tflite::TensorType_FLOAT32,
        0, _builder.CreateString("logits"));
    
    // Intermediary tensors
    // Holds results of LogSoftmax operation on logits input
    auto logsoftmax_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(input_shape), tflite::TensorType_FLOAT32);
    
    // Holds results of the Slice operations on logprobs
    auto text_slice_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(text_slice_shape), tflite::TensorType_FLOAT32);
    auto timestamp_slice_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(timestamp_slice_shape), tflite::TensorType_FLOAT32);
    auto nospeech_slice_tensor =tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(unary_shape),
                                                    tflite::TensorType_FLOAT32, 0, _builder.CreateString("no_speech_logprob")); 
    
    // Holds results of max reduction operation on the text slice
    auto text_max_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(unary_shape), tflite::TensorType_FLOAT32, 0, _builder.CreateString("text_logprob"));
    
    // Holds result of exp operation on the timestamp slice
    auto timestamp_exp_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(timestamp_slice_shape), tflite::TensorType_FLOAT32);
    // Holds results of sum reduction operation on the results of the exp operation
    auto timestamp_sum_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(unary_shape), tflite::TensorType_FLOAT32);
    // Holds results of log operation on the timestamp summed probability
    auto timestamp_log_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(unary_shape), tflite::TensorType_FLOAT32,
                                                    0, _builder.CreateString("timestamp_log"));

    // Buffers
    // Empty Buffer for IO tensors
    auto empty_buffer = tflite::CreateBuffer(_builder);
    // Text
    std::vector<int32_t> text_slice_begin_data = {0};
    std::vector<int32_t> text_slice_size_data = {TOKEN_TIMESTAMP_BEGIN};
    auto text_slice_begin_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        text_slice_begin_data.data()), text_slice_begin_data.size() * sizeof(int32_t));
    auto text_slice_size_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        text_slice_size_data.data()), text_slice_size_data.size() * sizeof(int32_t));
    auto text_slice_begin_buffer = tflite::CreateBuffer(_builder, text_slice_begin_buffer_data);
    auto text_slice_size_buffer = tflite::CreateBuffer(_builder, text_slice_size_buffer_data);
    
    // Timestamp
    std::vector<int32_t> timestamp_slice_begin_data = {TOKEN_TIMESTAMP_BEGIN};
    std::vector<int32_t> timestamp_slice_size_data = {LOGITS_SIZE-TOKEN_TIMESTAMP_BEGIN};
    auto timestamp_slice_begin_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        timestamp_slice_begin_data.data()), timestamp_slice_begin_data.size() * sizeof(int32_t));
    auto timestamp_slice_size_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        timestamp_slice_size_data.data()), timestamp_slice_size_data.size() * sizeof(int32_t));
    auto timestamp_slice_begin_buffer = tflite::CreateBuffer(_builder, timestamp_slice_begin_buffer_data);
    auto timestamp_slice_size_buffer = tflite::CreateBuffer(_builder, timestamp_slice_size_buffer_data);

    // No speech token
    std::vector<int32_t> nospeech_slice_begin_data = {TOKEN_NO_SPEECH};
    std::vector<int32_t> nospeech_slice_size_data = {1};
    auto nospeech_slice_begin_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        nospeech_slice_begin_data.data()), nospeech_slice_begin_data.size() * sizeof(int32_t));
    auto nospeech_slice_size_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        nospeech_slice_size_data.data()), nospeech_slice_size_data.size() * sizeof(int32_t));
    auto nospeech_slice_begin_buffer = tflite::CreateBuffer(_builder, nospeech_slice_begin_buffer_data);
    auto nospeech_slice_size_buffer = tflite::CreateBuffer(_builder, nospeech_slice_size_buffer_data);

    // Axis 
    std::vector<int32_t> reduction_axis_data = {0};
    auto reduction_axis_buffer_data = _builder.CreateVector(reinterpret_cast<const uint8_t*>(
        reduction_axis_data.data()), reduction_axis_data.size() * sizeof(int32_t));
    auto reduction_axis_buffer = tflite::CreateBuffer(_builder, reduction_axis_buffer_data);
    
    // Constant Tensors
    // Text
    auto text_slice_begin_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        1,                                   // buffer index
        _builder.CreateString("text_slice_begin"));
    auto text_slice_size_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        2,                                   // buffer index
        _builder.CreateString("text_slice_size"));

    // Timestamp
    auto timestamp_slice_begin_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        3,                                   // buffer index
        _builder.CreateString("timestamp_slice_begin"));
    auto timestamp_slice_size_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        4,                                   // buffer index
        _builder.CreateString("timestamp_slice_size"));

    // No speech
    auto nospeech_slice_begin_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        5,                                   // buffer index
        _builder.CreateString("nospeech_slice_begin"));
    auto nospeech_slice_size_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        6,                                   // buffer index
        _builder.CreateString("nospeech_slice_size"));

    // Reduction Axis
    auto reduction_axis_tensor = tflite::CreateTensor(
        _builder,
        _builder.CreateVector<int32_t>({1}),  // shape
        tflite::TensorType_INT32,            // type
        7,                                   // buffer index
        _builder.CreateString("reduction_axis"));
    
    // Holds results of max reduction operation on the timestamp slice
    auto timestamp_max_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(unary_shape.data(), unary_shape.size()),
                                                    tflite::TensorType_FLOAT32, 0, _builder.CreateString("timestamp_max_tensor"));
    // Holds results of timestamp logprobs 'normalized' by max timestamp logprob
    auto timestamp_norm_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(timestamp_slice_shape.data(), timestamp_slice_shape.size()),
                                                    tflite::TensorType_FLOAT32, 0, _builder.CreateString("timestamp_norm_tensor"));
    auto timestamp_add_tensor = tflite::CreateTensor(_builder, _builder.CreateVector<int32_t>(unary_shape.data(), unary_shape.size()),
                                                    tflite::TensorType_FLOAT32, 0, _builder.CreateString("timestamp_logprob"));


    // Operators
    // Log probabilities calc
    auto log_softmax_op = tflite::CreateOperator(_builder, 
                                                0, /* op_code index */
                                                _builder.CreateVector<int32_t>({0}), /* input indices */
                                                _builder.CreateVector<int32_t>({1}) /* output indices */
                                            );
    // Slice operations
    auto text_slice_op = tflite::CreateOperator(_builder, 
                                                1, /* op_code index */
                                                _builder.CreateVector<int32_t>({1, 2, 3}), /* input indices */
                                                _builder.CreateVector<int32_t>({4}) /* output indices */
                                            );
    auto timestamp_slice_op = tflite::CreateOperator(_builder, 
                                                1, /* op_code index */
                                                _builder.CreateVector<int32_t>({1, 5, 6}), /* input indices */
                                                _builder.CreateVector<int32_t>({7}) /* output indices */
                                            );
    auto nospeech_slice_op = tflite::CreateOperator(_builder, 
                                                1, /* op_code index */
                                                _builder.CreateVector<int32_t>({1, 8, 9}), /* input indices */
                                                _builder.CreateVector<int32_t>({10}) /* output indices */
                                            );
    // Max reduction
    auto text_max_op = tflite::CreateOperator(_builder, 
                                                5, /* op_code index */
                                                _builder.CreateVector<int32_t>({4, 11}), /* input indices */
                                                _builder.CreateVector<int32_t>({12}) /* output indices */
                                            );

    auto timestamp_max_op = tflite::CreateOperator(_builder, 
                                                    5, /* op_code index */
                                                    _builder.CreateVector<int32_t>({7, 11}), /* input indices */
                                                    _builder.CreateVector<int32_t>({13}) /* output indices */
                                            );

    // Log sum exp
    auto timestamp_sub_op = tflite::CreateOperator(_builder, 
                                                    6, /* op_code index */
                                                    _builder.CreateVector<int32_t>({7, 13}), /* input indices */
                                                    _builder.CreateVector<int32_t>({14}) /* output indices */
                                            );
    auto timestamp_exp_op = tflite::CreateOperator(_builder, 
                                                    2, /* op_code index */
                                                    _builder.CreateVector<int32_t>({14}), /* input indices */
                                                    _builder.CreateVector<int32_t>({15}) /* output indices */
                                            );
    auto timestamp_sum_op = tflite::CreateOperator(_builder, 
                                                        3, /* op_code index */
                                                        _builder.CreateVector<int32_t>({15, 11}), /* input indices */
                                                        _builder.CreateVector<int32_t>({16}) /* output indices */
                                            );
    auto timestamp_log_op = tflite::CreateOperator(_builder, 
                                                        4, /* op_code index */
                                                        _builder.CreateVector<int32_t>({16}), /* input indices */
                                                        _builder.CreateVector<int32_t>({17}) /* output indices */
                                            );
    auto timestamp_add_op = tflite::CreateOperator(_builder, 
                                                        7, /* op_code index */
                                                        _builder.CreateVector<int32_t>({17, 13}), /* input indices */
                                                        _builder.CreateVector<int32_t>({18}) /* output indices */
                                            );
    

    // Create graph
    auto graph = tflite::CreateSubGraph(_builder, 
        // Tensors
        _builder.CreateVector({
            input_tensor,
            logsoftmax_tensor, 
            // Text Slice tensors
            text_slice_begin_tensor,
            text_slice_size_tensor,
            text_slice_tensor,
            // Timestamp slice tensors
            timestamp_slice_begin_tensor,
            timestamp_slice_size_tensor,
            timestamp_slice_tensor,
            // No speech slice tensors
            nospeech_slice_begin_tensor,
            nospeech_slice_size_tensor,
            nospeech_slice_tensor, // output[2]
            // Reduction axis
            reduction_axis_tensor,
            // Max reduction tensors
            text_max_tensor,
            timestamp_max_tensor, // output[1]
            // Log timestamp probability calc tensors
            timestamp_exp_tensor,
            timestamp_sum_tensor,
            timestamp_log_tensor,
            timestamp_norm_tensor, // output[0]
            timestamp_add_tensor
        }),
        // Input Indices
        _builder.CreateVector<int32_t>({0}),
        // Output Indices
        _builder.CreateVector<int32_t>({18, 12, 10}),
        // Operations
        _builder.CreateVector({log_softmax_op,
                              text_slice_op,
                              timestamp_slice_op,
                              nospeech_slice_op,
                              text_max_op,
                              timestamp_max_op,
                              timestamp_sub_op,
                              timestamp_exp_op,
                              timestamp_sum_op,
                              timestamp_log_op,
                              timestamp_add_op
                            })
    );

    auto model = tflite::CreateModel(_builder, TFLITE_SCHEMA_VERSION,
        // Operation Codes
        _builder.CreateVector({op_code_log_softmax,
                              op_code_slice,
                              op_code_exp,
                              op_code_sum,
                              op_code_log,
                              op_code_reduce_max,
                              op_code_sub,
                              op_code_add}),
        _builder.CreateVector({graph}),
        _builder.CreateString("Post Process Model"),
        // Buffers
        _builder.CreateVector({empty_buffer,
                            text_slice_begin_buffer,
                            text_slice_size_buffer,
                            timestamp_slice_begin_buffer,
                            timestamp_slice_size_buffer,
                            nospeech_slice_begin_buffer,
                            nospeech_slice_size_buffer,
                            reduction_axis_buffer}));
    
    _builder.Finish(model, tflite::ModelIdentifier());

    // Wrap in FlatBufferModel object
    _model = tflite::FlatBufferModel::BuildFromBuffer(
                 reinterpret_cast<const char*>(_builder.GetBufferPointer()), _builder.GetSize());

    if (!_model) {
        std::cerr << "Failed to build TFLite model in memory!" << std::endl;
        return false;
    }

    // Build an interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder interpreter_builder(*_model, resolver);
    interpreter_builder(&_interpreter);

    if (!_interpreter) {
        std::cerr << "Failed to create TFLite interpreter!" << std::endl;
        return false;
    }

    // Allocate memory for tensors (see other default initializer method, we put here instead of initializeModelInMemory)
    if (_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
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
