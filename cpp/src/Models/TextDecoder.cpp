#include "TextDecoder.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "backend_class.hpp"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace WhisperKit;
// 'Monolithic KV Cache' ~ corresponds to the QUIC exported Whisper models

namespace WhisperKit {
constexpr const int kKvFactor = 2;
constexpr const int kLayersWhisperTiny = 4;
constexpr const int kLayersWhisperBase = 6;
constexpr const int kLayersWhisperSmall = 12;
constexpr const int kLayersWhisperMedium = 24;
constexpr const int kLayersWhisperLarge = 32;

constexpr const char* kVariantWhisperTiny = "tiny";
constexpr const char* kVariantWhisperBase = "base";
constexpr const char* kVariantWhisperSmall = "small";
constexpr const char* kVariantWhisperMedium = "medium";
constexpr const char* kVariantWhisperLarge = "large";
constexpr const char* kVariantNone = "none";
}  // namespace WhisperKit

namespace {

std::string normalize_name(const std::string& name) {
    // Names were padded with null characters to ensure alignment with original exported names
    // Remove extra null characters, or naive string matching will fail.
    auto name_copy = name.c_str();
    return std::string(name_copy);
};

}  // namespace

class FlatBuffersMetadata {
   public:
    FlatBuffersMetadata(const std::string& tflite_model_path) {
        _model_file_path = tflite_model_path;
        std::ifstream file(tflite_model_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Failed to open file");

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        _buffer = std::vector<char>(size);
        if (!file.read(_buffer.data(), size)) throw std::runtime_error("Failed to read file");

        const tflite::Model* model = tflite::GetModel(_buffer.data());

        if (!model) {
            throw std::runtime_error("Model is null");
        }
        if (!model->subgraphs()) {
            throw std::runtime_error("Model has no subgraphs");
        }
        _model = model;
        parse_model_metadata();
    }

    ~FlatBuffersMetadata() {
        _model = nullptr;
        _input_tensor_indices.clear();
        _output_tensor_indices.clear();
        _subgraphs = nullptr;
        _buffer.clear();
        _buffer.shrink_to_fit();
    }

    const std::string& get_model_file_path() const { return _model_file_path; }

    const std::unordered_map<std::string, std::pair<int, int>> get_input_tensor_indices(int subgraph_index = 0) const;
    const std::unordered_map<std::string, std::pair<int, int>> get_output_tensor_indices(int subgraph_index = 0) const;
    void print_metadata();
    const tflite::Model* get_model() const { return _model; }

   private:
    void parse_model_metadata();
    std::string _model_file_path;
    const tflite::Model* _model;
    std::vector<char> _buffer;
    const ::flatbuffers::Vector<::flatbuffers::Offset<tflite::SubGraph>>* _subgraphs;

    // name -> (tensor_index, io_index)
    std::vector<std::unordered_map<std::string, std::pair<int, int>>> _input_tensor_indices;
    std::vector<std::unordered_map<std::string, std::pair<int, int>>> _output_tensor_indices;
};

void FlatBuffersMetadata::parse_model_metadata() {
    _subgraphs = _model->subgraphs();

    // Primary subgraph is the only one we care about for now.
    bool only_first_subgraph = true;
    int max_subgraph_index = only_first_subgraph ? 1 : _subgraphs->size();

    _input_tensor_indices.resize(max_subgraph_index);
    _output_tensor_indices.resize(max_subgraph_index);

    for (int i = 0; i < max_subgraph_index; i++) {
        const tflite::SubGraph* subgraph = _subgraphs->Get(i);
        std::unordered_map<std::string, std::pair<int, int>> input_tensor_indices;
        std::unordered_map<std::string, std::pair<int, int>> output_tensor_indices;

        const auto* inputs = subgraph->inputs();
        const auto* outputs = subgraph->outputs();
        const auto* tensors = subgraph->tensors();

        for (int i = 0; i < inputs->size(); ++i) {
            int tensor_index = inputs->Get(i);
            auto name = normalize_name(tensors->Get(tensor_index)->name()->str());
            input_tensor_indices[name] = std::make_pair(tensor_index, i);
        }

        for (int i = 0; i < outputs->size(); ++i) {
            int tensor_index = outputs->Get(i);
            auto name = normalize_name(tensors->Get(tensor_index)->name()->str());
            output_tensor_indices[name] = std::make_pair(tensor_index, i);
        }
        _input_tensor_indices[i] = input_tensor_indices;
        _output_tensor_indices[i] = output_tensor_indices;
    }
}

void FlatBuffersMetadata::print_metadata() {
    std::cout << "Model file path: " << _model_file_path << std::endl;
    for (int i = 0; i < _input_tensor_indices.size(); i++) {
        std::cout << "Subgraph " << i << " input tensor indices:" << std::endl;
        for (const auto& [name, indices] : _input_tensor_indices[i]) {
            std::cout << "  " << name << ": (" << indices.first << ", " << indices.second << ")" << std::endl;
        }
    }
    for (int i = 0; i < _output_tensor_indices.size(); i++) {
        std::cout << "Subgraph " << i << " output tensor indices:" << std::endl;
        for (const auto& [name, indices] : _output_tensor_indices[i]) {
            std::cout << "  " << name << ": (" << indices.first << ", " << indices.second << ")" << std::endl;
        }
    }
}

const std::unordered_map<std::string, std::pair<int, int>> FlatBuffersMetadata::get_input_tensor_indices(
    int subgraph_index) const {
    if (subgraph_index >= _input_tensor_indices.size()) {
        throw std::runtime_error("Subgraph index out of bounds");
    }
    return _input_tensor_indices[subgraph_index];
}

const std::unordered_map<std::string, std::pair<int, int>> FlatBuffersMetadata::get_output_tensor_indices(
    int subgraph_index) const {
    if (subgraph_index >= _output_tensor_indices.size()) {
        throw std::runtime_error("Subgraph index out of bounds");
    }
    return _output_tensor_indices[subgraph_index];
}

// TODO: make a metadata class for the model signature and tensor indices,
// pass to the subclasses so we don't have to reopen the file outside of
// loading the actual tflite model.  TFLIte APIs require signature runner
// to get this information from interpreter, which is not available if the
// model doesn't have a signature.
bool is_exact_match_for_monolithic_kv_cache(const tflite::Model* model) {
    const std::unordered_set<std::string> expected_input_names = {
        "x", "index", "k_cache_cross", "v_cache_cross", "k_cache_self", "v_cache_self"};
    const std::unordered_set<std::string> expected_output_names = {"logits", "k_cache", "v_cache"};

    if (!model->subgraphs()) {
        throw std::runtime_error("Model has no subgraphs");
    }

    const auto* subgraphs = model->subgraphs();

    if (subgraphs->size() == 0) {
        throw std::runtime_error("Model has no subgraphs");
    }

    const auto* subgraph = subgraphs->Get(0);
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();

    if (inputs->size() != expected_input_names.size() || outputs->size() != expected_output_names.size()) {
        return false;
    }

    const auto* tensors = subgraph->tensors();
    std::unordered_set<std::string> input_names;
    for (int i = 0; i < inputs->size(); ++i) {
        int tensor_index = inputs->Get(i);
        auto name = tensors->Get(tensor_index)->name()->str();
        input_names.insert(name);
    }

    std::unordered_set<std::string> output_names;
    for (int i = 0; i < outputs->size(); ++i) {
        int tensor_index = outputs->Get(i);
        output_names.insert(tensors->Get(tensor_index)->name()->str());
    }

    if (input_names != expected_input_names || output_names != expected_output_names) {
        return false;
    }

    return true;
}

const int layers_for_variant(const std::string& variant) {
    if (variant == kVariantWhisperTiny) {
        return kLayersWhisperTiny;
    } else if (variant == kVariantWhisperBase) {
        return kLayersWhisperBase;
    } else if (variant == kVariantWhisperSmall) {
        return kLayersWhisperSmall;
    } else if (variant == kVariantWhisperMedium) {
        return kLayersWhisperMedium;
    } else if (variant == kVariantWhisperLarge) {
        return kLayersWhisperLarge;
    }
    return 0;
}

std::unordered_set<std::string> get_expected_input_names_for_variant(const char* variant) {
    auto input_names_for_variant_with_layers = [](const int num_layers) -> auto{
        std::unordered_set<std::string> input_names;
        input_names.insert(std::string("x"));
        input_names.insert(std::string("index"));
        input_names.insert(std::string("k_cache_cross"));
        input_names.insert(std::string("v_cache_cross"));
        for (int i = 0; i < num_layers; ++i) {
            input_names.insert(std::string("k_cache_self_" + std::to_string(i)));
            input_names.insert(std::string("v_cache_self_" + std::to_string(i)));
        }
        return input_names;
    };

    int num_layers = layers_for_variant(variant);

    return input_names_for_variant_with_layers(num_layers);
}

bool is_exact_match_for_separate_kv_cache_no_alignment_heads(const tflite::Model* model) {
    const auto* subgraph = model->subgraphs()->Get(0);
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();

    const auto& num_inputs = inputs->size();
    const auto& num_outputs = outputs->size();

    constexpr const int num_shared_inputs = 4;  // x, index, k_cache_cross, v_cache_cross
    constexpr const int num_layers_whisper_tiny = 4;
    constexpr const int kv_factor = 2;
    constexpr const int minimum_num_inputs = num_shared_inputs + num_layers_whisper_tiny * kv_factor;
    constexpr const int minimum_num_outputs = 2 * num_layers_whisper_tiny + 1;  // logits

    if (num_inputs < minimum_num_inputs || num_outputs < minimum_num_outputs) {
        return false;
    }

    /* helper functions to calculate the number of inputs and outputs for a given number of layers */
    auto calculate_num_inputs_for_variant_with_layers = [=](const int num_layers) -> auto{
        return num_shared_inputs + num_layers * kv_factor;
    };

    auto calculate_num_outputs_for_variant_with_layers = [=](const int num_layers) -> auto{
        return kv_factor * num_layers + 1;
    };

    auto output_names_for_variant_with_layers = [=](const int num_layers) -> auto{
        std::unordered_set<std::string> output_names;
        output_names.insert(std::string("logits"));
        for (int i = 0; i < num_layers; ++i) {
            output_names.insert(std::string("k_cache_" + std::to_string(i)));
            output_names.insert(std::string("v_cache_" + std::to_string(i)));
        }
        return output_names;
    };

    char* variant = const_cast<char*>(kVariantNone);

    if (num_inputs == get_expected_input_names_for_variant(kVariantWhisperTiny).size()) {
        variant = const_cast<char*>(kVariantWhisperTiny);
    } else if (num_inputs == get_expected_input_names_for_variant(kVariantWhisperBase).size()) {
        variant = const_cast<char*>(kVariantWhisperBase);
    } else if (num_inputs == get_expected_input_names_for_variant(kVariantWhisperSmall).size()) {
        variant = const_cast<char*>(kVariantWhisperSmall);
    } else if (num_inputs == get_expected_input_names_for_variant(kVariantWhisperMedium).size()) {
        variant = const_cast<char*>(kVariantWhisperMedium);
    } else if (num_inputs == get_expected_input_names_for_variant(kVariantWhisperLarge).size()) {
        variant = const_cast<char*>(kVariantWhisperLarge);
    }

    // no matches found for inputs
    if (variant == kVariantNone) {
        return false;
    }

    if (variant == kVariantWhisperTiny) {
        if (num_outputs != calculate_num_outputs_for_variant_with_layers(kLayersWhisperTiny)) {
            return false;
        }
    } else if (variant == kVariantWhisperBase) {
        if (num_outputs != calculate_num_outputs_for_variant_with_layers(kLayersWhisperBase)) {
            return false;
        }
    } else if (variant == kVariantWhisperSmall) {
        if (num_outputs != calculate_num_outputs_for_variant_with_layers(kLayersWhisperSmall)) {
            return false;
        }
    } else if (variant == kVariantWhisperMedium) {
        if (num_outputs != calculate_num_outputs_for_variant_with_layers(kLayersWhisperMedium)) {
            return false;
        }
    } else if (variant == kVariantWhisperLarge) {
        if (num_outputs != calculate_num_outputs_for_variant_with_layers(kLayersWhisperLarge)) {
            return false;
        }
    }

    auto expected_input_names = get_expected_input_names_for_variant(variant);
    auto expected_output_names = output_names_for_variant_with_layers(layers_for_variant(variant));

    std::unordered_set<std::string> input_names;
    std::unordered_set<std::string> output_names;
    const auto* tensors = subgraph->tensors();

    for (int i = 0; i < num_inputs; ++i) {
        auto name = normalize_name(tensors->Get(inputs->Get(i))->name()->str());
        input_names.insert(name);
    }

    for (int i = 0; i < num_outputs; ++i) {
        auto name = normalize_name(tensors->Get(outputs->Get(i))->name()->str());
        output_names.insert(name);
    }

    if (input_names != expected_input_names) {
        return false;
    }

    if (output_names != expected_output_names) {
        return false;
    }

    return true;
}

TextDecoder::~TextDecoder() {}

std::unique_ptr<TextDecoder> TextDecoderFactory::CreateFromFile(const std::string& tflite_model_path) {
    auto metadata = std::make_unique<FlatBuffersMetadata>(tflite_model_path);
    auto is_monolithic_kv_cache = is_exact_match_for_monolithic_kv_cache(metadata->get_model());
    if (is_monolithic_kv_cache) {
        return std::make_unique<MonolithicKVDecoder>(tflite_model_path);
    }

    auto is_separate_kv_cache_no_alignment_heads =
        is_exact_match_for_separate_kv_cache_no_alignment_heads(metadata->get_model());

    if (is_separate_kv_cache_no_alignment_heads) {
        return std::make_unique<PerLayerKVDecoder>(tflite_model_path);
    }

    throw std::runtime_error("Decoder model signature not recognized");
}

std::pair<char*, int> MonolithicKVDecoder::get_logits_tensor() { return decoder_outputs[0]; }

MonolithicKVDecoder::MonolithicKVDecoder(const std::string& tflite_model_path) {
    _model_path = tflite_model_path;
    metadata = std::make_unique<FlatBuffersMetadata>(tflite_model_path);
    // metadata->print_metadata();

    // Note that the decoder model is not initialized here, it is initialized in the initialize method
    _decoder_model = std::make_unique<MODEL_SUPER_CLASS>("TextDecoder");

    if (!_decoder_model) {
        throw std::runtime_error("Decoder model not initialized");
    }
}

MonolithicKVDecoder::~MonolithicKVDecoder() {
    _decoder_model.reset();
    metadata.reset();
}

bool MonolithicKVDecoder::initialize(std::string model_path, std::string lib_dir, std::string cache_dir, int backend,
                                     bool debug) {
    return _decoder_model->initialize(model_path, lib_dir, cache_dir, backend, debug);
}

void MonolithicKVDecoder::uninitialize() { _decoder_model->uninitialize(); }

void MonolithicKVDecoder::read_input_data(char* input_data, int idx) {
    _decoder_model->read_input_data(input_data, idx);
}

void MonolithicKVDecoder::bind_input_tensor(char* input_data, const std::string& tensor_name) {
    if (tensor_name == "x") {
        _decoder_model->read_input_data(input_data, 0);
        return;
    }

    if (tensor_name == "index") {
        _decoder_model->read_input_data(input_data, 1);
        return;
    }

    if (tensor_name == "k_cache_cross") {
        _decoder_model->read_input_data(input_data, 2);
        return;
    }

    if (tensor_name == "v_cache_cross") {
        _decoder_model->read_input_data(input_data, 3);
        return;
    }

    // self attention kv cache handled separately

    throw std::runtime_error("Invalid tensor name");
}

void MonolithicKVDecoder::invoke(bool measure_time) { _decoder_model->invoke(measure_time); }

void MonolithicKVDecoder::update_kv_cache() {
    if (decoder_outputs.empty()) {
        decoder_outputs = _decoder_model->get_output_ptrs();
    }
    // k_cache_self
    _decoder_model->read_input_data(decoder_outputs[1].first, 4);
    // v_cache_self
    _decoder_model->read_input_data(decoder_outputs[2].first, 5);
}

void MonolithicKVDecoder::initialize_kv_cache() {
    if (decoder_outputs.empty()) {
        decoder_outputs = _decoder_model->get_output_ptrs();
    }
    // first k_cache_self is all zeros
    memset(decoder_outputs[1].first, 0, decoder_outputs[1].second);
    // first v_cache_self is all zeros
    memset(decoder_outputs[2].first, 0, decoder_outputs[2].second);
}

float MonolithicKVDecoder::get_latency_median() { return _decoder_model->get_latency_median(); }

float MonolithicKVDecoder::get_latency_avg() { return _decoder_model->get_latency_avg(); }

std::unique_ptr<json> MonolithicKVDecoder::get_latency_json() { return _decoder_model->get_latency_json(); }

std::vector<std::pair<char*, int>> MonolithicKVDecoder::get_input_ptrs() { return _decoder_model->get_input_ptrs(); }

std::vector<std::pair<char*, int>> MonolithicKVDecoder::get_output_ptrs() { return _decoder_model->get_output_ptrs(); }

int MonolithicKVDecoder::get_inference_num() { return _decoder_model->get_inference_num(); }

float MonolithicKVDecoder::get_latency_sum() { return _decoder_model->get_latency_sum(); }
void MonolithicKVDecoder::dump_input_tensors() {
    // Not yet implemented
}

void MonolithicKVDecoder::dump_output_tensors() {
    // Not yet implemented
}

std::pair<char*, int> PerLayerKVDecoder::get_logits_tensor() {
    if (decoder_outputs.empty()) {
        decoder_outputs = _decoder_model->get_output_ptrs();
    }
    auto logits_index = output_tensor_indices.at("logits");
    return decoder_outputs[logits_index];
}

PerLayerKVDecoder::PerLayerKVDecoder(const std::string& tflite_model_path) {
    _model_path = tflite_model_path;
    metadata = std::make_unique<FlatBuffersMetadata>(tflite_model_path);
    initialize_io_metadata();
    metadata.reset();  // to close the .tflite file

    // Note that the decoder model is not initialized here, it is initialized in the initialize method
    _decoder_model = std::make_unique<MODEL_SUPER_CLASS>("TextDecoder");
    if (!_decoder_model) {
        throw std::runtime_error("Decoder model not initialized");
    }
}

PerLayerKVDecoder::~PerLayerKVDecoder() {
    _decoder_model.reset();
    metadata.reset();
}

bool PerLayerKVDecoder::initialize(std::string model_path, std::string lib_dir, std::string cache_dir, int backend,
                                   bool debug) {
    return _decoder_model->initialize(model_path, lib_dir, cache_dir, backend, debug);
}

void PerLayerKVDecoder::uninitialize() { _decoder_model->uninitialize(); }

void PerLayerKVDecoder::read_input_data(char* input_data, int idx) { _decoder_model->read_input_data(input_data, idx); }

void PerLayerKVDecoder::bind_input_tensor(char* input_data, const std::string& tensor_name) {
    if (tensor_name == "x") {
        // get value in int32 and upcast to int64 before passing to read_input_data, which
        // does a simple memcpy of all the bytes.
        int32_t* input_data_int32 = reinterpret_cast<int32_t*>(input_data);
        int64_t x = static_cast<int64_t>(*input_data_int32);
        auto x_tensor_index = input_tensor_indices["x"];
        _decoder_model->read_input_data(reinterpret_cast<char*>(&x), x_tensor_index);
        return;
    }

    if (tensor_name == "index") {
        int* input_data_int32 = reinterpret_cast<int*>(input_data);
        int64_t index = static_cast<int64_t>(*input_data_int32);
        auto index_tensor_index = input_tensor_indices["index"];

        _decoder_model->read_input_data(reinterpret_cast<char*>(&index), index_tensor_index);
        return;
    }

    if (tensor_name == "k_cache_cross") {
        auto k_cache_cross_tensor_index = input_tensor_indices["k_cache_cross"];
        _decoder_model->read_input_data(input_data, k_cache_cross_tensor_index);
        return;
    }

    if (tensor_name == "v_cache_cross") {
        auto v_cache_cross_tensor_index = input_tensor_indices["v_cache_cross"];
        _decoder_model->read_input_data(input_data, v_cache_cross_tensor_index);
        return;
    }

    else {
        auto tensor_index = kv_cache_input_tensor_indices[tensor_name];
        _decoder_model->read_input_data(input_data, tensor_index);
        return;
    }

    throw std::runtime_error("Invalid tensor name");
}

void PerLayerKVDecoder::invoke(bool measure_time) { _decoder_model->invoke(measure_time); }

template <typename T>
void save_to_binary_file(const std::string& filename, const std::vector<T>& data) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    printf("Saving to binary file: %s\n", filename.c_str());
    outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    outfile.close();
}

void PerLayerKVDecoder::dump_input_tensors() {
    printf("Dumping input tensors\n");
    auto input_ptrs = _decoder_model->get_input_ptrs();

    for (int index = 0; index < _decoder_model->_interpreter->inputs().size(); index++) {
        auto name = _decoder_model->_interpreter->GetInputName(index);
        auto safe_name = normalize_name(name);
        printf("safe name %s index %d\n", safe_name.c_str(), index);
        if (safe_name == "x" || safe_name == "index") {
            printf("Dumping input tensor: %s, index=%d\n", name, index);
            auto tensor_ptr = input_ptrs[index];
            int tensor_size = tensor_ptr.second;
            printf("Tensor size: %d\n", tensor_size);
            int64_t* input_data = reinterpret_cast<int64_t*>(tensor_ptr.first);
            size_t num_elements = tensor_size / sizeof(int64_t);
            std::vector<int64_t> data(num_elements);
            printf("==============================================\n");
            printf("name %s index %d num_elements: %ld\n", name, index, num_elements);
            for (int i = 0; i < num_elements; i++) {
                data[i] = input_data[i];
                printf("name %s index %d data[%d] = %ld\n", name, index, i, data[i]);
            }
            printf("==============================================\n");

            std::string _name = std::string(safe_name);
            std::string filename = "/src/AXIE/debug_inputs/input_" + _name + ".bin";
            save_to_binary_file<int64_t>(filename, data);  // float32
        } else {
            std::string _name = std::string(safe_name);

            printf("In float tensor path");
            printf("Dumping input tensor: %s, index=%d\n", _name.c_str(), index);
            auto tensor_ptr = input_ptrs[index];
            auto tensor_size = tensor_ptr.second;
            printf("Tensor size: %d\n", tensor_size);
            std::vector<float> data(tensor_size / sizeof(float));

            bool print_for_debug = true;
            if (data.size() > 100) {
                print_for_debug = false;
            }

            for (int i = 0; i < data.size(); i++) {
                data[i] = *reinterpret_cast<float*>(input_ptrs[index].first + i * sizeof(float));
                if (print_for_debug) {
                    printf("name %s index %d data[%d] = %f\n", _name.c_str(), index, i, data[i]);
                }
            }
            std::string filename = "/src/AXIE/debug_inputs/input_" + _name + ".bin";
            save_to_binary_file<float>(filename, data);
        }
    }
}

void PerLayerKVDecoder::dump_output_tensors() {
    printf("Dumping output tensors\n");
    auto output_ptrs = _decoder_model->get_output_ptrs();

    for (int index = 0; index < _decoder_model->_interpreter->outputs().size(); index++) {
        auto name = _decoder_model->_interpreter->GetOutputName(index);
        auto safe_name = normalize_name(name);
        printf("safe name %s index %d\n", safe_name.c_str(), index);

        if (safe_name == "logits") {
            std::string _name = std::string(safe_name);

            printf("In float tensor path");
            printf("Dumping output tensor: %s, index=%d\n", _name.c_str(), index);
            auto tensor_ptr = output_ptrs[index];
            auto tensor_size = tensor_ptr.second;
            printf("Tensor size: %d\n", tensor_size);
            std::vector<float> data(tensor_size / sizeof(float));

            bool print_for_debug = true;
            if (data.size() > 100) {
                print_for_debug = false;
            }
            for (int i = 0; i < data.size(); i++) {
                data[i] = *reinterpret_cast<float*>(output_ptrs[index].first + i * sizeof(float));
                if (print_for_debug) {
                    printf("name %s index %d data[%d] = %f\n", _name.c_str(), index, i, data[i]);
                }
            }
            std::string filename = "/src/AXIE/debug_inputs/output_" + _name + ".bin";
            save_to_binary_file<float>(filename, data);
        }
    }
}

void PerLayerKVDecoder::initialize_io_metadata() {
    // self attention kv cache tensors
    const auto& all_input_tensor_indices = metadata->get_input_tensor_indices(0);
    const auto& all_output_tensor_indices = metadata->get_output_tensor_indices(0);

    // store only the relative indices within the i/o tensor vectors
    auto logits_indices = all_output_tensor_indices.at("logits");
    output_tensor_indices["logits"] = logits_indices.second;

    auto token_indices = all_input_tensor_indices.at("x");
    input_tensor_indices["x"] = token_indices.second;

    auto index_indices = all_input_tensor_indices.at("index");
    input_tensor_indices["index"] = index_indices.second;

    auto k_cache_cross_indices = all_input_tensor_indices.at("k_cache_cross");
    input_tensor_indices["k_cache_cross"] = k_cache_cross_indices.second;

    auto v_cache_cross_indices = all_input_tensor_indices.at("v_cache_cross");
    input_tensor_indices["v_cache_cross"] = v_cache_cross_indices.second;

    for (const auto& [name, indices] : all_input_tensor_indices) {
        if (name == "x" || name == "index" || name == "k_cache_cross" || name == "v_cache_cross") {
            continue;
        }
        kv_cache_input_tensor_indices[name] = indices.second;
    }

    for (const auto& [name, indices] : all_output_tensor_indices) {
        if (name == "logits") {
            continue;
        }

        if (name.find("k_cache_") == 0 || name.find("v_cache_") == 0) {
            kv_cache_output_tensor_indices[name] = indices.second;
        }
    }

    auto extractNumericSuffix = [](const std::string& s) -> int {
        size_t pos = s.find_last_of('_');
        if (pos == std::string::npos || pos == s.length() - 1) {
            throw std::invalid_argument("String does not contain a valid numeric suffix");
        }
        return std::stoi(s.substr(pos + 1));
    };

    for (const auto& [name, index] : kv_cache_output_tensor_indices) {
        if (name.find("k_cache_") == 0) {
            auto layer_num = extractNumericSuffix(name);
            auto input_name = "k_cache_self_" + std::to_string(layer_num);

            kv_cache_io_tensor_names[input_name] = name;
        } else if (name.find("v_cache_") == 0) {
            auto layer_num = extractNumericSuffix(name);
            auto input_name = "v_cache_self_" + std::to_string(layer_num);

            kv_cache_io_tensor_names[input_name] = name;
        }
    }
}

void PerLayerKVDecoder::update_kv_cache() {
    if (decoder_outputs.empty()) {
        decoder_outputs = _decoder_model->get_output_ptrs();
    }

    for (const auto& [input_name, output_name] : kv_cache_io_tensor_names) {
        auto input_tensor_index = kv_cache_input_tensor_indices[input_name];
        auto output_tensor_index = kv_cache_output_tensor_indices[output_name];
        _decoder_model->read_input_data(decoder_outputs[output_tensor_index].first, input_tensor_index);
    }
}

void PerLayerKVDecoder::initialize_kv_cache() {
    if (decoder_outputs.empty()) {
        decoder_outputs = _decoder_model->get_output_ptrs();
    }

    auto input_ptrs = _decoder_model->get_input_ptrs();

    for (const auto& [name, index] : kv_cache_input_tensor_indices) {
        memset(input_ptrs[index].first, 0, input_ptrs[index].second);
    }

    for (const auto& [name, index] : kv_cache_output_tensor_indices) {
        memset(decoder_outputs[index].first, 0, decoder_outputs[index].second);
    }
}

float PerLayerKVDecoder::get_latency_median() { return _decoder_model->get_latency_median(); }

float PerLayerKVDecoder::get_latency_avg() { return _decoder_model->get_latency_avg(); }

std::unique_ptr<json> PerLayerKVDecoder::get_latency_json() { return _decoder_model->get_latency_json(); }

std::vector<std::pair<char*, int>> PerLayerKVDecoder::get_input_ptrs() { return _decoder_model->get_input_ptrs(); }

std::vector<std::pair<char*, int>> PerLayerKVDecoder::get_output_ptrs() { return _decoder_model->get_output_ptrs(); }

int PerLayerKVDecoder::get_inference_num() { return _decoder_model->get_inference_num(); }

float PerLayerKVDecoder::get_latency_sum() { return _decoder_model->get_latency_sum(); }
