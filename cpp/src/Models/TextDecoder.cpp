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

// TODO: make a metadata class for the model signature and tendor indices,
// pass to the subclasses so we don't have to reopen the file outside of
// loading the actual tflite model.  TFLIte APIs require signature runner
// to get this information from interpreter, which is not available if the
// model doesn't have a signature.
bool is_exact_match_for_monolithic_kv_cache(const tflite::Model* model) {
    const std::unordered_set<std::string> expected_input_names = {
        "x", "index", "k_cache_cross", "v_cache_cross", "k_cache_self", "v_cache_self"};
    const std::unordered_set<std::string> expected_output_names = {"logits", "k_cache", "v_cache"};

    const auto* subgraph = model->subgraphs()->Get(0);
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();

    if (inputs->size() != expected_input_names.size() || outputs->size() != expected_output_names.size()) {
        return false;
    }

    const auto* tensors = subgraph->tensors();

    std::unordered_set<std::string> input_names;
    for (int i = 0; i < inputs->size(); ++i) {
        int tensor_index = inputs->Get(i);
        input_names.insert(tensors->Get(tensor_index)->name()->str());
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
    auto calculate_num_inputs_for_variant_with_layers = [=](const int num_layers) -> auto {
        return num_shared_inputs + num_layers * kv_factor;
    };

    auto calculate_num_outputs_for_variant_with_layers = [=](const int num_layers) -> auto {
        return kv_factor * num_layers + 1;
    };

    auto output_names_for_variant_with_layers = [=](const int num_layers) -> auto {
        std::unordered_set<std::string> output_names;
        output_names.insert(std::string("logits"));
        for (int i = 0; i < num_layers; ++i) {
            output_names.insert(std::string("k_cache_" + std::to_string(i)));
            output_names.insert(std::string("v_cache_" + std::to_string(i)));
        }
        return output_names;
    };

    auto input_names_for_variant_with_layers = [](const int num_layers) -> auto {
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

    char* variant = const_cast<char*>(kVariantNone);

    if (num_inputs == calculate_num_inputs_for_variant_with_layers(kLayersWhisperTiny)) {
        variant = const_cast<char*>(kVariantWhisperTiny);
    } else if (num_inputs == calculate_num_inputs_for_variant_with_layers(kLayersWhisperBase)) {
        variant = const_cast<char*>(kVariantWhisperBase);
    } else if (num_inputs == calculate_num_inputs_for_variant_with_layers(kLayersWhisperSmall)) {
        variant = const_cast<char*>(kVariantWhisperSmall);
    } else if (num_inputs == calculate_num_inputs_for_variant_with_layers(kLayersWhisperMedium)) {
        variant = const_cast<char*>(kVariantWhisperMedium);
    } else if (num_inputs == calculate_num_inputs_for_variant_with_layers(kLayersWhisperLarge)) {
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

    auto expected_input_names = input_names_for_variant_with_layers(layers_for_variant(variant));
    auto expected_output_names = output_names_for_variant_with_layers(layers_for_variant(variant));

    std::unordered_set<std::string> input_names;
    std::unordered_set<std::string> output_names;
    const auto* tensors = subgraph->tensors();

    auto normalize_name = [](const std::string& name) -> std::string {
        // Names were padded with null characters to ensure alignment with original exported names
        // Remove extra null characters, or naive string matching will fail.
        auto name_copy = name.c_str();
        return std::string(name_copy);
    };

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

std::unique_ptr<TextDecoder> TextDecoderFactory::CreateFromFile(const std::string& tflite_model_path) {
    std::ifstream file(tflite_model_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open file");

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read file");

    const tflite::Model* model = tflite::GetModel(buffer.data());

    if (!model) throw std::runtime_error("Failed to load model");

    auto is_monolithic_kv_cache = is_exact_match_for_monolithic_kv_cache(model);

    if (is_monolithic_kv_cache) {
        return std::make_unique<MonolithicKVDecoder>(tflite_model_path);
    }

    auto is_separate_kv_cache_no_alignment_heads = is_exact_match_for_separate_kv_cache_no_alignment_heads(model);

    throw std::runtime_error("Decoder model signature not recognized");
}

std::pair<char*, int> MonolithicKVDecoder::get_logits_tensor() { return decoder_outputs[0]; }

MonolithicKVDecoder::MonolithicKVDecoder(const std::string& tflite_model_path) {
    _model_path = tflite_model_path;

    // Note that the decoder model is not initialized here, it is initialized in the initialize method
    _decoder_model = std::make_unique<MODEL_SUPER_CLASS>("TextDecoder");
    if (!_decoder_model) {
        throw std::runtime_error("Decoder model not initialized");
    }
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
