//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "backend_class.hpp"

namespace WhisperKit {
enum DecoderKVCacheType {
    DecoderKVCacheTypeMonolothic = 0,
    DecoderKVCacheTypeSeparate = 1,
};
}

class FlatBuffersMetadata;

// TODO:
// remove extraneous functions used for passthrough to MODEL_SUPER_CLASS
// to expedite integration
class TextDecoder {
   public:
    virtual ~TextDecoder();
    virtual void initialize_kv_cache() = 0;
    virtual void read_input_data(char* data, int index) = 0;

    virtual bool initialize(std::string model_path, std::string lib_dir, std::string cache_dir, int backend,
                            bool debug) = 0;
    virtual void uninitialize() = 0;
    virtual void invoke(bool measure_time = false) = 0;
    virtual void update_kv_cache() = 0;
    virtual std::vector<std::pair<char*, int>> get_input_ptrs() = 0;
    virtual std::vector<std::pair<char*, int>> get_output_ptrs() = 0;
    virtual void bind_input_tensor(char* input_data, const std::string& tensor_name) = 0;
    virtual std::pair<char*, int> get_logits_tensor() = 0;
    virtual int get_inference_num() = 0;
    virtual float get_latency_sum() = 0;
    virtual float get_latency_avg() = 0;
    virtual float get_latency_median() = 0;
    virtual std::unique_ptr<json> get_latency_json() = 0;

    virtual void dump_input_tensors() = 0;
    virtual void dump_output_tensors() = 0;

   protected:
    std::unique_ptr<FlatBuffersMetadata> metadata;
    // TODO: modify to hold tflite model from tensorflow & use delegate manager
    std::unique_ptr<MODEL_SUPER_CLASS> _decoder_model;
    std::string _model_path;
    std::vector<std::pair<char*, int>> decoder_outputs;
};

class MonolithicKVDecoder : public TextDecoder {
   public:
    MonolithicKVDecoder(const std::string& tflite_model_path);
    ~MonolithicKVDecoder();
    void initialize_kv_cache() override;

    bool initialize(std::string model_path, std::string lib_dir, std::string cache_dir, int backend,
                    bool debug) override;
    void uninitialize() override;
    void read_input_data(char* input_data, int idx) override;
    void invoke(bool measure_time = false) override;
    void update_kv_cache() override;
    std::vector<std::pair<char*, int>> get_input_ptrs() override;
    std::vector<std::pair<char*, int>> get_output_ptrs() override;
    void bind_input_tensor(char* input_data, const std::string& tensor_name) override;
    std::pair<char*, int> get_logits_tensor() override;
    int get_inference_num() override;
    float get_latency_sum() override;
    float get_latency_avg() override;
    float get_latency_median() override;
    std::unique_ptr<json> get_latency_json() override;

    void dump_input_tensors() override;
    void dump_output_tensors() override;
};

class PerLayerKVDecoder : public TextDecoder {
   public:
    PerLayerKVDecoder(const std::string& tflite_model_path);
    ~PerLayerKVDecoder();
    void initialize_kv_cache() override;

    bool initialize(std::string model_path, std::string lib_dir, std::string cache_dir, int backend,
                    bool debug) override;
    void uninitialize() override;
    void read_input_data(char* input_data, int idx) override;
    void invoke(bool measure_time = false) override;
    void update_kv_cache() override;
    std::vector<std::pair<char*, int>> get_input_ptrs() override;
    std::vector<std::pair<char*, int>> get_output_ptrs() override;
    void bind_input_tensor(char* input_data, const std::string& tensor_name) override;
    std::pair<char*, int> get_logits_tensor() override;
    int get_inference_num() override;
    float get_latency_sum() override;
    float get_latency_avg() override;
    float get_latency_median() override;
    std::unique_ptr<json> get_latency_json() override;

    void dump_input_tensors() override;
    void dump_output_tensors() override;

   private:
    void initialize_io_metadata();
    // self attention kv cache tensors
    std::unordered_map<std::string, std::string> kv_cache_io_tensor_names;  // <input_tensor_name, output_tensor_name>
    std::unordered_map<std::string, int> kv_cache_input_tensor_indices;     // <input_tensor_name, input_tensor_index>
    std::unordered_map<std::string, int> kv_cache_output_tensor_indices;    // <output_tensor_name, output_tensor_index>

    // non-kv cache tensors
    std::unordered_map<std::string, int>
        input_tensor_indices;  // <input_tensor_name, input_tensor_index>, non-kv cache tensors
    std::unordered_map<std::string, int>
        output_tensor_indices;  // <output_tensor_name, output_tensor_index>, non-kv cache tensors
};

class TextDecoderFactory {
   public:
    static std::unique_ptr<TextDecoder> CreateFromFile(const std::string& tflite_model_path);
};
