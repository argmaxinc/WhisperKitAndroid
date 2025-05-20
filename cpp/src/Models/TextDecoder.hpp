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

// TODO:
// remove extraneous functions used for passthrough to MODEL_SUPER_CLASS
// to expedite integration
class TextDecoder {
   public:
    virtual ~TextDecoder() = default;
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

   protected:
    // TODO: modify to hold tflite model from tensorflow & use delegate manager
    std::unique_ptr<MODEL_SUPER_CLASS> _decoder_model;
    std::string _model_path;
};

class MonolithicKVDecoder : public TextDecoder {
   public:
    MonolithicKVDecoder(const std::string& tflite_model_path);
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

   private:
    std::vector<std::pair<char*, int>> decoder_outputs;
};

class TextDecoderFactory {
   public:
    static std::unique_ptr<TextDecoder> CreateFromFile(const std::string& tflite_model_path);
};
