#pragma once

#include <string>

#include "WhisperKit.h"

struct whisperkit_configuration_t {
   public:
    whisperkit_configuration_t();

    void set_audio_encoder(const char* audio_encoder) noexcept;
    void set_text_decoder(const char* text_decoder) noexcept;
    void set_tokenizer(const char* tokenizer) noexcept;
    void set_melspectrogram_model(const char* melspectrogram_model) noexcept;
    void set_lib_dir(const char* lib_dir) noexcept;
    void set_cache_dir(const char* cache_dir) noexcept;
    void set_verbose(bool verbose) noexcept;
    void set_log_level(int log_level) noexcept;
    void set_prewarm(bool prewarm) noexcept;
    void set_load(bool load) noexcept;
    void set_model_path(const char* model_path) noexcept;
    void set_report_path(const char* report_path) noexcept;
    void set_backends(whisperkit_backend_t encoder_backend, whisperkit_backend_t decoder_backend) noexcept;

    const std::string get_audio_encoder() const noexcept;
    const std::string get_text_decoder() const noexcept;
    const std::string get_tokenizer() const noexcept;
    const std::string get_melspectrogram_model() const noexcept;
    const std::string get_lib_dir() const noexcept;
    const std::string get_cache_dir() const noexcept;
    const std::string get_model_path() const noexcept;
    const std::string get_report_path() const noexcept;
    int get_encoder_backend() const noexcept;
    int get_decoder_backend() const noexcept;
    bool get_verbose() const noexcept;
    int get_log_level() const noexcept;
    bool get_prewarm() const noexcept;
    bool get_load() const noexcept;

    whisperkit_pipeline_t* get_pipeline() const noexcept;

   private:
    std::string audio_encoder;
    std::string text_decoder;
    std::string tokenizer;
    std::string melspectrogram_model;
    std::string model_path;
    std::string report_path;
    std::string lib_dir;
    std::string cache_dir;
    whisperkit_pipeline_t* pipeline;
    int encoder_backend;
    int decoder_backend;

    bool verbose;
    int log_level;
    bool prewarm;
    bool load;
};
