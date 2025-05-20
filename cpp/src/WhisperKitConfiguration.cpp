
#include "WhisperKitConfiguration.hpp"

#include "WhisperKitPipeline.hpp"
#include "backend_class.hpp"

whisperkit_configuration_t::whisperkit_configuration_t() {};

void whisperkit_configuration_t::set_audio_encoder(const char* audio_encoder) noexcept {
    this->audio_encoder = audio_encoder;
}

void whisperkit_configuration_t::set_text_decoder(const char* text_decoder) noexcept {
    this->text_decoder = text_decoder;
}

void whisperkit_configuration_t::set_tokenizer(const char* tokenizer) noexcept { this->tokenizer = tokenizer; }

void whisperkit_configuration_t::set_melspectrogram_model(const char* melspectrogram_model) noexcept {
    this->melspectrogram_model = melspectrogram_model;
}

void whisperkit_configuration_t::set_model_path(const char* model_path) noexcept { this->model_path = model_path; }

void whisperkit_configuration_t::set_report_path(const char* report_path) noexcept { this->report_path = report_path; }

void whisperkit_configuration_t::set_backends(whisperkit_backend_t encoder_backend,
                                              whisperkit_backend_t decoder_backend) noexcept {
    if (encoder_backend == WHISPERKIT_COMPUTE_BACKEND_CPU) {
        this->encoder_backend = ComputeBackend::CPU;
    } else if (encoder_backend == WHISPERKIT_COMPUTE_BACKEND_GPU) {
        this->encoder_backend = ComputeBackend::GPU;
    } else if (encoder_backend == WHISPERKIT_COMPUTE_BACKEND_NPU) {
        this->encoder_backend = ComputeBackend::NPU;
    } else {
        this->encoder_backend = ComputeBackend::GPU;
    }

    if (decoder_backend == WHISPERKIT_COMPUTE_BACKEND_CPU) {
        this->decoder_backend = ComputeBackend::CPU;
    } else if (decoder_backend == WHISPERKIT_COMPUTE_BACKEND_GPU) {
        this->decoder_backend = ComputeBackend::GPU;
    } else if (decoder_backend == WHISPERKIT_COMPUTE_BACKEND_NPU) {
        this->decoder_backend = ComputeBackend::NPU;
    } else {
        this->decoder_backend = ComputeBackend::GPU;
    }
}

void whisperkit_configuration_t::set_lib_dir(const char* lib_dir) noexcept { this->lib_dir = lib_dir; }

void whisperkit_configuration_t::set_cache_dir(const char* cache_dir) noexcept { this->cache_dir = cache_dir; }

void whisperkit_configuration_t::set_verbose(bool verbose) noexcept { this->verbose = verbose; }

void whisperkit_configuration_t::set_log_level(int log_level) noexcept { this->log_level = log_level; }

void whisperkit_configuration_t::set_prewarm(bool prewarm) noexcept { this->prewarm = prewarm; }

void whisperkit_configuration_t::set_load(bool load) noexcept { this->load = load; }

const std::string whisperkit_configuration_t::get_audio_encoder() const noexcept { return this->audio_encoder; }
const std::string whisperkit_configuration_t::get_text_decoder() const noexcept { return this->text_decoder; }
const std::string whisperkit_configuration_t::get_tokenizer() const noexcept { return this->tokenizer; }
const std::string whisperkit_configuration_t::get_melspectrogram_model() const noexcept {
    return this->melspectrogram_model;
}

const std::string whisperkit_configuration_t::get_model_path() const noexcept { return this->model_path; }

const std::string whisperkit_configuration_t::get_report_path() const noexcept { return this->report_path; }

const std::string whisperkit_configuration_t::get_lib_dir() const noexcept { return this->lib_dir; }

const std::string whisperkit_configuration_t::get_cache_dir() const noexcept { return this->cache_dir; }

bool whisperkit_configuration_t::get_verbose() const noexcept { return this->verbose; }

int whisperkit_configuration_t::get_log_level() const noexcept { return this->log_level; }

bool whisperkit_configuration_t::get_prewarm() const noexcept { return this->prewarm; }

bool whisperkit_configuration_t::get_load() const noexcept { return this->load; }

int whisperkit_configuration_t::get_encoder_backend() const noexcept { return this->encoder_backend; }

int whisperkit_configuration_t::get_decoder_backend() const noexcept { return this->decoder_backend; }

whisperkit_pipeline_t* whisperkit_configuration_t::get_pipeline() const noexcept { return this->pipeline; }
