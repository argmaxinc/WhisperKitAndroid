
#include "WhisperKitConfiguration.h"
#include "WhisperKitPipeline.h"

 whisperkit_configuration_t::whisperkit_configuration_t() {};

void whisperkit_configuration_t::set_audio_encoder(const char* audio_encoder) {
        this->audio_encoder = audio_encoder;
}

void whisperkit_configuration_t::set_text_decoder(const char* text_decoder) {
        this->text_decoder = text_decoder;
}

void whisperkit_configuration_t::set_voice_activity_detector(const char* voice_activity_detector) {
        this->voice_activity_detector = voice_activity_detector;
}

void whisperkit_configuration_t::set_tokenizer(const char* tokenizer) {
        this->tokenizer = tokenizer;
}

void whisperkit_configuration_t::set_melspectrogram_model(const char* melspectrogram_model) {
        this->melspectrogram_model = melspectrogram_model;
}

void whisperkit_configuration_t::set_postproc(const char* postproc) {
        this->postproc = postproc;
}

void whisperkit_configuration_t::set_lib_dir(const char* lib_dir) {
        this->lib_dir = lib_dir;
}

void whisperkit_configuration_t::set_cache_dir(const char* cache_dir) {
        this->cache_dir = cache_dir;
}

void whisperkit_configuration_t::set_verbose(bool verbose) {
        this->verbose = verbose;
}

void whisperkit_configuration_t::set_log_level(int log_level) {
        this->log_level = log_level;
}

void whisperkit_configuration_t::set_prewarm(bool prewarm) {
        this->prewarm = prewarm;
}

void whisperkit_configuration_t::set_load(bool load) {
        this->load = load;
}

const std::string whisperkit_configuration_t::get_audio_encoder() const {
        return this->audio_encoder;
}
const std::string whisperkit_configuration_t::get_text_decoder() const {
        return this->text_decoder;
}
const std::string whisperkit_configuration_t::get_voice_activity_detector() const {
        return this->voice_activity_detector;
}
const std::string whisperkit_configuration_t::get_tokenizer() const {
        return this->tokenizer;
}
const std::string whisperkit_configuration_t::get_melspectrogram_model() const {
        return this->melspectrogram_model;
}

const std::string whisperkit_configuration_t::get_postproc() const {
        return this->postproc;
}

const std::string whisperkit_configuration_t::get_lib_dir() const {
        return this->lib_dir;
}

const std::string whisperkit_configuration_t::get_cache_dir() const {
        return this->cache_dir;
}

bool whisperkit_configuration_t::get_verbose() const {
        return this->verbose;
}

int whisperkit_configuration_t::get_log_level() const {
        return this->log_level;
}

bool whisperkit_configuration_t::get_prewarm() const {
        return this->prewarm;
}

bool whisperkit_configuration_t::get_load() const {
        return this->load;
}

whisperkit_pipeline_t* whisperkit_configuration_t::get_pipeline() const {
        return this->pipeline;
}