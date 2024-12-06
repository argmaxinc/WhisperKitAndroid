#pragma once

#include <string>

struct whisperkit_configuration_t {

    public:
        whisperkit_configuration_t();
        whisperkit_configuration_t(const whisperkit_configuration_t&) = delete;
        whisperkit_configuration_t& operator=(const whisperkit_configuration_t&) = delete;

        void set_audio_encoder(const char* audio_encoder) noexcept;
        void set_text_decoder(const char* text_decoder) noexcept;
        void set_voice_activity_detector(const char* voice_activity_detector) noexcept;
        void set_tokenizer(const char* tokenizer) noexcept;
        void set_melspectrogram_model(const char* melspectrogram_model) noexcept;
        void set_postproc(const char* postproc) noexcept;
        void set_lib_dir(const char* lib_dir) noexcept;
        void set_cache_dir(const char* cache_dir) noexcept;
        void set_verbose(bool verbose) noexcept;
        void set_log_level(int log_level) noexcept;
        void set_prewarm(bool prewarm) noexcept;
        void set_load(bool load) noexcept;

        const std::string& get_audio_encoder() const noexcept;
        const std::string& get_text_decoder() const noexcept;
        const std::string& get_voice_activity_detector() const noexcept;
        const std::string& get_tokenizer() const noexcept;
        const std::string& get_melspectrogram_model() const noexcept;
        const std::string& get_postproc() const noexcept;
        const std::string& get_lib_dir() const noexcept;
        const std::string& get_cache_dir() const noexcept;
        bool get_verbose() const noexcept;
        int get_log_level() const noexcept;
        bool get_prewarm() const noexcept;
        bool get_load() const noexcept;
        whisperkit_pipeline_t* get_pipeline() const noexcept;
        
    private:

        std::string audio_encoder;
        std::string text_decoder;
        std::string voice_activity_detector;
        std::string tokenizer;
        std::string melspectrogram_model;
        std::string postproc;
        std::string lib_dir;
        std::string cache_dir;
        whisperkit_pipeline_t* pipeline;
        bool verbose;
        int log_level;
        bool prewarm;
        bool load;
};