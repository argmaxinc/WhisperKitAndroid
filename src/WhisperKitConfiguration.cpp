
#include "WhisperKitConfiguration.h"

        void set_audio_encoder(const char* audio_encoder);
        void set_text_decoder(const char* text_decoder);
        void set_voice_activity_detector(const char* voice_activity_detector);
        void set_tokenizer(const char* tokenizer);
        void set_melspectrogram_model(const char* melspectrogram_model);
        void set_postproc(const char* postproc);
        void set_lib_dir(const char* lib_dir);
        void set_cache_dir(const char* cache_dir);
        void set_verbose(bool verbose);
        void set_log_level(int log_level);
        void set_prewarm(bool prewarm);
        void set_load(bool load);

        const std::string& get_audio_encoder() const;
        const std::string& get_text_decoder() const;
        const std::string& get_voice_activity_detector() const;
        const std::string& get_tokenizer() const;
        const std::string& get_melspectrogram_model() const;
        const std::string& get_postproc() const;
        const std::string& get_lib_dir() const;
        const std::string& get_cache_dir() const;
        bool get_verbose() const;
        int get_log_level() const;
        bool get_prewarm() const;
        bool get_load() const;
        