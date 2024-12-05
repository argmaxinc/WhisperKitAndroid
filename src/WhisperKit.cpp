#include "WhisperKit.h"
#include "WhisperKitConfiguration.h"
#include "WhisperKitPipeline.h"

#pragma mark - initializers
whisperkit_status_t whisperkit_create_configuration(whisperkit_configuration_t **configuration) {
    if(configuration == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    *configuration = new whisperkit_configuration_t();
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_create_pipeline(whisperkit_pipeline_t **pipeline) {
    if(pipeline == nullptr || configuration == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    *pipeline = new whisperkit_pipeline_t();
    return WHISPERKIT_STATUS_SUCCESS;
};

#pragma mark - configuration setters
whisperkit_status_t whisperkit_configuration_set_audio_model(whisperkit_configuration_t *config, const char* audio_model) {
    if(config == nullptr || audio_model == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_audio_model(audio_model);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_audio_encoder(whisperkit_configuration_t *config, const char* audio_encoder) {
    if(config == nullptr || audio_encoder == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_audio_encoder(audio_encoder);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_text_decoder(whisperkit_configuration_t *config, const char* text_decoder) {
    if(config == nullptr || text_decoder == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_text_decoder(text_decoder);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_voice_activity_detector(whisperkit_configuration_t *config, const char* voice_activity_detector) {
    if(config == nullptr || voice_activity_detector == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_voice_activity_detector(voice_activity_detector);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_tokenizer(whisperkit_configuration_t *config, const char* tokenizer) {
    if(config == nullptr || tokenizer == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_tokenizer(tokenizer);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_melspectrogram_model(whisperkit_configuration_t *config, const char* melspectrogram_model) {
    if(config == nullptr || melspectrogram_model == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_melspectrogram_model(melspectrogram_model);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_postproc(whisperkit_configuration_t *config, const char* postproc) {
    if(config == nullptr || postproc == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_postproc(postproc);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_lib_dir(whisperkit_configuration_t *config, const char* lib_dir) {
    if(config == nullptr || lib_dir == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_lib_dir(lib_dir);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_cache_dir(whisperkit_configuration_t *config, const char* cache_dir) {
    if(config == nullptr || cache_dir == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_cache_dir(cache_dir);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_verbose(whisperkit_configuration_t *config, bool verbose) {
    if(config == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_verbose(verbose);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_log_level(whisperkit_configuration_t *config, int log_level) {
    if(config == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_log_level(log_level);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_prewarm(whisperkit_configuration_t *config, bool prewarm) {
    if(config == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_prewarm(prewarm);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_load(whisperkit_configuration_t *config, bool load) {
    if(config == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(config->pipeline != nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    config->set_load(load);
    return WHISPERKIT_STATUS_SUCCESS;
};


#pragma mark - transcription
whisperkit_status_t whisperkit_transcribe(whisperkit_pipeline_t *pipeline, const char* audio_file, char **transcription) {
    if(pipeline == nullptr || audio_file == nullptr || transcription == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    if(pipeline->configuration == nullptr) {
        return WHISPERKIT_STATUS_INVALID_STATE;
    }
    return pipeline->transcribe(audio_file, transcription);
};

#pragma mark - teardown
whisperkit_status_t whisperkit_destroy_config(whisperkit_configuration_t *config) {
    if(config == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    delete config;
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_destroy_pipeline(whisperkit_pipeline_t *pipeline) {
    if(pipeline == nullptr) {
        return WHISPERKIT_STATUS_INVALID_ARGUMENT;
    }
    delete pipeline;
    return WHISPERKIT_STATUS_SUCCESS;
};
