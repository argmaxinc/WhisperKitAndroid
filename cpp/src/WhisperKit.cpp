#include "WhisperKit.h"

#include "WhisperKitConfiguration.hpp"
#include "WhisperKitPipeline.hpp"
#include "WhisperKitTranscriptionResult.hpp"
#pragma mark - initializers
whisperkit_status_t whisperkit_configuration_create(whisperkit_configuration_t **configuration) {
    if (configuration == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    *configuration = new whisperkit_configuration_t();
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_pipeline_create(whisperkit_pipeline_t **pipeline) {
    if (pipeline == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    *pipeline = new whisperkit_pipeline_t();
    (*pipeline)->set_state(WHISPERKIT_PIPELINE_STATUS_INITIALIZED);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_transcription_result_create(whisperkit_transcription_result_t **transcription_result) {
    if (transcription_result == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    *transcription_result = new whisperkit_transcription_result_t();
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_audio_encoder(whisperkit_configuration_t *config,
                                                               const char *audio_encoder) {
    if (config == nullptr || audio_encoder == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_audio_encoder(audio_encoder);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_text_decoder(whisperkit_configuration_t *config,
                                                              const char *text_decoder) {
    if (config == nullptr || text_decoder == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_text_decoder(text_decoder);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_tokenizer(whisperkit_configuration_t *config, const char *tokenizer) {
    if (config == nullptr || tokenizer == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_tokenizer(tokenizer);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_melspectrogram_model(whisperkit_configuration_t *config,
                                                                      const char *melspectrogram_model) {
    if (config == nullptr || melspectrogram_model == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_melspectrogram_model(melspectrogram_model);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_lib_dir(whisperkit_configuration_t *config, const char *lib_dir) {
    if (config == nullptr || lib_dir == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_lib_dir(lib_dir);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_cache_dir(whisperkit_configuration_t *config, const char *cache_dir) {
    if (config == nullptr || cache_dir == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_cache_dir(cache_dir);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_verbose(whisperkit_configuration_t *config, bool verbose) {
    if (config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_verbose(verbose);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_log_level(whisperkit_configuration_t *config, int log_level) {
    if (config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_log_level(log_level);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_prewarm(whisperkit_configuration_t *config, bool prewarm) {
    if (config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_prewarm(prewarm);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_load(whisperkit_configuration_t *config, bool load) {
    if (config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_load(load);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_model_path(whisperkit_configuration_t *config,
                                                            const char *model_path) {
    if (config == nullptr || model_path == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_model_path(model_path);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_report_path(whisperkit_configuration_t *config,
                                                             const char *report_dir) {
    if (config == nullptr || report_dir == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_report_path(report_dir);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_configuration_set_backends(whisperkit_configuration_t *config,
                                                          whisperkit_backend_t encoder_backend,
                                                          whisperkit_backend_t decoder_backend) {
    if (config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    config->set_backends(encoder_backend, decoder_backend);
    return WHISPERKIT_STATUS_SUCCESS;
};

#pragma mark - pipeline state
whisperkit_status_t whisperkit_pipeline_get_status(whisperkit_pipeline_t *pipeline,
                                                   whisperkit_pipeline_status_t *status) {
    if (pipeline == nullptr || status == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    *status = pipeline->get_state();
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_pipeline_set_configuration(whisperkit_pipeline_t *pipeline,
                                                          whisperkit_configuration_t *config) {
    if (pipeline == nullptr || config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_INITIALIZED &&
        pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_CONFIGURED) {
        return WHISPERKIT_STATUS_ERROR_INVALID_STATE;
    }
    pipeline->set_configuration(config);
    pipeline->set_state(WHISPERKIT_PIPELINE_STATUS_CONFIGURED);
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_pipeline_build(whisperkit_pipeline_t *pipeline) {
    if (pipeline == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_CONFIGURED) {
        return WHISPERKIT_STATUS_ERROR_INVALID_STATE;
    }

    try {
        pipeline->build();
        pipeline->set_state(WHISPERKIT_PIPELINE_STATUS_BUILT);
    } catch (const std::exception &e) {
        return WHISPERKIT_STATUS_ERROR_GENERIC;
    }
    return WHISPERKIT_STATUS_SUCCESS;
};

#pragma mark - transcription
whisperkit_status_t whisperkit_pipeline_transcribe(whisperkit_pipeline_t *pipeline, const char *audio_file,
                                                   whisperkit_transcription_result_t *transcription_result) {
    if (pipeline == nullptr || audio_file == nullptr || transcription_result == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_BUILT) {
        return WHISPERKIT_STATUS_ERROR_INVALID_STATE;
    }

    try {
        pipeline->transcribe(audio_file, transcription_result);
    } catch (const std::exception &e) {
        return WHISPERKIT_STATUS_ERROR_TRANSCRIPTION_FAILED;
    }
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_pipeline_initstreaming(whisperkit_pipeline_t *pipeline,
                                                      whisperkit_transcription_result_t *transcription_result,
                                                      int sample_rate, int num_channels) {
    if (pipeline == nullptr || transcription_result == nullptr || sample_rate <= 0 || num_channels <= 0) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_BUILT) {
        return WHISPERKIT_STATUS_ERROR_INVALID_STATE;
    }

    try {
        pipeline->init_streaming(transcription_result, sample_rate, num_channels);
        pipeline->set_state(WHISPERKIT_PIPELINE_STATUS_AUDIOINIT);
    } catch (const std::exception &e) {
        return WHISPERKIT_STATUS_ERROR_TRANSCRIPTION_FAILED;
    }
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_pipeline_appendaudio(whisperkit_pipeline_t *pipeline, int size, char *buffer,
                                                    int *transcribed) {
    if (pipeline == nullptr || size <= 0 || buffer == nullptr || transcribed == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_AUDIOINIT) {
        return WHISPERKIT_STATUS_ERROR_INVALID_STATE;
    }

    *transcribed = 0;
    try {
        bool is_transcribed = pipeline->append_audio(size, buffer);
        if (is_transcribed) *transcribed = 1;
    } catch (const std::exception &e) {
        return WHISPERKIT_STATUS_ERROR_TRANSCRIPTION_FAILED;
    }
    return WHISPERKIT_STATUS_SUCCESS;
}

whisperkit_status_t whisperkit_pipeline_closestreaming(whisperkit_pipeline_t *pipeline) {
    if (pipeline == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (pipeline->get_state() != WHISPERKIT_PIPELINE_STATUS_AUDIOINIT) {
        return WHISPERKIT_STATUS_ERROR_INVALID_STATE;
    }

    try {
        pipeline->close_streaming();
    } catch (const std::exception &e) {
        return WHISPERKIT_STATUS_ERROR_TRANSCRIPTION_FAILED;
    }
    return WHISPERKIT_STATUS_SUCCESS;
}

whisperkit_status_t whisperkit_transcription_result_get_all_transcription(
    whisperkit_transcription_result_t *transcription_result, char **transcription) {
    if (transcription_result == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const auto &transcription_result_string = transcription_result->get_transcription();

    size_t size = transcription_result_string.size() + 1;
    *transcription = new char[size];

    std::snprintf(*transcription, size, "%s", transcription_result_string.c_str());
    return WHISPERKIT_STATUS_SUCCESS;
}

whisperkit_status_t whisperkit_transcription_result_get_transcription(
    whisperkit_transcription_result_t *transcription_result, char **transcription) {
    if (transcription_result == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const auto &transcription_result_string = transcription_result->get_chunk_transcription();

    size_t size = transcription_result_string.size() + 1;
    *transcription = new char[size];

    std::snprintf(*transcription, size, "%s", transcription_result_string.c_str());
    return WHISPERKIT_STATUS_SUCCESS;
};

#pragma mark - teardown
whisperkit_status_t whisperkit_configuration_destroy(whisperkit_configuration_t **config) {
    if (config == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    delete *config;
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_pipeline_destroy(whisperkit_pipeline_t **pipeline) {
    if (pipeline == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    delete *pipeline;
    return WHISPERKIT_STATUS_SUCCESS;
};

whisperkit_status_t whisperkit_transcription_result_destroy(whisperkit_transcription_result_t **transcription_result) {
    if (transcription_result == nullptr) {
        return WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT;
    }
    delete *transcription_result;
    return WHISPERKIT_STATUS_SUCCESS;
};
