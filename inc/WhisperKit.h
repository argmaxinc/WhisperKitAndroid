#pragma once

//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#ifdef __cplusplus
extern "C" {
#endif


// Versioning
#define WHISPERKIT_VERSION_MAJOR 0
#define WHISPERKIT_VERSION_MINOR 1

// Error codes
typedef enum {
    WHISPERKIT_SUCCESS = 0,
    WHISPERKIT_ERROR_INVALID_ARGUMENT = 1,
    WHISPERKIT_ERROR_TOKENIZER_UNAVAILABLE = 2,
    WHISPERKIT_ERROR_MODEL_UNAVAILABLE = 3, 
    WHISPERKIT_ERROR_PREFILL_FAILED = 4,
    WHISPERKIT_ERROR_AUDIO_PROCESSING_FAILED = 5,
    WHISPERKIT_ERROR_DECODING_LOGITS_FAILED = 6, 
    WHISPERKIT_ERROR_SEGMENTING_FAILED = 7,
    WHISPERKIT_ERROR_LOAD_AUDIO_FAILED = 8,
    WHISPERKIT_ERROR_PREPARE_DECODER_INPUTS_FAILED = 9, 
    WHISPERKIT_ERROR_TRANSCRIPTION_FAILED = 10,
    WHISPERKIT_ERROR_DECODING_FAILED = 11,
    WHISPERKIT_ERROR_MICROPHONE_UNAVAILABLE = 12, 
    WHISPERKIT_ERROR_GENERIC = 1000,
} whisperkit_status_t;

#pragma mark - Delegate Backend Options


#pragma mark - Configuration
typedef struct whisperkit_configuration_t whisperkit_configuration_t;

#pragma mark - WhisperKit
typedef struct whisperkit_pipeline_t whisperkit_pipeline_t;

#pragma mark - initializers
whisperkit_status_t whisperkit_create_configuration(whisperkit_configuration_t **configuration);

whisperkit_status_t whisperkit_create_pipeline(whisperkit_pipeline_t **pipeline, whisperkit_configuration_t *configuration);

#pragma mark - configuration setters
whisperkit_status_t whisperkit_configuration_set_audio_model(whisperkit_configuration_t *config, const char* audio_model);

whisperkit_status_t whisperkit_configuration_set_audio_encoder(whisperkit_configuration_t *config, const char* audio_encoder);

whisperkit_status_t whisperkit_configuration_set_text_decoder(whisperkit_configuration_t *config, const char* text_decoder);

whisperkit_status_t whisperkit_configuration_set_voice_activity_detector(whisperkit_configuration_t *config, const char* voice_activity_detector);

whisperkit_status_t whisperkit_configuration_set_tokenizer(whisperkit_configuration_t *config, const char* tokenizer);

whisperkit_status_t whisperkit_configuration_set_melspectrogram_model(whisperkit_configuration_t *config, const char* melspectrogram_model);

whisperkit_status_t whisperkit_configuration_set_postproc(whisperkit_configuration_t *config, const char* postproc);

whisperkit_status_t whisperkit_configuration_set_lib_dir(whisperkit_configuration_t *config, const char* lib_dir);

whisperkit_status_t whisperkit_configuration_set_cache_dir(whisperkit_configuration_t *config, const char* cache_dir);

whisperkit_status_t whisperkit_configuration_set_verbose(whisperkit_configuration_t *config, bool verbose);

whisperkit_status_t whisperkit_configuration_set_log_level(whisperkit_configuration_t *config, int log_level);

whisperkit_status_t whisperkit_configuration_set_prewarm(whisperkit_configuration_t *config, bool prewarm);

whisperkit_status_t whisperkit_configuration_set_load(whisperkit_configuration_t *config, bool load);

whisperkit_status_t whisperkit_configuration_set_use_background_download_session(whisperkit_configuration_t *config, bool use_background_download_session);

#pragma mark - configuration getters

#pragma mark - transcription
whisperkit_status_t whisperkit_transcribe(whisperkit_t *whisperkit, const char* audio_file, char **transcription);

#pragma mark - teardown
whisperkit_status_t whisperkit_destroy_config(whisperkit_configuration_t *config);

whisperkit_status_t whisperkit_destroy_pipeline(whisperkit_pipeline_t *pipeline);


// Delegate backend options

// 
/*
        let config = WhisperKitConfig(model: modelName,
                                      downloadBase: downloadModelFolder,
                                      modelFolder: cliArguments.modelPath,
                                      tokenizerFolder: downloadTokenizerFolder,
                                      computeOptions: computeOptions,
                                      verbose: cliArguments.verbose,
                                      logLevel: .debug,
                                      prewarm: false,
                                      load: true,
                                      useBackgroundDownloadSession: false)
        return try await WhisperKit(config)
        */