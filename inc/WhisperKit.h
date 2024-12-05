#pragma once

//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#ifdef __cplusplus
extern "C" {
#endif


// Versioning
#define WHISPERKIT_VERSION_MAJOR 0
#define WHISPERKIT_VERSION_MINOR 1

/** \brief WhisperKit status codes.
 *
 *  All C-API functions return a status code of type whisperkit_status_t.
 *  Please always check for success or error status after calling a function.
 */
typedef enum {
    WHISPERKIT_STATUS_SUCCESS = 0,
    WHISPERKIT_STATUS_ERROR_INVALID_ARGUMENT = 1,
    WHISPERKIT_STATUS_ERROR_TOKENIZER_UNAVAILABLE = 2,
    WHISPERKIT_STATUS_ERROR_MODEL_UNAVAILABLE = 3, 
    WHISPERKIT_STATUS_ERROR_PREFILL_FAILED = 4,
    WHISPERKIT_STATUS_ERROR_AUDIO_PROCESSING_FAILED = 5,
    WHISPERKIT_STATUS_ERROR_DECODING_LOGITS_FAILED = 6, 
    WHISPERKIT_STATUS_ERROR_SEGMENTING_FAILED = 7,
    WHISPERKIT_STATUS_ERROR_LOAD_AUDIO_FAILED = 8,
    WHISPERKIT_STATUS_ERROR_PREPARE_DECODER_INPUTS_FAILED = 9, 
    WHISPERKIT_STATUS_ERROR_TRANSCRIPTION_FAILED = 10,
    WHISPERKIT_STATUS_ERROR_DECODING_FAILED = 11,
    WHISPERKIT_STATUS_ERROR_MICROPHONE_UNAVAILABLE = 12, 
    WHISPERKIT_STATUS_ERROR_GENERIC = 1000,
} whisperkit_status_t;

#pragma mark - Configuration

/** \brief WhisperKit configuration object.
 * 
 *  An opaque object that holds configuration settings for the WhisperKit pipeline.
 */
typedef struct whisperkit_configuration_t whisperkit_configuration_t;

#pragma mark - WhisperKit

/** \brief WhisperKit pipeline object
 * 
 *  An opaque object that represents the WhisperKit pipeline.
 *  The pipeline object is the execution context for the WhisperKit transcription process.
 */
typedef struct whisperkit_pipeline_t whisperkit_pipeline_t;

#pragma mark - initializers

/** \brief WhisperKit configuration initializer
 * 
 *  Allocates and initializes a new WhisperKit configuration object.
 *  The configuration object is used to set up the pipeline with the desired settings.
 *  The returned object is owned by the client, and should be destroyed using 
 *  whisperkit_configuration_destroy().
 */
whisperkit_status_t whisperkit_configuration_create(whisperkit_configuration_t **configuration);

/** \brief WhisperKit pipeline initializer
 * 
 *  Allocates and initializes a new WhisperKit pipeline object.
 *  The returned object is owned by the client, and should be destroyed using
 *  whisperkit_pipeline_destroy().
 */
whisperkit_status_t whisperkit_pipeline_create(whisperkit_pipeline_t **pipeline);

#pragma mark - configuration setters

/** \brief Set the audio encoder model for the WhisperKit pipeline.
 * 
 *  Sets the path to the audio encoder model file for use in transcription.
 *  The audio encoder model must be a TFLite model file.
 */
whisperkit_status_t whisperkit_configuration_set_audio_encoder(whisperkit_configuration_t *config, const char* audio_encoder);


/** \brief Set the text decoder model for the WhisperKit pipeline.
 * 
 *  Sets the path to the text decoder model file for use in transcription.
 *  The audio encoder model must be a TFLite model file.
 */
whisperkit_status_t whisperkit_configuration_set_text_decoder(whisperkit_configuration_t *config, const char* text_decoder);

/** \brief Set the voice activity detector model for the WhisperKit pipeline.
 * 
 *  Sets the path to the text decoder model file for use in transcription.
 *  The audio encoder model must be a TFLite model file.
 */
whisperkit_status_t whisperkit_configuration_set_voice_activity_detector(whisperkit_configuration_t *config, const char* voice_activity_detector);

/** \brief Set the path to the tokenizer configuration file for the WhisperKit pipeline.
 * 
 *  Sets the path to the tokenizer for use in the WhisperKit pipeline.
 *  The tokenizer configuration file must be a JSON file.
 */
whisperkit_status_t whisperkit_configuration_set_tokenizer(whisperkit_configuration_t *config, const char* tokenizer);

/** \brief Set the path to the MelSpectrogram model for the WhisperKit pipeline.
 * 
 *  Sets the path to the MelSpectrogram model for use in the WhisperKit pipeline.
 *  The MelSpectrogram model must be a TFLite model file.
 */
whisperkit_status_t whisperkit_configuration_set_melspectrogram_model(whisperkit_configuration_t *config, const char* melspectrogram_model);

/** \brief Set the path to the post-processing model for the WhisperKit pipeline.
 * 
 *  Sets the path to the post-processing model for use in the WhisperKit pipeline.
 *  The post-processing model must be a TFLite model file.
 */
whisperkit_status_t whisperkit_configuration_set_postproc(whisperkit_configuration_t *config, const char* postproc);

/** \brief Set the path to QNN Skel library directory for the WhisperKit pipeline.
 * 
 *  Sets the path to the Skel library directory for use with models delegated to QNN backend.
 */
whisperkit_status_t whisperkit_configuration_set_lib_dir(whisperkit_configuration_t *config, const char* lib_dir);

/** \brief Set the path to cache directory for use by models within the WhisperKit pipeline.
 * 
 *  Sets the path to a directory to be used as a cache for assorted purposes, including 
 *  but not limited to GPU shader compilation and model caching.
 */
whisperkit_status_t whisperkit_configuration_set_cache_dir(whisperkit_configuration_t *config, const char* cache_dir);

/** \brief Set verbosity on or off for the WhisperKit pipeline
 * 
 *  Enables verbose printing and logging for debugging purposes.
 *  This should not be used in production scenarios.
 */
whisperkit_status_t whisperkit_configuration_set_verbose(whisperkit_configuration_t *config, bool verbose);

/** \brief Set log level for the WhisperKit pipeline
 * 
 *  Sets the log level to a given specification.
 */
whisperkit_status_t whisperkit_configuration_set_log_level(whisperkit_configuration_t *config, int log_level);

/** \brief Enable or disable prewarming for the WhisperKit pipeline
 * 
 *  Prewarming can lead to faster first inference times, but increase peak memory or 
 *  power consumption while initializing the pipeline.
 */
whisperkit_status_t whisperkit_configuration_set_prewarm(whisperkit_configuration_t *config, bool prewarm);

/** \brief Enable or disable prewarming for the WhisperKit pipeline
 * 
 *  Prewarming can lead to faster first inference times, but increase peak memory or 
 *  power consumption while initializing the pipeline.
 */
whisperkit_status_t whisperkit_configuration_set_load(whisperkit_configuration_t *config, bool load);

#pragma mark - pipeline configuration

/** \brief WhisperKit pipeline builder
 * 
 *  Builds and initializes the WhisperKit pipeline with the configuration settings that
 *  have been set.
 *  Pipeline creation may fail if the configuration settings are invalid, if 
 *  models fail to load, or if other resources fail to initialize.
 */
whisperkit_status_t whisperkit_pipeline_set_configuration(whisperkit_pipeline_t *pipeline, whisperkit_configuration_t *config);

#pragma mark - pipeline building

/** \brief WhisperKit pipeline builder
 * 
 *  Builds and initializes the WhisperKit pipeline with the configuration settings that
 *  have been set.
 *  Pipeline creation may fail if the configuration settings are invalid, if 
 *  models fail to load, or if other resources fail to initialize.
 */
whisperkit_status_t whisperkit_pipeline_build(whisperkit_pipeline_t *pipeline);

#pragma mark - transcription

/** \brief WhisperKit pipeline transcription
 * 
 *  Transcribes the provided audio file using the created WhisperKit pipeline object.
 *  The transcription is returned as a string.
 *  The pipeline must be in the BUILT state before whisperkit_pipeline_transcribe can be 
 *  called.
 */
whisperkit_status_t whisperkit_pipeline_transcribe(whisperkit_pipeline_t *pipeline, const char* audio_file, char **transcription);

#pragma mark - teardown

/** \brief WhisperKit configuration destroyer
 * 
 *  Releases the created configuration object and frees the memory.
 */
whisperkit_status_t whisperkit_configuration_destroy(whisperkit_configuration_t *config);

/** \brief WhisperKit pipeline destroyer
 * 
 *  Releases the created pipeline object and frees the memory.
 */
whisperkit_status_t whisperkit_pipeline_destroy(whisperkit_pipeline_t *pipeline);

#ifdef __cplusplus
}
#endif
