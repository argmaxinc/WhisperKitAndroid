#pragma once

//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#ifdef __cplusplus
extern "C" {
#endif


// Versioning
#define WHISPERKIT_VERSION_MAJOR 0
#define WHISPERKIT_VERSION_MINOR 1
#define WHISPERKIT_VERSION_PATCH 0

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
    WHISPERKIT_STATUS_ERROR_INVALID_STATE = 13,
    WHISPERKIT_STATUS_ERROR_GENERIC = 1000,
} whisperkit_status_t;


/** \brief WhisperKit pipeline enum codes
 *
 *  Pipeline status codes for WhisperKit.
 *  The pipeline status is used to track the state of the pipeline object.
 */
typedef enum {
    WHISPERKIT_PIPELINE_STATUS_INITIALIZED = 0,
    WHISPERKIT_PIPELINE_STATUS_CONFIGURED = 1,
    WHISPERKIT_PIPELINE_STATUS_BUILT = 2,
    WHISPERKIT_PIPELINE_STATUS_INVALID = 999,
} whisperkit_pipeline_status_t;

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

/** \brief WhisperKit transcription result object 
 * 
 *  An opaque object that holds the result of a transcription.  
 *  The object is owned by the client, and should be destroyed using 
 *  whisperkit_transcription_result_destroy().
 */
typedef struct whisperkit_transcription_result_t whisperkit_transcription_result_t;

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
 * 
 *  The pipeline object is in state INITIALIZED after creation.
 */
whisperkit_status_t whisperkit_pipeline_create(whisperkit_pipeline_t **pipeline);

/** \brief WhisperKit transcription result initializer
 * 
 *  Allocates and initializes a new WhisperKit transcription result object.
 *  The returned object is owned by the client, and should be destroyed using
 *  whisperkit_transcription_result_destroy().  
 */
whisperkit_status_t whisperkit_transcription_result_create(whisperkit_transcription_result_t **transcription_result);

#pragma mark - configuration setters

/** \brief Set the path to the model directory for the WhisperKit pipeline.
 * 
 *  Sets the path to the model directory for use in the WhisperKit pipeline.
 *  The model directory must contain the audio encoder, text decoder, voice activity detector,
 *  tokenizer, MelSpectrogram, and post-processing models.
 */
whisperkit_status_t whisperkit_configuration_set_model_path(whisperkit_configuration_t *config, const char* model_dir);

/** \brief Set the path to the report (json) directory for the WhisperKit pipeline.
 * 
 *  Sets the path to the report directory for use in the WhisperKit pipeline.
 *  The report directory will contain the result json file.
 */
whisperkit_status_t whisperkit_configuration_set_report_path(whisperkit_configuration_t *config, const char* report_dir);

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

#pragma mark - pipeline state

/** \brief WhisperKit pipeline status query
 * 
 *  Returns the current status of the WhisperKit pipeline.
 */
whisperkit_status_t whisperkit_pipeline_get_status(whisperkit_pipeline_t *pipeline, whisperkit_pipeline_status_t* status);

#pragma mark - pipeline configuration

/** \brief WhisperKit pipeline builder
 * 
 *  Sets a configuration object for the WhisperKit pipeline.
 *  The configuration object must be set before the pipeline can be built.
 * 
 *  The pipeline must be in the INITIALIZED state before 
 *  whisperkit_pipeline_set_configuration can be called.  
 *  It is illegal to set the configuration on pipelines in the BUILT state.
 * 
 *  whisperkit_pipeline_set_configuration takes pipeline objects from the INITIALIZED state
 *  to the CONFIGURED state.
 */
whisperkit_status_t whisperkit_pipeline_set_configuration(whisperkit_pipeline_t *pipeline, whisperkit_configuration_t *config);

#pragma mark - pipeline building

/** \brief WhisperKit pipeline builder
 * 
 *  Builds the WhisperKit pipeline with the configuration settings that
 *  have been set.  Pipeline creation may fail if the configuration settings are invalid, if 
 *  models fail to load, or if other resources fail to initialize.
 * 
 *  Pipelines must be in the CONFIGURED state before calling whisperkit_pipeline_build.
 *  Pipelines that have been built are in the BUILT state.
 */
whisperkit_status_t whisperkit_pipeline_build(whisperkit_pipeline_t *pipeline);

#pragma mark - transcription

/** \brief WhisperKit pipeline transcription
 * 
 *  Transcribes the provided audio file using the created WhisperKit pipeline object.
 *  The transcription is returned as a string.
 * 
 *  The pipeline must be in the BUILT state before whisperkit_pipeline_transcribe can be 
 *  called.
 */
whisperkit_status_t whisperkit_pipeline_transcribe(whisperkit_pipeline_t *pipeline, const char* audio_file, whisperkit_transcription_result_t *transcription_result);

/** \brief WhisperKit transcription result getter   
 * 
 *  Retrieves the transcription from the transcription result object.
 *  The returned string has the lifetime of the transcription result object.
 */
whisperkit_status_t whisperkit_transcription_result_get_transcription(whisperkit_transcription_result_t *transcription_result, char **transcription);

#pragma mark - teardown

/** \brief WhisperKit configuration destroyer
 * 
 *  Releases the created configuration object and frees the memory.
 */
whisperkit_status_t whisperkit_configuration_destroy(whisperkit_configuration_t **config);

/** \brief WhisperKit pipeline destroyer
 * 
 *  Releases the created pipeline object and frees the memory.
 */
whisperkit_status_t whisperkit_pipeline_destroy(whisperkit_pipeline_t **pipeline);

/** \brief WhisperKit transcription result destroyer
 * 
 *  Releases the created transcription result object and frees the memory.
 */
whisperkit_status_t whisperkit_transcription_result_destroy(whisperkit_transcription_result_t **transcription_result);

#ifdef __cplusplus
}
#endif
