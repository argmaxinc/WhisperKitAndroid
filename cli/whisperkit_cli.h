#ifndef WHISPERKIT_CLI_H
#define WHISPERKIT_CLI_H

#include <iostream>
#include <string>

#include "WhisperKit.h"
#include "cxxopts.hpp"

struct WhisperKitConfig {
   public:
    std::string audioPath;
    std::string modelPath;
    std::string audioEncoderComputeUnits;
    std::string textDecoderComputeUnits;
    float temperature;
    float temperatureIncrementOnFallback;
    int temperatureFallbackCount;
    int bestOf;
    bool skipSpecialTokens;
    bool withoutTimestamps;
    bool wordTimestamps;
    float logprobThreshold;
    float firstTokenLogProbThreshold;
    float noSpeechThreshold;
    bool report;
    std::string reportPath;
    int concurrentWorkerCount;
    bool verbose;
    whisperkit_backend_t encoder_backend;
    whisperkit_backend_t decoder_backend;

    WhisperKitConfig();
};

void CHECK_WHISPERKIT_STATUS(whisperkit_status_t status);

class WhisperKitRunner {
   public:
    explicit WhisperKitRunner(WhisperKitConfig& config);
    ~WhisperKitRunner();
    void buildPipeline();
    void transcribe();
    whisperkit_transcription_result_t* transcriptionResult;

   private:
    WhisperKitConfig& config;
    whisperkit_pipeline_t* pipeline;
    whisperkit_configuration_t* configuration;
};

#endif  // WHISPERKIT_CLI_H