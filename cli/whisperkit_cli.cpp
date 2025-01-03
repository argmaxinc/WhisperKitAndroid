//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include <iostream>
#include "cxxopts.hpp"
#include "WhisperKit.h"

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

    WhisperKitConfig()  {
        audioPath = "";
        modelPath = "";
        audioEncoderComputeUnits = "";
        textDecoderComputeUnits = "";
        temperature = 0.f;
        temperatureIncrementOnFallback = 0.2f;
        temperatureFallbackCount = 5;
        bestOf = 5;
        skipSpecialTokens = false;
        withoutTimestamps = false;
        wordTimestamps = false;
        logprobThreshold = -1.f;
        firstTokenLogProbThreshold = -1.f;
        noSpeechThreshold = 0.3f;
        report = false;
        reportPath = ".";
        concurrentWorkerCount = 4;
    };


};

void CHECK_WHISPERKIT_STATUS(whisperkit_status_t status) {
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        throw std::runtime_error("WhisperKit error: " + std::to_string(status));
    }
}

struct WhisperKitRunner {

    WhisperKitRunner(WhisperKitConfig& config) : config(config) {
    }

    ~WhisperKitRunner() {
        if (pipeline) {
            whisperkit_pipeline_destroy(pipeline);
        }
        if (configuration) {
            whisperkit_configuration_destroy(configuration);
        }
    }

    void buildPipeline() {
        whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;
        status = whisperkit_configuration_create(&configuration);
        CHECK_WHISPERKIT_STATUS(status);

        auto audioEncoderPath = config.modelPath + "/AudioEncoder.tflite";
        status = whisperkit_configuration_set_audio_encoder(configuration, audioEncoderPath.c_str());
        CHECK_WHISPERKIT_STATUS(status);

        auto textDecoderPath = config.modelPath + "/TextDecoder.tflite";
        status = whisperkit_configuration_set_text_decoder(configuration, textDecoderPath.c_str());
        CHECK_WHISPERKIT_STATUS(status);

        status = whisperkit_pipeline_create(&pipeline);
        CHECK_WHISPERKIT_STATUS(status);

        status = whisperkit_pipeline_set_configuration(pipeline, configuration);
        CHECK_WHISPERKIT_STATUS(status);

        status = whisperkit_pipeline_build(pipeline);
        CHECK_WHISPERKIT_STATUS(status);

    }

    void transcribe() {
        whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;
        char* transcription = nullptr;
        status = whisperkit_pipeline_transcribe(pipeline, config.audioPath.c_str(), &transcription);
        CHECK_WHISPERKIT_STATUS(status);
        std::cout << "Transcription: " << transcription << std::endl;

        free(transcription);
    }

    private: 
        WhisperKitConfig config;
        whisperkit_pipeline_t* pipeline;
        whisperkit_configuration_t* configuration;
};



int main(int argc, char* argv[]) {

    WhisperKitConfig config;

    try {
        cxxopts::Options options("whisperkit-cli", "WhisperKit CLI for Android & Linux");

        options.add_options("transcribe")
            ("h,help", "Print help")
            ("audio-path", "Path to audio file", cxxopts::value<std::string>())
            ("model-path", "Path of model files", cxxopts::value<std::string>())
            //("audio-encoder-compute-units", "Compute units for audio encoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}", cxxopts::value<std::string>())
            //("text-decoder-compute-units", "Compute units for text decoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine,random}", cxxopts::value<std::string>())
            //("temperature", "Temperature to use for sampling", cxxopts::value<float>()->default_value("0.0"))
            //("temperature-increment-on-fallback", "Temperature to increase on fallbacks during decoding", cxxopts::value<float>()->default_value("0.2"))
            //("temperature-fallback-count", "Number of times to increase temperature when falling back during decoding", cxxopts::value<int>()->default_value("5"))
            //("best-of", "Number of candidates when sampling with non-zero temperature", cxxopts::value<int>()->default_value("5"))
            //("skip-special-tokens", "Skip special tokens in the output", cxxopts::value<bool>()->default_value("false"))
            //("without-timestamps", "Force no timestamps when decoding", cxxopts::value<bool>()->default_value("false"))
            //("word-timestamps", "Add timestamps for each word in the output", cxxopts::value<bool>()->default_value("false"))
            //("logprob-threshold", "Average log probability threshold for decoding failure", cxxopts::value<float>()->default_value("-1.0")) // TODO: check
            //("first-token-log-prob-threshold", "Log probability threshold for first token decoding failure", cxxopts::value<float>()->default_value("-1.0"))
            //("no-speech-threshold",  "Probability threshold to consider a segment as silence", cxxopts::value<float>()->default_value("-1.0"))
            ("report","Output a report of the results", cxxopts::value<bool>()->default_value("false"))
            ("report-path", "Directory to save the report", cxxopts::value<std::string>()->default_value("."))
            //("concurrent-worker-count", "Maximum concurrent inference, might be helpful when processing more than 1 audio file at the same time. 0 means unlimited. Default: 4", cxxopts::value<int>()->default_value("4"))
            ("v,verbose", "Verbose mode for debug", cxxopts::value<bool>()->default_value("false"));

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        

        if (result.count("audio-path")) {
            config.audioPath = result["audio-path"].as<std::string>();
        }
        if (result.count("model-path")) {
            config.modelPath = result["model-path"].as<std::string>();
        }
        if (result.count("audio-encoder-compute-units")) {
            config.audioEncoderComputeUnits = result["audio-encoder-compute-units"].as<std::string>();
        }
        if (result.count("text-decoder-compute-units")) {
            config.textDecoderComputeUnits = result["text-decoder-compute-units"].as<std::string>();
        }
        if (result.count("temperature")) {
            config.temperature = result["temperature"].as<float>();
        }
        if (result.count("temperature-increment-on-fallback")) {
            config.temperatureIncrementOnFallback = result["temperature-increment-on-fallback"].as<float>();
        }
        if (result.count("temperature-fallback-count")) {
            config.temperatureFallbackCount = result["temperature-fallback-count"].as<int>();
        }
        if (result.count("best-of")) {
            config.bestOf = result["best-of"].as<int>();
        }
        if (result.count("skip-special-tokens")) {
            config.skipSpecialTokens = result["skip-special-tokens"].as<bool>();
        }
        if (result.count("without-timestamps")) {
            config.withoutTimestamps = result["without-timestamps"].as<bool>();
        }
        if (result.count("word-timestamps")) {
            config.wordTimestamps = result["word-timestamps"].as<bool>();
        }
        if (result.count("logprob-threshold")) {
            config.logprobThreshold = result["logprob-threshold"].as<float>();
        }
        if (result.count("first-token-log-prob-threshold")) {
            config.firstTokenLogProbThreshold = result["first-token-log-prob-threshold"].as<float>();
        }
        if (result.count("no-speech-threshold")) {
            config.noSpeechThreshold = result["no-speech-threshold"].as<float>();
        }
        if (result.count("report")) {
            config.report = result["report"].as<bool>();
        }
        if (result.count("report-path")) {
            config.reportPath = result["report-path"].as<std::string>();
        }
        if (result.count("concurrent-worker-count")) {
            config.concurrentWorkerCount = result["concurrent-worker-count"].as<int>();
        }
        if (result["verbose"].as<bool>()) {
            std::cout << "Verbose mode is ON." << std::endl;
        }
        

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return 1;
    }

    WhisperKitRunner runner(config);

    try {
        runner.buildPipeline();
        runner.transcribe();

    } catch (const std::exception& e) {
        std::cerr << "Error transcribing audio: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}


