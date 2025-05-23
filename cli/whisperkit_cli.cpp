//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include "whisperkit_cli.h"

#include <iostream>
#include <string>

WhisperKitConfig::WhisperKitConfig() {
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
    verbose = false;
#if QNN_DELEGATE
    encoder_backend = WHISPERKIT_COMPUTE_BACKEND_NPU;
    decoder_backend = WHISPERKIT_COMPUTE_BACKEND_NPU;
#else
    encoder_backend = WHISPERKIT_COMPUTE_BACKEND_GPU;
    decoder_backend = WHISPERKIT_COMPUTE_BACKEND_GPU;
#endif
};

void CHECK_WHISPERKIT_STATUS(whisperkit_status_t status) {
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        throw std::runtime_error("WhisperKit error: " + std::to_string(status));
    }
}

WhisperKitRunner::WhisperKitRunner(WhisperKitConfig& config) : config(config) {
    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;
    status = whisperkit_configuration_create(&configuration);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_pipeline_create(&pipeline);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_configuration_set_backends(configuration, config.encoder_backend, config.decoder_backend);
    CHECK_WHISPERKIT_STATUS(status);
}

void WhisperKitRunner::buildPipeline() {
    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;

    status = whisperkit_configuration_set_model_path(configuration, config.modelPath.c_str());
    CHECK_WHISPERKIT_STATUS(status);

    if (config.report) {
        status = whisperkit_configuration_set_report_path(configuration, config.reportPath.c_str());
        CHECK_WHISPERKIT_STATUS(status);
    }
    status = whisperkit_configuration_set_verbose(configuration, config.verbose);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_pipeline_set_configuration(pipeline, configuration);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_pipeline_build(pipeline);
    CHECK_WHISPERKIT_STATUS(status);
}

void WhisperKitRunner::transcribe() {
    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;

    status = whisperkit_transcription_result_create(&transcriptionResult);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_pipeline_transcribe(pipeline, config.audioPath.c_str(), transcriptionResult);
    CHECK_WHISPERKIT_STATUS(status);

    char* transcription = nullptr;
    status = whisperkit_transcription_result_get_all_transcription(transcriptionResult, &transcription);
    CHECK_WHISPERKIT_STATUS(status);

    if (transcription != nullptr) {
        free((void*)transcription);
    }
}

WhisperKitRunner::~WhisperKitRunner() {
    if (pipeline) {
        whisperkit_pipeline_destroy(&pipeline);
    }
    if (configuration) {
        whisperkit_configuration_destroy(&configuration);
    }
    if (transcriptionResult) {
        whisperkit_transcription_result_destroy(&transcriptionResult);
    }
}

int main(int argc, char* argv[]) {
    WhisperKitConfig config;

    try {
        cxxopts::Options options("whisperkit-cli", "WhisperKit CLI for Android & Linux");

        options.add_options("transcribe")("h,help", "Print help")(
            "a,audio-path", "Path to audio file", cxxopts::value<std::string>())("m,model-path", "Path of model files",
                                                                                 cxxopts::value<std::string>())(
            "r,report", "Output a report of the results", cxxopts::value<bool>()->default_value("false"))(
            "p,report-path", "Directory to save the report", cxxopts::value<std::string>()->default_value("."))(
            "v,verbose", "Verbose mode for debug", cxxopts::value<bool>()->default_value("false"))
#if QNN_DELEGATE
            ("c,compute-unit", "CPU/GPU/NPU", cxxopts::value<std::string>()->default_value("NPU"));
#else
            ("c,compute-unit", "CPU/GPU", cxxopts::value<std::string>()->default_value("GPU"));
#endif

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
        if (result.count("report")) {
            config.report = result["report"].as<bool>();
        }
        if (result.count("report-path")) {
            config.reportPath = result["report-path"].as<std::string>();
        }
        if (result["verbose"].as<bool>()) {
            config.verbose = true;
            std::cout << "Verbose mode is ON." << std::endl;
        }

        if (result.count("compute-unit")) {
            auto unit = result["compute-unit"].as<std::string>();
            if (unit == "CPU") {
                config.encoder_backend = WHISPERKIT_COMPUTE_BACKEND_CPU;
                config.decoder_backend = WHISPERKIT_COMPUTE_BACKEND_CPU;
            } else if (unit == "NPU") {
                config.encoder_backend = WHISPERKIT_COMPUTE_BACKEND_NPU;
                config.decoder_backend = WHISPERKIT_COMPUTE_BACKEND_NPU;
            } else {
                config.encoder_backend = WHISPERKIT_COMPUTE_BACKEND_GPU;
                config.decoder_backend = WHISPERKIT_COMPUTE_BACKEND_GPU;
            }
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
