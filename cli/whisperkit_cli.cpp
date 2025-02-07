//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include "whisperkit_cli.h"
#include <iostream>
#include <string>
#include <android/log.h>

#define LOG_TAG "WhisperKitCLI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

class Timer
{
public:
    void start()
    {
        m_StartTime = std::chrono::system_clock::now();
        m_bRunning = true;
    }
    
    void stop()
    {
        m_EndTime = std::chrono::system_clock::now();
        m_bRunning = false;
    }
    
    double elapsedMilliseconds()
    {
        std::chrono::time_point<std::chrono::system_clock> endTime;
        
        if(m_bRunning)
        {
            endTime = std::chrono::system_clock::now();
        }
        else
        {
            endTime = m_EndTime;
        }
        
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
    }
    
    double elapsedSeconds()
    {
        return elapsedMilliseconds() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool                                               m_bRunning = false;
};

WhisperKitConfig::WhisperKitConfig()  {
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


void CHECK_WHISPERKIT_STATUS(whisperkit_status_t status) {
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("WhisperKit error: %d", status);
        throw std::runtime_error("WhisperKit error: " + std::to_string(status));
    }
}


WhisperKitRunner::WhisperKitRunner(WhisperKitConfig& config) : config(config) {

    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;
    status = whisperkit_configuration_create(&configuration);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_pipeline_create(&pipeline);
    CHECK_WHISPERKIT_STATUS(status);
    LOGI("WhisperKitRunner constructor ran without errors!");
}

void WhisperKitRunner::buildPipeline() {
    LOGI("Entered buildPipeline function");
    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;
    if (configuration == nullptr) {
        LOGE("buildPipeline: configuration pointer is null");
        return;
    }
    status = whisperkit_configuration_set_model_path(configuration, config.modelPath.c_str());
    CHECK_WHISPERKIT_STATUS(status);
    LOGI("Model path set");
    if (config.report){
        status = whisperkit_configuration_set_report_path(configuration, config.reportPath.c_str());
        CHECK_WHISPERKIT_STATUS(status);
    }

    status = whisperkit_pipeline_set_configuration(pipeline, configuration);
    CHECK_WHISPERKIT_STATUS(status);
    LOGI("Pipeline configuration set");
    status = whisperkit_pipeline_build(pipeline);
    CHECK_WHISPERKIT_STATUS(status);
    LOGI("Pipeline build succeeded");
}

void WhisperKitRunner::transcribe() {
    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;

    status = whisperkit_transcription_result_create(&transcriptionResult);
    CHECK_WHISPERKIT_STATUS(status);

    status = whisperkit_pipeline_transcribe(pipeline, config.audioPath.c_str(), transcriptionResult);
    CHECK_WHISPERKIT_STATUS(status);

    char* transcription = nullptr;
    status = whisperkit_transcription_result_get_transcription(transcriptionResult, &transcription);
    CHECK_WHISPERKIT_STATUS(status);

    std::string transcriptionString(transcription);
    std::cout << "Transcription: " << transcriptionString.c_str() << std::endl;

    if(transcription != nullptr) {
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

        options.add_options("transcribe")
            ("h,help", "Print help")
            ("audio-path", "Path to audio file", cxxopts::value<std::string>())
            ("model-path", "Path of model files", cxxopts::value<std::string>())
            ("report","Output a report of the results", cxxopts::value<bool>()->default_value("false"))
            ("report-path", "Directory to save the report", cxxopts::value<std::string>()->default_value("."))
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
        if (result.count("report")) {
            config.report = result["report"].as<bool>();
        }
        if (result.count("report-path")) {
            config.reportPath = result["report-path"].as<std::string>();
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
        Timer t = Timer();
        t.start();
        runner.buildPipeline();
        t.stop();
        double buildTime = t.elapsedMilliseconds();
        
        t.start();
        runner.transcribe();
        t.stop();
        double transcribeTime = t.elapsedMilliseconds();
        std::cout << "Elapsed time (build + transcription) " << transcribeTime + buildTime << " ms" << std::endl;
        std::cout << "Build " << buildTime << " ms" << std::endl;
        std::cout << "Transcribe " << transcribeTime << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error transcribing audio: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}


