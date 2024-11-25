//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include <iostream>
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

    // Delete assignment, copy ctor
    WhisperKitConfig& operator=(const WhisperKitConfig&) = delete;
    WhisperKitConfig(const WhisperKitConfig&) = delete;

};


int main(int argc, char* argv[]) {
    try {
        cxxopts::Options options("whisperkit-cli", "WhisperKit CLI for Android & Linux");

        options.add_options()
            ("h,help", "Print help")
            ("audioPath", "Path to audio file", cxxopts::value<std::string>())
            ("modelPath", "Path of model files", cxxopts::value<std::string>())
            ("audioEncoderComputeUnits", "Compute units for audio encoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}", cxxopts::value<std::string>())
            ("textDecoderComputeUnits", "Compute units for text decoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine,random}", cxxopts::value<std::string>())
            ("temperature", "Temperature to use for sampling", cxxopts::value<float>()->default_value("0.0"))
            ("temperatureIncrementOnFallback", "Temperature to increase on fallbacks during decoding", cxxopts::value<float>()->default_value("0.2"))
            ("temperatureFallbackCount", "Number of times to increase temperature when falling back during decoding", cxxopts::value<int>()->default_value("5"))
            ("bestOf", "Number of candidates when sampling with non-zero temperature", cxxopts::value<int>()->default_value("5"))
            ("skipSpecialTokens", "Skip special tokens in the output", cxxopts::value<bool>()->default_value("false"))
            ("withoutTimestamps", "Force no timestamps when decoding", cxxopts::value<bool>()->default_value("false"))
            ("wordTimestamps", "Add timestamps for each word in the output", cxxopts::value<bool>()->default_value("false"))
            ("logprobThreshold", "Average log probability threshold for decoding failure", cxxopts::value<float>()->default_value("-1.0")) // TODO: check
            ("firstTokenLogProbThreshold", "Log probability threshold for first token decoding failure", cxxopts::value<float>()->default_value("-1.0"))
            ("noSpeechThreshold",  "Probability threshold to consider a segment as silence", cxxopts::value<float>()->default_value("-1.0"))
            ("report","Output a report of the results", cxxopts::value<bool>()->default_value("false"))
            ("reportPath", "Directory to save the report", cxxopts::value<std::string>()->default_value("."))
            ("concurrentWorkerCount", "Maximum concurrent inference, might be helpful when processing more than 1 audio file at the same time. 0 means unlimited. Default: 4", cxxopts::value<int>()->default_value("4"))
            ("v,verbose", "Verbose mode for debug", cxxopts::value<bool>()->default_value("false"));

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        
        WhisperKitConfig config;

        if (result.count("audioPath")) {
            config.audioPath = result["audioPath"].as<std::string>();
        }

        if (result.count("modelPath")) {
            config.modelPath = result["modelPath"].as<std::string>();
        }
        if (result.count("audioEncoderComputeUnits")) {
            config.audioEncoderComputeUnits = result["audioEncoderComputeUnits"].as<std::string>();
        }
        if (result.count("textDecoderComputeUnits")) {
            config.textDecoderComputeUnits = result["textDecoderComputeUnits"].as<std::string>();
        }
        if (result.count("temperature")) {
            config.temperature = result["temperature"].as<float>();
        }
        if (result.count("temperatureIncrementOnFallback")) {
            config.temperatureIncrementOnFallback = result["temperatureIncrementOnFallback"].as<float>();
        }
    
        if (result.count("temperatureFallbackCount")) {
            config.temperatureFallbackCount = result["temperatureFallbackCount"].as<int>();
        }
        if (result.count("bestOf")) {
            config.bestOf = result["bestOf"].as<int>();
        }
        if (result.count("skipSpecialTokens")) {
            config.skipSpecialTokens = result["skipSpecialTokens"].as<bool>();
        }
        if (result.count("withoutTimestamps")) {
            config.withoutTimestamps = result["withoutTimestamps"].as<bool>();
        }
        if (result.count("wordTimestamps")) {
            config.wordTimestamps = result["wordTimestamps"].as<bool>();
        }
        if (result.count("logprobThreshold")) {
            config.logprobThreshold = result["logprobThreshold"].as<float>();
        }
        if (result.count("firstTokenLogProbThreshold")) {
            config.firstTokenLogProbThreshold = result["firstTokenLogProbThreshold"].as<float>();
        }
        if (result.count("noSpeechThreshold")) {
            config.noSpeechThreshold = result["noSpeechThreshold"].as<float>();
        }
        if (result.count("report")) {
            config.report = result["report"].as<bool>();
        }
        if (result.count("reportPath")) {
            config.reportPath = result["reportPath"].as<std::string>();
        }
        if (result.count("concurrentWorkerCount")) {
            config.concurrentWorkerCount = result["concurrentWorkerCount"].as<int>();
        }
        if (result["verbose"].as<bool>()) {
            std::cout << "Verbose mode is ON." << std::endl;
        }
        

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

