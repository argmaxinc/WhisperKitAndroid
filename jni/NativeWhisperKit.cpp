#include <jni.h>
#include <string>
#include "../WhisperKit/WhisperKit.h"
#include "../cli/whisperkit_cli.h"
#include <android/log.h>

#define LOG_TAG "NativeWhisperKit"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Wrapper class to manage native resources
class NativeWhisperKit {
public:
    WhisperKitRunner* runner;
    WhisperKitConfig config;

    NativeWhisperKit(const std::string& modelPath,
                    const std::string& audioPath, 
                    const std::string& reportPath,
                    bool enableReport,
                    int concurrentWorkers) {
       LOGI("Entering NativeWhisperKit constructor");
        config.modelPath = modelPath;
        config.reportPath = reportPath;
        config.report = enableReport;
        config.concurrentWorkerCount = concurrentWorkers;
        config.audioPath = audioPath;

        LOGI("ModelPath: %s, ReportPath: %s, Report: %d, Workers: %d",
         config.modelPath.c_str(), config.reportPath.c_str(), config.report, config.concurrentWorkerCount);
        
        runner = new WhisperKitRunner(config);
        LOGI("WhisperKitRunner created successfully");

        try {
            LOGI("Calling runner->buildPipeline()");
            runner->buildPipeline();
            LOGI("buildPipeline() completed successfully");
        } catch (const std::exception& e) {
            LOGE("Exception caught in buildPipeline(): %s", e.what());
            delete runner;
            throw;
        }
        LOGI("Exiting NativeWhisperKit constructor");
    }

    std::string transcribe(const std::string& audioPath) {
        config.audioPath = audioPath;
        try {
            runner->transcribe();
            char* result;
            whisperkit_transcription_result_get_transcription(runner->transcriptionResult, &result);
            return std::string(result);
        } catch (const std::exception& e) {
            throw;
        }
    }

    ~NativeWhisperKit() {
        if (runner) {
            delete runner;
        }
    }
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_whispertflite_WhisperKitNative_init(
    JNIEnv* env,
    jobject thiz,
    jstring modelPath,
    jstring audioPath,
    jstring reportPath,
    jboolean enableReport,
    jint concurrentWorkers) {
    
    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    const char* audioPathStr = env->GetStringUTFChars(audioPath, nullptr);
    const char* reportPathStr = env->GetStringUTFChars(reportPath, nullptr);
    
    try {
        auto nativeInst = new NativeWhisperKit(
            std::string(modelPathStr),
            std::string(audioPathStr),
            std::string(reportPathStr),
            enableReport,
            concurrentWorkers
        );
        
        env->ReleaseStringUTFChars(modelPath, modelPathStr);
        env->ReleaseStringUTFChars(audioPath, audioPathStr);
        env->ReleaseStringUTFChars(reportPath, reportPathStr);
        
        return reinterpret_cast<jlong>(nativeInst);
    } catch (const std::exception& e) {
        env->ReleaseStringUTFChars(modelPath, modelPathStr);
        env->ReleaseStringUTFChars(audioPath, audioPathStr);
        env->ReleaseStringUTFChars(reportPath, reportPathStr);
        env->ThrowNew(env->FindClass("java/io/IOException"), e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_whispertflite_WhisperKitNative_transcribe(
    JNIEnv* env,
    jobject thiz,
    jlong nativePtr,
    jstring audioPath) {
    
    auto nativeInst = reinterpret_cast<NativeWhisperKit*>(nativePtr);
    if (!nativeInst) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), 
                     "Instance not initialized");
        return nullptr;
    }
    
    const char* audioPathStr = env->GetStringUTFChars(audioPath, nullptr);
    
    try {
        std::string result = nativeInst->transcribe(std::string(audioPathStr));
        env->ReleaseStringUTFChars(audioPath, audioPathStr);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        env->ReleaseStringUTFChars(audioPath, audioPathStr);
        env->ThrowNew(env->FindClass("java/io/IOException"), e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_whispertflite_WhisperKitNative_release(
    JNIEnv* env,
    jobject thiz,
    jlong nativePtr) {
    
    auto nativeInst = reinterpret_cast<NativeWhisperKit*>(nativePtr);
    if (nativeInst) {
        delete nativeInst;
    }
}

// UNCOMMENT IF YOU WANT TO BUILD THE CLI EXECUTABLE

// int main(int argc, char* argv[]) {
//     std::string modelPath;
//     std::string audioPath;
//     try {
//         cxxopts::Options options("native-whisperkit", "WhisperKit CLI for Android & Linux");

//         options.add_options("transcribe")
//             ("h,help", "Print help")
//             ("audio-path", "Path to audio file", cxxopts::value<std::string>())
//             ("model-path", "Path of model files", cxxopts::value<std::string>())
//             ("report","Output a report of the results", cxxopts::value<bool>()->default_value("false"))
//             ("report-path", "Directory to save the report", cxxopts::value<std::string>()->default_value("."))
//             ("v,verbose", "Verbose mode for debug", cxxopts::value<bool>()->default_value("false"));

//         auto result = options.parse(argc, argv);

//         if (result.count("help")) {
//             std::cout << options.help() << std::endl;
//             return 0;
//         }

//         if (result.count("audio-path")) {
//             audioPath = result["audio-path"].as<std::string>();
//         }
//         if (result.count("model-path")) {
//             modelPath = result["model-path"].as<std::string>();
//         }
//         // if (result.count("report")) {
//         //     config.report = result["report"].as<bool>();
//         // }
//         // if (result.count("report-path")) {
//         //     config.reportPath = result["report-path"].as<std::string>();
//         // }
//         if (result["verbose"].as<bool>()) {
//             std::cout << "Verbose mode is ON." << std::endl;
//         }
        

//     } catch (const cxxopts::exceptions::exception& e) {
//         std::cerr << "Error parsing options: " << e.what() << std::endl;
//         return 1;
//     }

//     auto nativeInst = new NativeWhisperKit(
//             std::string(modelPath),
//             std::string(audioPath),
//             ".",
//             false,
//             4
//         );
//      try {
//         std::string result = nativeInst->transcribe(std::string(audioPath));
//         std::cout << result << std::endl;
//         delete nativeInst;
//         return 0;
//     } catch (const std::exception& e) {
//         std::cerr << "Exception while transcribing: " << e.what() << std::endl;
//         delete nativeInst;
//         return -1;
//     }
// }