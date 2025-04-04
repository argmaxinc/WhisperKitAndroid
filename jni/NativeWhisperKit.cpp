#ifdef ANDROID_JNI

#include <jni.h>
#include <string>
#include "../WhisperKit/WhisperKit.h"
#include "../cli/whisperkit_cli.h"

// Wrapper class to manage native resources
class NativeWhisperKit {
public:
    std::unique_ptr<WhisperKitRunner> runner;
    WhisperKitConfig config;

    NativeWhisperKit(const std::string& modelPath,
                    const std::string& audioPath, 
                    const std::string& reportPath,
                    const std::string& nativeLibsDir,
                    bool enableReport,
                    int concurrentWorkers) {
        config.modelPath = modelPath;
        config.reportPath = reportPath;
        config.report = enableReport;
        config.concurrentWorkerCount = concurrentWorkers;
        config.audioPath = audioPath;
        
        runner = std::make_unique<WhisperKitRunner>(config, nativeLibsDir);

        try {
            runner->buildPipeline();
        } catch (const std::exception& e) {
            throw;
        }
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
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_whispertflite_WhisperKitNative_init(
    JNIEnv* env,
    jobject thiz,
    jstring modelPath,
    jstring audioPath,
    jstring reportPath,
    jstring nativeLibsDir,
    jboolean enableReport,
    jint concurrentWorkers) {
    
    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    const char* audioPathStr = env->GetStringUTFChars(audioPath, nullptr);
    const char* reportPathStr = env->GetStringUTFChars(reportPath, nullptr);
    const char* nativeLibsDirStr = env->GetStringUTFChars(nativeLibsDir, nullptr);
    
    try {
        auto nativeInst = new NativeWhisperKit(
            std::string(modelPathStr),
            std::string(audioPathStr),
            std::string(reportPathStr),
            std::string(nativeLibsDirStr),
            enableReport,
            concurrentWorkers
        );
        
        env->ReleaseStringUTFChars(modelPath, modelPathStr);
        env->ReleaseStringUTFChars(audioPath, audioPathStr);
        env->ReleaseStringUTFChars(reportPath, reportPathStr);
        env->ReleaseStringUTFChars(nativeLibsDir, nativeLibsDirStr);
        
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

#endif