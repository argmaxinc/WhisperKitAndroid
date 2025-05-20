#include "whisperkit_jni.h"

#include <android/log.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Simple JSON parser
class SimpleJson {
   public:
    std::map<std::string, std::string> values;

    static SimpleJson parse(const std::string& jsonStr) {
        SimpleJson json;

        size_t pos = 0;
        while (pos < jsonStr.length()) {
            // Find key start
            pos = jsonStr.find("\"", pos);
            if (pos == std::string::npos) break;

            // Extract key
            size_t keyStart = pos + 1;
            pos = jsonStr.find("\"", keyStart);
            if (pos == std::string::npos) break;

            std::string key = jsonStr.substr(keyStart, pos - keyStart);

            // Find value
            pos = jsonStr.find(":", pos);
            if (pos == std::string::npos) break;
            pos++;

            // Skip whitespace
            while (pos < jsonStr.length() && isspace(jsonStr[pos])) pos++;

            std::string value;
            if (pos < jsonStr.length()) {
                if (jsonStr[pos] == '\"') {
                    // String value
                    size_t valueStart = pos + 1;
                    pos = jsonStr.find("\"", valueStart);
                    if (pos == std::string::npos) break;

                    value = jsonStr.substr(valueStart, pos - valueStart);
                    pos++;
                } else if (isdigit(jsonStr[pos]) || jsonStr[pos] == '-' || jsonStr[pos] == '+') {
                    // Number value
                    size_t valueStart = pos;
                    while (pos < jsonStr.length() &&
                           (isdigit(jsonStr[pos]) || jsonStr[pos] == '.' || jsonStr[pos] == '-' ||
                            jsonStr[pos] == '+' || jsonStr[pos] == 'e' || jsonStr[pos] == 'E')) {
                        pos++;
                    }

                    value = jsonStr.substr(valueStart, pos - valueStart);
                } else if (jsonStr.compare(pos, 4, "true") == 0) {
                    value = "true";
                    pos += 4;
                } else if (jsonStr.compare(pos, 5, "false") == 0) {
                    value = "false";
                    pos += 5;
                } else if (jsonStr.compare(pos, 4, "null") == 0) {
                    value = "null";
                    pos += 4;
                }

                json.values[key] = value;
            }

            // Find next key-value pair
            pos = jsonStr.find(",", pos);
            if (pos == std::string::npos) break;
            pos++;
        }

        return json;
    }

    std::string get(const std::string& key, const std::string& defaultValue = "") const {
        auto it = values.find(key);
        if (it != values.end()) {
            return it->second;
        }
        return defaultValue;
    }

    int getInt(const std::string& key, int defaultValue = 0) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    long getLong(const std::string& key, long defaultValue = 0) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stol(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    double getDouble(const std::string& key, double defaultValue = 0.0) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    bool getBool(const std::string& key, bool defaultValue = false) const {
        auto it = values.find(key);
        if (it != values.end()) {
            return it->second == "true";
        }
        return defaultValue;
    }

    std::string toJsonString() const {
        std::stringstream ss;
        ss << "{";
        bool first = true;

        for (const auto& pair : values) {
            if (!first) {
                ss << ",";
            }
            first = false;

            ss << "\"" << pair.first << "\":";

            // Check if the value is a number, boolean, or null
            if (pair.second == "true" || pair.second == "false" || pair.second == "null" ||
                (pair.second.find_first_not_of("-+.0123456789eE") == std::string::npos)) {
                ss << pair.second;
            } else {
                ss << "\"" << pair.second << "\"";
            }
        }

        ss << "}";
        return ss.str();
    }
};

// Global state
static struct {
    // Configuration
    std::string modelPath;
    std::string reportPath;
    std::string libDir;
    std::string cacheDir;
    std::string modelSize;
    int sampleRate;
    int channels;
    long duration;

    // WhisperKit objects
    whisperkit_configuration_t* config;
    whisperkit_pipeline_t* pipeline;
    whisperkit_transcription_result_t* result;

    std::mutex audioMutex;

    // Java references
    JavaVM* javaVM;
    jobject mainActivity;
    jmethodID onTextOutputMethod;

    // Performance metrics
    std::map<std::string, SimpleJson> perfMetrics;
    long startTime;
    unsigned int appended_bytes;
    int encoder_backend;
    int decoder_backend;
} g_state;

// Helper function to get JNI environment
JNIEnv* getEnv() {
    JNIEnv* env;
    if (g_state.javaVM->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        LOGE("Failed to get JNI environment");
        return nullptr;
    }
    return env;
}

// Callback to send text to Java
void sendTextToJava(JNIEnv* env, jobject thiz, CallbackMsgType what, jfloat timestamp, const char* text) {
    if (!env || !thiz) {
        LOGE("Invalid environment or object for callback");
        return;
    }

    jstring jText = env->NewStringUTF(text);
    env->CallVoidMethod(thiz, g_state.onTextOutputMethod, what, timestamp, jText);
    env->DeleteLocalRef(jText);
}

// Initialize WhisperKit with configuration from JSON
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_loadModels(JNIEnv* env, jobject thiz,
                                                                               jstring jsonstr) {
    // Store Java VM and MainActivity reference
    env->GetJavaVM(&g_state.javaVM);
    g_state.mainActivity = env->NewGlobalRef(thiz);

    // Get the onTextOutput method ID
    jclass cls = env->GetObjectClass(thiz);
    g_state.onTextOutputMethod = env->GetMethodID(cls, "onTextOutput", "(IFLjava/lang/String;)V");

    if (!g_state.onTextOutputMethod) {
        LOGE("Failed to get onTextOutput method");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_state.audioMutex);

    // Parse JSON configuration
    const char* jsonString = env->GetStringUTFChars(jsonstr, nullptr);
    SimpleJson jsonConfig = SimpleJson::parse(jsonString);
    env->ReleaseStringUTFChars(jsonstr, jsonString);

    // Extract configuration values
    g_state.sampleRate = jsonConfig.getInt("freq");
    g_state.channels = jsonConfig.getInt("ch");
    g_state.duration = jsonConfig.getLong("dur");
    g_state.modelSize = jsonConfig.get("size");
    g_state.libDir = jsonConfig.get("lib");
    g_state.cacheDir = jsonConfig.get("cache");
    g_state.encoder_backend = jsonConfig.getInt("encoder_backend");
    g_state.decoder_backend = jsonConfig.getInt("decoder_backend");

    g_state.modelPath = jsonConfig.get("model_path");
    g_state.reportPath = jsonConfig.get("report_path");

    LOGI("Model path: %s", g_state.modelPath.c_str());
    LOGI("Report path: %s", g_state.reportPath.c_str());
    LOGI("Lib dir: %s", g_state.libDir.c_str());
    LOGI("Cache dir: %s", g_state.cacheDir.c_str());

    // Notify initialization success
    sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f, "WhisperKit models loaded successfully");
    return 0;
}

// Initialize WhisperKit with configuration from JSON
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_init(JNIEnv* env, jobject thiz, jstring jsonstr) {
    // Store Java VM and MainActivity reference
    env->GetJavaVM(&g_state.javaVM);
    g_state.mainActivity = env->NewGlobalRef(thiz);

    // Get the onTextOutput method ID
    jclass cls = env->GetObjectClass(thiz);
    g_state.onTextOutputMethod = env->GetMethodID(cls, "onTextOutput", "(IFLjava/lang/String;)V");

    if (!g_state.onTextOutputMethod) {
        LOGE("Failed to get onTextOutput method");
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_state.audioMutex);

    // Parse JSON configuration
    const char* jsonString = env->GetStringUTFChars(jsonstr, nullptr);
    SimpleJson jsonConfig = SimpleJson::parse(jsonString);
    env->ReleaseStringUTFChars(jsonstr, jsonString);

    // Extract configuration values
    g_state.sampleRate = jsonConfig.getInt("freq");
    g_state.channels = jsonConfig.getInt("ch");
    g_state.duration = jsonConfig.getLong("dur");

    g_state.startTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    // Create WhisperKit configuration
    whisperkit_status_t status = whisperkit_configuration_create(&g_state.config);
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to create configuration: %d", status);
        sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f,
                       ("Failed to create configuration: " + std::to_string(status)).c_str());
        return -1;
    }

    // Set configuration options
    status = whisperkit_configuration_set_model_path(g_state.config, g_state.modelPath.c_str());
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to set model path: %d", status);
        sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f,
                       ("Failed to set model path: " + std::to_string(status)).c_str());
        return -1;
    }

    status = whisperkit_configuration_set_report_path(g_state.config, g_state.reportPath.c_str());
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to set report path: %d", status);
    }

    status = whisperkit_configuration_set_lib_dir(g_state.config, g_state.libDir.c_str());
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to set lib dir: %d", status);
    }

    status = whisperkit_configuration_set_cache_dir(g_state.config, g_state.cacheDir.c_str());
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to set cache dir: %d", status);
    }

    // Set additional configuration options
    whisperkit_configuration_set_verbose(g_state.config, true);
    whisperkit_configuration_set_log_level(g_state.config, 3);
    whisperkit_configuration_set_prewarm(g_state.config, true);
    whisperkit_configuration_set_load(g_state.config, true);

    status = whisperkit_configuration_set_backends(g_state.config, (whisperkit_backend_t)g_state.encoder_backend,
                                                   (whisperkit_backend_t)g_state.decoder_backend);

    // Create pipeline
    status = whisperkit_pipeline_create(&g_state.pipeline);
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to create pipeline: %d", status);
        sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f,
                       ("Failed to create pipeline: " + std::to_string(status)).c_str());
        return -1;
    }

    // Set pipeline configuration
    status = whisperkit_pipeline_set_configuration(g_state.pipeline, g_state.config);
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to set pipeline configuration: %d", status);
        sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f,
                       ("Failed to set pipeline configuration: " + std::to_string(status)).c_str());
        return -1;
    }

    // Build pipeline
    status = whisperkit_pipeline_build(g_state.pipeline);
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to build pipeline: %d", status);
        sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f,
                       ("Failed to build pipeline: " + std::to_string(status)).c_str());
        return -1;
    }

    // Create transcription result
    status = whisperkit_transcription_result_create(&g_state.result);
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE("Failed to create transcription result: %d", status);
        sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f,
                       ("Failed to create transcription result: " + std::to_string(status)).c_str());
        return -1;
    }

    g_state.appended_bytes = 0;
    // all transcribing via JNI is in streaming mode
    status = whisperkit_pipeline_initstreaming(g_state.pipeline, g_state.result, g_state.sampleRate, g_state.channels);

    // Notify initialization success
    sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f, "WhisperKit initialized successfully");
    LOGI("WhisperKit initialized");
    return 0;
}

// Close and clean up WhisperKit
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_close(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_state.audioMutex);

    whisperkit_status_t status;
    status = whisperkit_pipeline_closestreaming(g_state.pipeline);

    // Get final transcription
    char* finalTranscription = nullptr;
    status = whisperkit_transcription_result_get_transcription(g_state.result, &finalTranscription);

    if (status == WHISPERKIT_STATUS_SUCCESS && finalTranscription != nullptr && strlen(finalTranscription) > 0) {
        // Send final text to Java
        sendTextToJava(env, thiz, CallbackMsgType::CLOSE, 0.0f, finalTranscription);
    }

    // Clean up WhisperKit resources
    if (g_state.result) {
        whisperkit_transcription_result_destroy(&g_state.result);
    }

    if (g_state.pipeline) {
        whisperkit_pipeline_destroy(&g_state.pipeline);
    }

    if (g_state.config) {
        whisperkit_configuration_destroy(&g_state.config);
    }

    // Clean up Java resources
    if (g_state.mainActivity) {
        env->DeleteGlobalRef(g_state.mainActivity);
        g_state.mainActivity = nullptr;
    }

    // Record final duration in performance metrics
    long endTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    double duration = (endTime - g_state.startTime) / 1000.0;

    SimpleJson durationJson;
    durationJson.values["value"] = std::to_string(duration);
    g_state.perfMetrics["duration"] = durationJson;

    LOGI("WhisperKit closed");
    return 0;
}

// Write audio data to the buffer
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_writeData(JNIEnv* env, jobject thiz,
                                                                              jbyteArray pcmbuffer) {
    jboolean isCopy = JNI_TRUE;
    jsize num_bytes = env->GetArrayLength(pcmbuffer);
    jbyte* buffer = env->GetByteArrayElements(pcmbuffer, &isCopy);

    // Convert byte array to int16_t samples
    std::lock_guard<std::mutex> lock(g_state.audioMutex);

    g_state.appended_bytes += num_bytes;

    // in streaming mode: just pass the buffer pointer with its size
    int transcribed = 0;
    whisperkit_status_t status =
        whisperkit_pipeline_appendaudio(g_state.pipeline, num_bytes, reinterpret_cast<char*>(buffer), &transcribed);

    env->ReleaseByteArrayElements(pcmbuffer, buffer, 0);

    auto buffered_secs = (int)(g_state.appended_bytes / (g_state.sampleRate * g_state.channels * 2));
    if (transcribed == 1) {
        LOGI("** buffered_secs: %d, %d bytes\n", buffered_secs, g_state.appended_bytes);
        // Get any available results
        char* transcription = nullptr;
        whisperkit_status_t status = whisperkit_transcription_result_get_transcription(g_state.result, &transcription);
        if (status == WHISPERKIT_STATUS_SUCCESS && transcription != nullptr && strlen(transcription) > 0) {
            // Send text to Java
            sendTextToJava(env, thiz, CallbackMsgType::TEXT_OUT, 0.0f, transcription);
        }
    }

    return buffered_secs;
}

JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_setBackend(JNIEnv* env, jobject thiz,
                                                                               jint encoder_backend,
                                                                               jint decoder_backend) {
    std::lock_guard<std::mutex> lock(g_state.audioMutex);

    g_state.encoder_backend = encoder_backend;
    g_state.decoder_backend = decoder_backend;

    whisperkit_status_t status = whisperkit_configuration_set_backends(
        g_state.config, (whisperkit_backend_t)g_state.encoder_backend, (whisperkit_backend_t)g_state.decoder_backend);
    sendTextToJava(env, thiz, CallbackMsgType::INIT, 0.0f, "Backend is configured successfully");
    LOGI("Backend is configured successfully");
    return 0;
}

// JNI_OnLoad function
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    // Initialize global state
    g_state.javaVM = vm;
    g_state.mainActivity = nullptr;
    g_state.onTextOutputMethod = nullptr;
    g_state.config = nullptr;
    g_state.pipeline = nullptr;
    g_state.result = nullptr;

    return JNI_VERSION_1_6;
}

JNIEXPORT jstring JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_getPerfString(JNIEnv* env, jobject thiz) {
    // ... existing implementation ...
}
