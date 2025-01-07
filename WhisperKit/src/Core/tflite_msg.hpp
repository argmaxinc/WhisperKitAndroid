//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <string>
#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>

#ifdef ANDROID_JNI
#include <android/log.h>

constexpr const char* ARGMAX_WHISPERKIT_BUNDLE_INFO = "com.argmax.whisperax";
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  ARGMAX_WHISPERKIT_BUNDLE_INFO, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, ARGMAX_WHISPERKIT_BUNDLE_INFO, __VA_ARGS__)

#else

#define LOGI(...) fprintf(stdout, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)

#endif

class TFLiteMessenger{
public:
    TFLiteMessenger() {
        _msg = std::make_unique<std::string>();
    }
    std::mutex _mutex;
    std::condition_variable _cond_var;

    bool _running = false;
    float _timestamp = 0;
    std::unique_ptr<std::string> _msg;

    ~TFLiteMessenger() {
        _msg.reset();
    }

    std::string get_message() {
        return std::string(*(_msg.get()));
    }

    void print(){
        if (_msg->empty()) {
            return;
        }
        if (!_running) {
            LOGI("\nFinal Text: %s\n", _msg.get()->c_str());
        } else {
            LOGI("\nText: %s\n", _msg.get()->c_str());
        }
    };
};
