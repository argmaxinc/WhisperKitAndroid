# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.22.1)

# project name and language
project(whisperkit LANGUAGES CXX)

if (NOT DEFINED TENSORFLOW_SOURCE_DIR)
    message(FATAL_ERROR  "TensorFlow path is not defined. Please use cmake -DTENSORFLOW_SOURCE_DIR=/path/to/tensorflow")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_BUILD_TYPE Release)

set(WHISPERKIT_SRC_DIR ${CMAKE_SOURCE_DIR}/WhisperKit/src/)
set(EXT_INC_DIR ${CMAKE_SOURCE_DIR}/external/inc)
set(WHISPERKIT_CLI_DIR ${CMAKE_SOURCE_DIR}/cli)
set(WHISPERKIT_JNI_DIR ${CMAKE_SOURCE_DIR}/jni)
set(WHISPERKIT_PUBLIC_HEADERS ${CMAKE_SOURCE_DIR}/WhisperKit/)
set(WHISPERKIT_EXTERNAL_HEADERS ${CMAKE_SOURCE_DIR}/WhisperKit/external/)

file(GLOB_RECURSE WHISPERKIT_SOURCES ${WHISPERKIT_SRC_DIR}/*.cpp)

set(WHISPERKIT_INTERNAL_HEADERS_DIRECTORIES
  ${WHISPERKIT_SRC_DIR}/Core/
  ${WHISPERKIT_SRC_DIR}/Models/
  ${WHISPERKIT_SRC_DIR}/Text/
  ${WHISPERKIT_SRC_DIR}/Audio/
  ${WHISPERKIT_SRC_DIR}/
)

# QNN delegate legal only on Android with QCOM SoC;
# GPU delegate legal on all Android;
# linux configuration currently supports CPU only.
OPTION(QNN_DELEGATE "QNN delegate" OFF) # Disabled by default
OPTION(GPU_DELEGATE "GPU delegate" OFF) # Disabled by default
OPTION(JNI_BUILD "Build for JNI" OFF)
SET(DELEGATE_ALLOWED_VALUES 0 1)

if(${ANDROID})
  message(NOTICE  "Config build for Android ARM64..")
  set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake)
  set(ANDROID_ABI arm64-v8a)
  set(EXT_LIB_DIR ${CMAKE_SOURCE_DIR}/external/libs/android)
  find_library(ANDROID_LOG log)
else()
  message(NOTICE  "Config build for linux..")
  set(EXT_LIB_DIR ${CMAKE_SOURCE_DIR}/external/libs/linux)
endif()

link_directories(
  ${EXT_LIB_DIR}
)

# build top library: whisperax
if(${ANDROID})
  if(${QNN_DELEGATE}) # QCOM QNN delegate
    message(NOTICE  " Delegate via QCOM QNN..")
    SET(QNN_DELEGATE 1)
    SET(GPU_DELEGATE 1)
  else() # generic TFLite GPU delegate
    message(NOTICE  " Delegate via Generic TFLite GPU..")
    SET(QNN_DELEGATE 0)
    SET(GPU_DELEGATE 1)
  endif()
else() # other generic arch, CPU only
  message(NOTICE  " Generic CPU only..")
  SET(QNN_DELEGATE 0)
  SET(GPU_DELEGATE 0)
endif()

if(${JNI})
  SET(JNI_BUILD 1)
else()
  SET(JNI_BUILD 0)
endif()

# Delegates (QNN, GPU, ...) are required to be defined, and values are required to be 0 or 1
if(NOT (DEFINED QNN_DELEGATE AND QNN_DELEGATE IN_LIST DELEGATE_ALLOWED_VALUES))
    message(FATAL_ERROR "QNN_DELEGATE must be defined and set to either 0 or 1. Current value: '${QNN_DELEGATE}'")
endif()

if(NOT (DEFINED GPU_DELEGATE AND GPU_DELEGATE IN_LIST DELEGATE_ALLOWED_VALUES))
    message(FATAL_ERROR "GPU_DELEGATE must be defined and set to either 0 or 1. Current value: '${GPU_DELEGATE}'")
endif()

add_library(whisperkit   SHARED
    ${WHISPERKIT_SOURCES}
)

target_include_directories(whisperkit
  PUBLIC ${WHISPERKIT_PUBLIC_HEADERS}
  PRIVATE ${WHISPERKIT_INTERNAL_HEADERS_DIRECTORIES}
  PRIVATE ${WHISPERKIT_EXTERNAL_HEADERS}
  PRIVATE ${EXT_INC_DIR}
  PRIVATE ${TENSORFLOW_SOURCE_DIR}
)

if (${JNI})
  message(NOTICE "Building for JNI")
  add_subdirectory(${SDL3_DIR})
  add_library(native-whisper SHARED
    ${WHISPERKIT_JNI_DIR}/NativeWhisperKit.cpp
    ${WHISPERKIT_CLI_DIR}/whisperkit_cli.cpp

  )

  target_include_directories(native-whisper
    PUBLIC ${WHISPERKIT_PUBLIC_HEADERS}
    PRIVATE ${WHISPERKIT_CLI_DIR}/external/
    PRIVATE ${WHISPERKIT_CLI_DIR}
  )
endif()


set(AUDIO_CODEC_LIBRARIES avformat avcodec avutil swresample)
if (${JNI})
  set(BASE_LIBRARIES tensorflowlite SDL3::SDL3-shared)
else()
  set(BASE_LIBRARIES tensorflowlite SDL3)
endif()
set(ALL_LINKED_LIBRARIES ${BASE_LIBRARIES} ${AUDIO_CODEC_LIBRARIES})

if(${ANDROID})

  if(${QNN_DELEGATE})
    add_library(qnn_delegate SHARED IMPORTED)
    set_property(TARGET qnn_delegate PROPERTY IMPORTED_LOCATION
      ${EXT_LIB_DIR}/libQnnTFLiteDelegate.so
    )

    list(APPEND ALL_LINKED_LIBRARIES qnn_delegate tensorflowlite_gpu_delegate ${ANDROID_LOG})

  else()

    list(APPEND ALL_LINKED_LIBRARIES tensorflowlite_gpu_delegate ${ANDROID_LOG})

  endif()
endif()

ADD_DEFINITIONS(-DGPU_DELEGATE=${GPU_DELEGATE} -DQNN_DELEGATE=${QNN_DELEGATE} -DJNI_BUILD=${JNI_BUILD})

target_link_libraries(whisperkit PRIVATE ${ALL_LINKED_LIBRARIES})

if(${JNI})
  target_link_libraries(native-whisper PRIVATE whisperkit)
  target_compile_definitions(native-whisper PRIVATE ANDROID_JNI)
endif()


execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_executable(whisperkit-cli
    ${WHISPERKIT_CLI_DIR}/whisperkit_cli.cpp
)

target_include_directories(whisperkit-cli PRIVATE
  ${WHISPERKIT_PUBLIC_HEADERS}
  ${WHISPERKIT_CLI_DIR}/external/
  ${WHISPERKIT_CLI_DIR}
)

target_link_libraries(whisperkit-cli PRIVATE
  whisperkit
)

target_compile_definitions(whisperkit-cli PRIVATE
  "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\""
)
