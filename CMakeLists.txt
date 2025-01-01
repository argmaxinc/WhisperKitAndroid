# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.22.1)

# project name and language
project(whisperax LANGUAGES CXX)

if (NOT DEFINED TENSORFLOW_SOURCE_DIR)
    message(FATAL_ERROR  "TensorFlow path is not defined. Please use cmake -DTENSORFLOW_SOURCE_DIR=/path/to/tensorflow")
endif()

set(CMAKE_CXX_STANDARD 23)
set(COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_BUILD_TYPE Release)

set(AXIE_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(AXIE_INC_DIR ${CMAKE_SOURCE_DIR}/inc)
set(EXT_INC_DIR ${CMAKE_SOURCE_DIR}/external/inc)
set(AXIE_CLI_DIR ${CMAKE_SOURCE_DIR}/cli)

if(${ANDROID})
  message(NOTICE  "Config build for Android ARM64..")
  OPTION(QNN_DELEGATE "QNN delegate" OFF) # Disabled by default
  set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake)
  set(ANDROID_ABI arm64-v8a)
  set(EXT_LIB_DIR ${CMAKE_SOURCE_DIR}/external/libs/android)
  find_library(android_log log)
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
    add_library(whisperax   SHARED
      ${AXIE_SRC_DIR}/tflite_model.cpp
      ${AXIE_SRC_DIR}/tflite_qnn_model.cpp
      ${AXIE_SRC_DIR}/post_proc.cpp
      ${AXIE_SRC_DIR}/audio_input.cpp
      ${AXIE_SRC_DIR}/whisperax.cpp
      ${AXIE_SRC_DIR}/WhisperKit.cpp
      ${AXIE_SRC_DIR}/WhisperKitConfiguration.cpp
      ${AXIE_SRC_DIR}/WhisperKitPipeline.cpp
    )
  else() # generic TFLite GPU delegate
    message(NOTICE  " Delegate via Generic TFLite GPU..")
    add_library(whisperax   SHARED
      ${AXIE_SRC_DIR}/tflite_model.cpp
      ${AXIE_SRC_DIR}/tflite_gpu_model.cpp
      ${AXIE_SRC_DIR}/post_proc.cpp
      ${AXIE_SRC_DIR}/audio_input.cpp
      ${AXIE_SRC_DIR}/whisperax.cpp
      ${AXIE_SRC_DIR}/WhisperKit.cpp
      ${AXIE_SRC_DIR}/WhisperKitConfiguration.cpp
      ${AXIE_SRC_DIR}/WhisperKitPipeline.cpp
    )

  endif()

else() # other generic arch, CPU only
  message(NOTICE  " Generic CPU only..")
  add_library(whisperax   SHARED
    ${AXIE_SRC_DIR}/tflite_model.cpp
    ${AXIE_SRC_DIR}/post_proc.cpp
    ${AXIE_SRC_DIR}/audio_input.cpp
    ${AXIE_SRC_DIR}/whisperax.cpp
    ${AXIE_SRC_DIR}/WhisperKit.cpp
    ${AXIE_SRC_DIR}/WhisperKitConfiguration.cpp
    ${AXIE_SRC_DIR}/WhisperKitPipeline.cpp
  )
endif()

target_include_directories(whisperax PRIVATE
  ${AXIE_INC_DIR}
  ${EXT_INC_DIR}
  ${TENSORFLOW_SOURCE_DIR}
)

if(${ANDROID})

  if(${QNN_DELEGATE})
    ADD_DEFINITIONS(-DQNN_DELEGATE)
    add_library(qnn_delegate SHARED IMPORTED)
    set_property(TARGET qnn_delegate PROPERTY IMPORTED_LOCATION
      ${EXT_LIB_DIR}/libQnnTFLiteDelegate.so
    )
    target_link_libraries(whisperax PRIVATE
      qnn_delegate
      tensorflowlite
      tensorflowlite_gpu_delegate
      ${android_log}
      SDL3
    )

  else()
    ADD_DEFINITIONS(-DGPU_DELEGATE)
    target_link_libraries(whisperax PRIVATE
      tensorflowlite
      tensorflowlite_gpu_delegate
      ${android_log}
      SDL3
    )
  endif()

else()
  target_link_libraries(whisperax PRIVATE
    tensorflowlite
    SDL3
  )
endif()

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# build CLI: whisperax_cli
add_executable(whisperax_cli
  ${AXIE_CLI_DIR}/whisperax_cli.cpp
  ${AXIE_CLI_DIR}/audio_codec.cpp
)

add_executable(whisperkit-cli
    ${AXIE_CLI_DIR}/whisperkit_cli.cpp
    ${AXIE_CLI_DIR}/audio_codec.cpp
)

target_include_directories(whisperax_cli PRIVATE
  ${AXIE_INC_DIR}
  ${AXIE_CLI_DIR}
  ${EXT_INC_DIR}
)

target_include_directories(whisperkit-cli PRIVATE
  ${AXIE_INC_DIR}
  ${AXIE_CLI_DIR}
  ${AXIE_SRC_DIR}/external/
  ${EXT_INC_DIR}
)

target_link_libraries(whisperax_cli PRIVATE
  whisperax
  avformat
  avcodec
  avutil
  swresample
)

target_link_libraries(whisperkit-cli PRIVATE
  whisperax
  avformat
  avcodec
  avutil
  swresample
)

target_compile_definitions(whisperax_cli PRIVATE
  "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\""
)

target_compile_definitions(whisperkit-cli PRIVATE
  "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\""
)
