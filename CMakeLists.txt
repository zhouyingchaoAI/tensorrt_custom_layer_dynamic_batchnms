cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(NMSPlugin LANGUAGES CXX CUDA)

set(PLUGIN_EXPORT_MAP ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

set_ifndef(CUDA_VERSION 10.2)
set_ifndef(CUDA_ROOT /usr/local/cuda-${CUDA_VERSION})
set_ifndef(CUDNN_ROOT ${CUDA_ROOT})
set_ifndef(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR /usr/include/x86_64-linux-gnu)
set_ifndef(CUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub)

# Find dependencies.
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

find_library(_CUDA_LIB cudart HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

find_library(_CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

find_path(_CUDNN_INC_DIR cudnn.h HINTS ${CUDNN_ROOT} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(CUDNN_INC_DIR ${_CUDNN_INC_DIR})
    
find_library(_CUDNN_LIB cudnn HINTS ${CUDNN_ROOT} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(CUDNN_LIB ${_CUDNN_LIB})
    
find_library(_TRT_INC_DIR NvInfer.h HINTS ${TRT_INC_DIR} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR ${_TRT_INC_DIR})

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVPARSERS_LIB nvparsers HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVPARSERS_LIB ${_NVPARSERS_LIB})

find_library(_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_PLUGIN_LIB ${_NVINFER_PLUGIN_LIB})

# -------- BUILDING --------


set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    batchedNMSPlugin
    )


# Add include directories
include_directories(${TRT_INC_DIR} ${CUDA_INC_DIR} ${CUDNN_INC_DIR})
message(${CUB_DIR})
include_directories(${CUB_DIR})
include_directories(common common/kernels samples/common)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)



list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

add_subdirectory(common)

list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

#list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InferPlugin.cpp")
list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/samples/common/logger.cpp")




add_library(${PROJECT_NAME} SHARED
    ${PLUGIN_SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")

#message(${PLUGIN_SOURCES})
#message(${NVINFER_PLUGIN_LIB} ${CUDA_LIB} ${CUDNN_LIB} ${CUBLAS_LIB} ${NVINFER_LIB})

target_link_libraries(${PROJECT_NAME}  ${NVINFER_PLUGIN_LIB} ${CUDA_LIB} ${CUDNN_LIB} ${CUBLAS_LIB} ${NVINFER_LIB})
