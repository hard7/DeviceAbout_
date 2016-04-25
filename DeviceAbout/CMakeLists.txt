

find_package(CUDA REQUIRED)

#message(FATAL_ERROR ${CUDA_FOUND})
#add_library(DeviceAbout STATIC impl/DeviceAbout.cpp)
set(CUDA_HELPER_LOCATE ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
include_directories(${CUDA_HELPER_LOCATE})


CUDA_ADD_LIBRARY(DeviceAbout STATIC impl/DeviceAbout.cpp)
target_link_libraries(DeviceAbout cuda)