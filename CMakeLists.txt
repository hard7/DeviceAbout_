cmake_minimum_required(VERSION 3.4)
project(DeviceAbout)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(DeviceAbout)
set(SOURCE_FILES main.cpp)

add_executable(DeviceAboutMain ${SOURCE_FILES})
target_link_libraries(DeviceAboutMain DeviceAbout)