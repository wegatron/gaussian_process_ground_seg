cmake_minimum_required(VERSION 3.10)
project(ground_seg)

include($ENV{HOME}/usr/share/cmake_modules/cmake_self.cmake)

# 设置编译输出的路径
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

set(app_name ground_seg)

add_definitions(-D_USE_MATH_DEFINES)

# find_package(Eigen3  REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})

# python3
set(ENV{PATH} "$ENV{PATH};/home/wegatron/opt/anaconda3/envs/pc_course/bin")
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#boost
find_package(Boost 1.65 COMPONENTS python3 numpy3 REQUIRED)
add_definitions(-DBOOST_SYSTEM_NO_DEPRECATED)
add_definitions(-DBOOST_PYTHON_STATIC_LIB) # 这里必须使用静态库
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(ZSW_LIB REQUIRED)
include_directories(${ZSW_LIB_INCLUDE_DIR})

file(GLOB headers *.h *.hpp)
file(GLOB src *.cpp)

PYTHON_ADD_MODULE(wegatron_ground_seg ${headers} ${src})

target_link_libraries(wegatron_ground_seg
      ${OpenCV_LIBS}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      ${PYTHON_LIBRARIES}
    )

#add_executable(${app_name}
#${headers}
#${src})

#target_link_libraries(${app_name}
#  ${ZSW_LIBRARIES}
#  ${OpenCV_LIBS}
#  ${PCL_LIBRARIES}
#  ${Boost_LIBRARIES}
#)
