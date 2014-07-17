cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(microscopes_kernels)

set(CMAKE_CXX_FLAGS_BASE "${CMAKE_CXX_FLAGS} -fPIC -g -MD -Wall -std=c++0x")

if(APPLE)
  # clang complains about register
  set(CMAKE_CXX_FLAGS_BASE "${CMAKE_CXX_FLAGS_BASE} -Wno-deprecated-register")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_BASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_BASE} -DDEBUG_MODE")

find_package(Protobuf REQUIRED)
message(STATUS "found protobuf INC=${PROTOBUF_INCLUDE_DIRS}, LIB=${PROTOBUF_LIBRARY_DIRS}")
include_directories(${PROTOBUF_INCLUDE_DIRS})
link_directories(${PROTOBUF_LIBRARY_DIRS})

find_package(Distributions REQUIRED)
message(STATUS "found distributions INC=${DISTRIBUTIONS_INCLUDE_DIRS}, LIB=${DISTRIBUTIONS_LIBRARY_DIRS}")
include_directories(${DISTRIBUTIONS_INCLUDE_DIRS})
link_directories(${DISTRIBUTIONS_LIBRARY_DIRS})

find_package(MicroscopesCommon REQUIRED)
message(STATUS "found microscopes_common INC=${MICROSCOPES_COMMON_INCLUDE_DIRS}, LIB=${MICROSCOPES_COMMON_LIBRARY_DIRS}")
include_directories(${MICROSCOPES_COMMON_INCLUDE_DIRS})

find_package(MicroscopesMixturemodel REQUIRED)
message(STATUS "found microscopes_mixturemodel INC=${MICROSCOPES_MIXTUREMODEL_INCLUDE_DIRS}, LIB=${MICROSCOPES_MIXTUREMODEL_LIBRARY_DIRS}")
include_directories(${MICROSCOPES_MIXTUREMODEL_INCLUDE_DIRS})
link_directories(${MICROSCOPES_MIXTUREMODEL_LIBRARY_DIRS})
link_directories(${MICROSCOPES_MIXTUREMODEL_LIBRARY_DIRS})

include_directories(include)
install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY microscopes DESTINATION cython FILES_MATCHING PATTERN "*.pxd" PATTERN "__init__.py")

set(MICROSCOPES_KERNELS_SOURCE_FILES 
    src/kernels/bootstrap.cpp
    src/kernels/gibbs.cpp
    src/kernels/slice.cpp)
add_library(microscopes_kernels SHARED ${MICROSCOPES_KERNELS_SOURCE_FILES})
target_link_libraries(microscopes_kernels protobuf distributions_shared microscopes_common)
install(TARGETS microscopes_kernels LIBRARY DESTINATION lib)
