cmake_minimum_required(VERSION 3.2)
project(TASORL LANGUAGES CXX)


file(GLOB_RECURSE TASORL_SRCS
  src/*.cc
  )

#find_package(Protobuf REQUIRED)
#message(STATUS "PROTOBUF=" ${PROTOBUF_LIBRARY})
#include_directories(${PROTOBUF_INCLUDE_DIR})
#protobuf_generate_cpp(PROTO_CPPS PROTO_HDRS src/rules.proto)
#list(APPEND TASORL_LINK_LIBS ${PROTOBUF_LIBRARY})

list(APPEND TASORL_LINK_LIBS "/usr/local/lib/libtaso_runtime.so")

add_library(taso_rl SHARED ${TASORL_SRCS})

target_link_libraries(taso_rl ${TASORL_LINK_LIBS})

set_target_properties(taso_rl
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS taso_rl
    LIBRARY DESTINATION lib)