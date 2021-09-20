cmake_minimum_required(VERSION 3.7.2)
project(nms VERSION 0.1.0)

#include(CTest)
#enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package (Threads REQUIRED)
find_package(OpenMP REQUIRED)
include_directories("3rd_party")
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
add_executable(tests tests.cpp nms.cpp reference.cpp ipu_utils.cpp)
target_link_libraries (tests PUBLIC Threads::Threads OpenMP::OpenMP_CXX -lpopnn -lpoplar -lpoputil -lpopops -lpoplin)
##add_executable(ipu_main ipu_main.cpp lbm.cpp reference.cpp ipu_utils.cpp)
#target_link_libraries (ipu_main PUBLIC Threads::Threads OpenMP::OpenMP_CXX -lpoplar -lpoputil -lpopops -lpoplin)

add_executable(benchmark benchmark.cpp nms.cpp ipu_utils.cpp reference.cpp)
target_link_libraries (benchmark PRIVATE Threads::Threads -lpoplar -lpopnn -lpoputil -lpopops)

#add_executable(benchmarks benchmarks.cpp ipu_utils.cpp topkmerge.cpp)
#target_link_libraries (benchmarks PRIVATE Threads::Threads -lpoplar -lpopnn -lpoputil -lpopops)
#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)

list(APPEND POPC_ENVIRONMENT
  "ASAN_OPTIONS=detect_leaks=0,halt_on_error=0"
  "MSAN_OPTIONS=halt_on_error=0"
)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -g -DENABLE_POPLAR_RUNTIME_CHECKS=1)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  list(APPEND POPC_FLAGS -O3 -g)
else()
  list(APPEND POPC_FLAGS -O3 --target=ipu2)
endif()
list(APPEND POPC_FLAGS -Werror -Wdouble-promotion)

set(CODELETS
  "${CMAKE_SOURCE_DIR}/codelet.cpp")
set(CODELETS_GP
  "${CMAKE_SOURCE_DIR}/codelet.gp")

add_custom_command(OUTPUT "${CMAKE_SOURCE_DIR}/codelet.gp"
  COMMAND popc ${POPC_FLAGS} ${CODELETS} -o ${CODELETS_GP}
  DEPENDS ${CODELETS}
  COMMENT "Compiling codelets."
  VERBATIM)

add_custom_target(codelet_gp ALL DEPENDS ${CODELETS_GP})

add_library(codelet INTERFACE)
add_dependencies(codelet codelet_gp)
