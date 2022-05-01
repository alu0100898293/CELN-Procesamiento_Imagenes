cmake_minimum_required(VERSION 3.16)

set( CMAKE_XX_STANDARD 14)

project( BorderDetectionSeq LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_CXX_FLAGS "/Qstd=c++11")

include(CheckLanguage)
check_language(CUDA)

#look for SFML and add it
find_package(SFML 2.5 
COMPONENTS 
  system window graphics network audio REQUIRED)

add_executable( main 
  source/main.cu
)

target_include_directories( main PRIVATE
  source
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_package(Thrust REQUIRED CONFIG )
thrust_create_target(Thrust)

target_link_libraries( main sfml-graphics Thrust)

