cmake_minimum_required(VERSION 3.16)
project( BorderDetectionMpi LANGUAGES CXX)

find_package( MPI REQUIRED COMPONENTS CXX)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

#look for SFML and add it
find_package(SFML 2.5 
COMPONENTS 
  system window graphics network audio REQUIRED)

add_executable( main 
  source/main.cpp 
  source/MpiFunctions.cpp
)

target_include_directories( main PRIVATE
  source
  include
)

target_link_libraries( main ${MPI_CXX_LIBRARIES})
target_link_libraries( main sfml-graphics)

if(MPI_COMPILE_FLAGS)
  set_target_properties( main PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties( main PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
