cmake_minimum_required(VERSION 3.16)
project( BorderDetectionOpenmpCollapse )

set( CMAKE_XX_STANDARD 14)

#look for SFML and add it
find_package(SFML 2.5 
COMPONENTS 
  system window graphics network audio REQUIRED)

find_package(OpenMP)
  
add_executable( main 
  source/main.cpp
)

target_include_directories( main PRIVATE
  source
)

target_link_libraries( main sfml-graphics)

if(OpenMP_CXX_FOUND)
    target_link_libraries( main OpenMP::OpenMP_CXX)
endif()