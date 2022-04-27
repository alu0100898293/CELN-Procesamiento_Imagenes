cmake_minimum_required(VERSION 3.16)
project( BorderDetectionSeq )

set( CMAKE_XX_STANDARD 14)

#look for SFML and add it
find_package(SFML 2.5 
COMPONENTS 
  system window graphics network audio REQUIRED)

add_executable( main 
  source/main.cpp
)

target_include_directories( main PRIVATE
  source
)

target_link_libraries( main sfml-graphics)

