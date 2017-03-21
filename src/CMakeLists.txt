include_directories(BEFORE ../include)
file(GLOB JANUS_HEADERS ../include/*.h)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(${JANUS_IMPLEMENTATION} STREQUAL pittpatt)
  find_package(PP5 REQUIRED)
  add_subdirectory(pittpatt)
endif()

add_subdirectory(glaive)

if(${JANUS_IO_IMPLEMENTATION} STREQUAL opencv_io)
    add_subdirectory(opencv_io)
endif()

# Janus API documentation
add_subdirectory(doxygen)

# Janus command line utilities
add_subdirectory(utils)

add_subdirectory(cserver)
