add_library(opencv_io SHARED opencv_io.cpp)


link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/)

set_target_properties(opencv_io PROPERTIES
                      DEFINE_SYMBOL JANUS_LIBRARY
                      VERSION ${JANUS_VERSION_MAJOR}.${JANUS_VERSION_MINOR}.${JANUS_VERSION_PATCH}
                      SOVERSION ${JANUS_VERSION_MAJOR}.${JANUS_VERSION_MINOR})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/include/)


set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_core.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_highgui.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_videoio.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_imgcodecs.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_video.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_imgproc.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_videostab.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_objdetect.so")

install(TARGETS opencv_io RUNTIME DESTINATION bin
                          LIBRARY DESTINATION lib
                          ARCHIVE DESTINATION lib)

# Add this to the list of implementations to test
option(JANUS_TEST_OPENCV_IO "Test the OpenCV implementation of the Janus I/O API" ON)
if(${JANUS_TEST_OPENCV_IO})
  set(JANUS_IO_TEST_IMPLEMENTATIONS ${JANUS_IO_TEST_IMPLEMENTATIONS} "opencv_io" PARENT_SCOPE)
endif()
