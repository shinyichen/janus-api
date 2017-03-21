link_directories(
	/nfs/isicvlnas01/share/opencv-3.1.0/lib/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/yolo/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/atlas
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/caffe/build/lib
	${CMAKE_SOURCE_DIR}/lib
	${CMAKE_SOURCE_DIR}/../../lib
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/OpenMesh-3.2/build/Build/lib/OpenMesh/
)


if(NOT ${JANUS_IMPLEMENTATION} STREQUAL "")
  file(GLOB UTILS *.cpp)
  foreach(UTIL ${UTILS})
    get_filename_component(UTIL_NAME ${UTIL} NAME_WE)
    add_executable(${UTIL_NAME} ${UTIL} ${JANUS_HEADERS})

    target_link_libraries(${UTIL_NAME} ${JANUS_IMPLEMENTATION} ${JANUS_IO_IMPLEMENTATION})
    install(TARGETS ${UTIL_NAME} RUNTIME DESTINATION bin)
  endforeach()
endif()
