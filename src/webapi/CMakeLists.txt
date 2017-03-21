cmake_minimum_required(VERSION 2.8.3)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
if(APPLE AND ${PYTHON_VERSION_MAJOR} EQUAL 3)
    FIND_PACKAGE(Boost COMPONENTS python3)
else()
    FIND_PACKAGE(Boost COMPONENTS python)
endif()

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
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
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

PYTHON_ADD_MODULE(janus_web_api janus_web_api.cpp)
TARGET_LINK_LIBRARIES(janus_web_api ${JANUS_IMPLEMENTATION} ${JANUS_IO_IMPLEMENTATION})
FILE(COPY test.py DESTINATION .)
FILE(COPY example.jpeg DESTINATION .)
