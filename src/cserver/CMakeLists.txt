cmake_minimum_required(VERSION 2.8.3)

#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
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
#LINK_LIBRARIES(${Boost_LIBRARIES}) # Deprecated but so convenient!

#if (MSVC)
#set(Boost_USE_STATIC_LIBS "On")
#find_package( Boost 1.52 COMPONENTS system thread regex REQUIRED )
#else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++1y -pedantic -Wextra")
#find_package( Boost 1.52 COMPONENTS system thread REQUIRED )
#endif()

include_directories( ${Boost_INCLUDE_DIR} )

add_executable(c_example c_example.cpp)
TARGET_LINK_LIBRARIES(c_example ${JANUS_IMPLEMENTATION} ${JANUS_IO_IMPLEMENTATION})
target_link_libraries(c_example ${Boost_LIBRARIES})
#target_link_libraries(c_example ${CMAKE_THREAD_LIBS_INIT})

FILE(COPY test.py DESTINATION .)
