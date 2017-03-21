MESSAGE( STATUS "CMAKE_SOURCE_DIR " ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include_directories(
	../../src/glaive/ 
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/include/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/include/opencv/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/caffe/include/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/caffe/build/src/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/yolo/
	/usr/include/python2.7
    /nfs/isicvlnas01/share/mlpack-2.1.1/include/
    /nfs/isicvlnas01/users/xpeng/tparty/armadillo/armadillo-7.700.0/include
	/usr/local/cuda-7.5/targets/x86_64-linux/include/
	/usr/local/cuda-7.5/include/
)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/atlas
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/caffe/build/lib
	${CMAKE_SOURCE_DIR}/lib
	${CMAKE_SOURCE_DIR}/../../lib
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/OpenMesh-3.2/build/Build/lib/OpenMesh/
	${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/yolo
    /nfs/isicvlnas01/share/mlpack-2.1.1/lib/
    /nfs/isicvlnas01/users/xpeng/tparty/armadillo/armadillo-7.700.0/lib
)
# /nfs/isicvlnas01/users/xpeng/tparty/mlpack/mlpack.2.1.1/lib
add_library(glaive SHARED
  glaive.cpp CnnFeatex.cpp CnnServer.cpp ImagePreproc.cpp python_wrappers.cpp PyServer.cpp CnnPoseDetection.cpp ../janus_io.cpp

)

set_target_properties(glaive PROPERTIES
		      COMPILE_FLAGS "-DJANUS_LIBRARY"
                      VERSION ${JANUS_VERSION_MAJOR}.${JANUS_VERSION_MINOR}.${JANUS_VERSION_PATCH}
                      SOVERSION ${JANUS_VERSION_MAJOR}.${JANUS_VERSION_MINOR})



set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_calib3d.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_core.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_features2d.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_flann.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_highgui.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_imgproc.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_ml.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_objdetect.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_photo.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_stitching.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_superres.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_video.so ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/libopencv_imgcodecs.so /nfs/isicvlnas01/share/mlpack-2.1.1/lib/libmlpack.so /nfs/isicvlnas01/users/xpeng/tparty/armadillo/armadillo-7.700.0/lib/libarmadillo.so " )




set ( PYTHON_FLAGS "-I/usr/include/python2.7/ -fno-strict-aliasing -fPIC -DNDEBUG -g -fwrapv -Wall" )
set ( PYTHON_LINKER_FLAGS "-L/usr/lib64/python2.7/config -lpthread -ldl -lutil -lm -lpython2.7 -Xlinker -export-dynamic")
set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PYTHON_FLAGS} -Wall -Wextra -fvisibility=hidden -gdwarf-2")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYTHON_FLAGS} -Wall -Wextra -fvisibility=hidden -std=c++11 -gdwarf-2")
set ( CMAKE_EXE_LINKER_FLAGS "-L${GLAIVE_LIB}" "${PYTHON_LINKER_FLAGS}" )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${CMAKE_SOURCE_DIR}/../../lib" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${CMAKE_SOURCE_DIR}/lib" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/caffe/build/lib")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/OpenMesh-3.2/build/Build/lib/OpenMesh/")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib/atlas")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/lib")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,/usr/local/cuda-7.5/lib64")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,/usr/local/cuda-7.5/lib")


target_link_libraries(glaive 
			     CLNF_0_9 caffe hdf5 hdf5_cpp satlas pthread dl util m glog
			     boost_system boost_filesystem
			     OpenMeshCored OpenMeshToolsd darknet-yolo
			     OSMesa GLU GL SM ICE X11 Xext glut Xmu Xi GLEW
			     python2.7 config++
)


install(TARGETS glaive   RUNTIME DESTINATION bin
                         LIBRARY DESTINATION lib
                         ARCHIVE DESTINATION lib)

# Add this to the list of implementations to test
set(JANUS_TEST_IMPLEMENTATION ${JANUS_TEST_IMPLEMENTATIONS} "glaive" PARENT_SCOPE)

