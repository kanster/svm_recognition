cmake_minimum_required(VERSION 2.8)
project( rgbd_classification )


# For building using OpenCV
find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    set(OpenCV_FOUND TRUE)
else()
    set(OpenCV_FOUND FALSE)
endif()

# For building using PCL
find_package(PCL 1.7 REQUIRED )
if(PCL_FOUND)
    message(STATUS "Found PCL ${PCL_LIBRARY_DIRS}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    set(PCL_FOUND TRUE)
else()
    set(PCL_FOUND FALSE)
endif()

find_package(Boost COMPONENTS filesystem regex program_options thread)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(./include/)
set(CPP
    src/BoW.cpp
    src/ObjectClassifier.cpp
    src/ObjectDescription.cpp
    src/OCLCommon.cpp
    src/OCLPose3D.cpp
    src/OCLUtils.cpp
    src/svm.cpp
)

set(HEADER
    include/BoW.h
    include/ObjectClassifier.h
    include/ObjectDescription.h
    include/OCLCommon.h
    include/OCLPose3D.h
    include/OCLUtils.h
    include/SVMUtils.hpp
    include/svm.h
)

add_executable( ${PROJECT_NAME} main.cpp ${HEADER} ${CPP})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})


