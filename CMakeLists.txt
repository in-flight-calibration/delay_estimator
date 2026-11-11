cmake_minimum_required(VERSION 3.20)
project(delay_estimator)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ifc_msgs REQUIRED)
find_package(ifc_tools REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d highgui imgproc plot)
find_package(px4_msgs REQUIRED)
find_package(rclcpp REQUIRED)

function(add_delay_executable name source)
    add_executable(${name} ${source})

    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(${name} PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        ifc_msgs::ifc_msgs__rosidl_typesupport_cpp
        ifc_tools::ifc_tools
        px4_msgs::px4_msgs__rosidl_typesupport_cpp
        rclcpp::rclcpp
        ${OpenCV_LIBS}
    )
endfunction()

add_delay_executable(delay_estimator src/main.cpp)

install(TARGETS delay_estimator
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
    enable_testing()
    add_delay_executable(delay_estimator_attitude_buffer test/test_attitude_buffer.cpp)
    add_test(NAME delay_estimator_attitude_buffer COMMAND delay_estimator_attitude_buffer)
    add_delay_executable(delay_estimator_core test/test_delay_search.cpp)
    add_test(NAME delay_estimator_core COMMAND delay_estimator_core)
endif()

ament_export_dependencies(Ceres Eigen3 ifc_msgs ifc_tools OpenCV px4_msgs rclcpp)

ament_package()
