cmake_minimum_required(VERSION 3.5)
project(KarePlakaModel VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

### --- ADDED FOR ONNX RUNTIME ---
# 1. Point CMake to the folder containing onnxruntimeConfig.cmake
# (Assuming you put them in a folder named 'cmake' next to this CMakeLists.txt)
set(onnxruntime_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 2. Find the ONNX Runtime package using those files
find_package(onnxruntime REQUIRED)
# Find OpenCV
find_package(OpenCV REQUIRED)
### ------------------------------

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(KarePlakaModel
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        platefinder.h platefinder.cpp
        README.MD
        plateresult.h
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET KarePlakaModel APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
else()
    if(ANDROID)
        add_library(KarePlakaModel SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(KarePlakaModel
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# --- Copy model.onnx to the build directory automatically ---
add_custom_command(TARGET KarePlakaModel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx/model/kareplaka.onnx"
        "$<TARGET_FILE_DIR:KarePlakaModel>/model/kareplaka.onnx"
    COMMENT "Copying model.onnx to the output directory..."
)

### --- ADDED FOR ONNX RUNTIME ---
# 3. Add 'onnxruntime' to your target_link_libraries
target_link_libraries(KarePlakaModel PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    onnxruntime
    ${OpenCV_LIBS}
)
### ------------------------------

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.KarePlakaModel)
endif()
set_target_properties(KarePlakaModel PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS KarePlakaModel
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(KarePlakaModel)
endif()

