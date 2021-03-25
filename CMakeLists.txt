cmake_minimum_required(VERSION 3.10...3.12 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else ()
    cmake_policy(VERSION 3.12)
endif()

project(gadgetron_Bartholomew LANGUAGES CXX C)

if (WIN32)
    add_definitions(-D__BUILD_GADGETRON_BARTHOLOMEW__)
endif ()

# --- Provide good defaults for searching for packages (i.e. ismrmrd)
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

if (EXISTS $ENV{CMAKE_PREFIX_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif ()
if (EXISTS $ENV{ISMRMRD_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif ()
if (EXISTS $ENV{TORCH_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_HOME})
endif ()
if (CMAKE_PREFIX_PATH)
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif ()
message(STATUS "Looking for packages in : ${CMAKE_PREFIX_PATH}")
# ---

# check the compiler version
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (NOT CMAKE_CXX_COMPILER_VERSION) #Compiler version is not set on Ubuntu 12.02 (gcc 4.6)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
    endif ()
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message("CXX COMPILER VERSION: ${CMAKE_CXX_COMPILER_ID} : ${CMAKE_CXX_COMPILER_VERSION}")
        message(FATAL_ERROR "Gadgetron requires GCC version >= 6.0")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
        message(FATAL_ERROR "Gadgetron requires Clang version >= 3.5")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
        message(FATAL_ERROR "Gadgetron requires MSVC 2013 or above ")
    endif ()
else ()
    message(WARNING "Unsupported compiler!")
endif ()

#VERSIONING AND INSTALL PATHS
set(GADGETRON_VERSION_MAJOR 4)
set(GADGETRON_VERSION_MINOR 1)
set(GADGETRON_VERSION_PATCH 1)
set(GADGETRON_VERSION_STRING ${GADGETRON_VERSION_MAJOR}.${GADGETRON_VERSION_MINOR}.${GADGETRON_VERSION_PATCH})
set(GADGETRON_SOVERSION ${GADGETRON_VERSION_MAJOR}.${GADGETRON_VERSION_MINOR})
find_package(Git)
if (GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE GT_BARTS_GIT_SHA1_HASH)
  string(STRIP ${GT_BARTS_GIT_SHA1_HASH} GT_BARTS_GIT_SHA1_HASH)
else()
  set(GT_BARTS_GIT_SHA1_HASH "NA")
endif()

set(GADGETRON_INSTALL_CMAKE_PATH share/gadgetron/cmake)
set(GADGETRON_INSTALL_CONFIG_PATH share/gadgetron/config)
set(GADGETRON_INSTALL_LOGO_PATH share/gadgetron/logo)
set(GADGETRON_INSTALL_MATLAB_PATH share/gadgetron/matlab)
set(GADGETRON_INSTALL_PYTHON_MODULE_PATH share/gadgetron/python)
set(GADGETRON_INSTALL_SCHEMA_PATH schema)
set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)
set(GADGETRON_INSTALL_CHROOT_SCRIPTS_PATH share/gadgetron/chroot)
if (WIN32)
    set(GADGETRON_VAR_DIR $ENV{APPDATA}/gadgetron CACHE PATH "Path where Gadgetron will store calibration data, etc" FORCE)
else ()
    set(GADGETRON_VAR_DIR /var/lib/gadgetron/ CACHE PATH "Path where Gadgetron will store calibration data, etc" FORCE)
endif ()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#Set the build type to Release if not specified
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# build options for 64 bits system
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  message("64bit system is found")
  set( HAS_64_BIT On CACHE BOOL "64bit build")
else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  message("32bit system is found")
  set( HAS_64_BIT Off CACHE BOOL "64bit build")
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

set(CMAKE_INSTALL_RPATH ".:${CMAKE_INSTALL_PREFIX}/lib")

# build options for OpenMP support
find_package(OpenMP)
option(USE_OPENMP "Use OpenMP" On)
if (OPENMP_FOUND)
    if (USE_OPENMP)
        message("OpenMP multithreading enabled")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OMP)
    else ()
        message("OpenMP multithreading is supported, but disabled")
    endif ()
else ()
    message("OpenMP multithreading not supported")
endif ()

set(CMAKE_CXX_STANDARD 17)
if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-DWIN32 -D_WIN32 -D_WINDOWS -DWIN -D_AMD64_)
    add_definitions(-Zc:__cplusplus)
    add_definitions(-DNOMINMAX)
    #add_definitions(-DUNICODE -D_UNICODE)
    add_definitions("/wd4251") #disable warnings, 4251: needs to have dll-interface to be used by clients/
    if (HAS_64_BIT)
        add_definitions(-DWIN64 -D_WIN64)
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /openmp:experimental /permissive- /FS /DNOGDI /we4667")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc /MP /bigobj /w1")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /INCREMENTAL:NO")
    set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} /INCREMENTAL:NO")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    # The two flags below is to fix Windows problems in relation to multiple defined operators new/delete and some constructors that are defined in our headers
    #  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE")
    #  set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE")
else ()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    include(CheckCXXCompilerFlag)
    if (APPLE)
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        endif ()
    endif ()

    check_cxx_compiler_flag(-Werror=return-type COMPILER_SUPPORTS_WRETURNTYPE)
    if (COMPILER_SUPPORTS_WRETURNTYPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
    endif ()
    check_cxx_compiler_flag(-Werror=terminate COMPILER_SUPPORTS_WTERMINATE)
    if (COMPILER_SUPPORTS_WTERMINATE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=terminate")
    endif ()
    check_cxx_compiler_flag(-Werror=non-virtual-dtor COMPILER_SUPPORT_WNON_VIRTUAL)
    if (COMPILER_SUPPORT_WNON_VIRTUAL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=non-virtual-dtor")
    endif ()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif ()

# whether to suppress compilation warnings
option(BUILD_SUPPRESS_WARNINGS "Build package while suppressing warnings" Off)
if (BUILD_SUPPRESS_WARNINGS)
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    elseif (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif ()
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# uncomment these if all compiled targets are to be stored in the same directory
if (WIN32)
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/prod/bin CACHE STRING "Where to put the executables")
    SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/prod/bin CACHE STRING "Where to put the libraries")
endif (WIN32)

# whether to suppress compilation warnings
OPTION(BUILD_FOR_TIMING "Build package while timing purposes" Off)
if (BUILD_FOR_TIMING)
    ADD_DEFINITIONS(-DBUILD_WITH_TIMING)
endif (BUILD_FOR_TIMING)

###############################################################
#Bootstrap search for libraries (We need to find cmake modules in Gadgetron)
###############################################################

find_path(GADGETRON_CMAKE_MODULES FindGadgetron.cmake HINTS
  ${CMAKE_SOURCE_DIR}/cmake 
  $ENV{GADGETRON_HOME}/share/gadgetron/cmake
  ${GADGETRON_HOME}/share/gadgetron/cmake
  /usr/local/share/gadgetron/cmake)

if (NOT GADGETRON_CMAKE_MODULES)
   #TODO: Check if path found otherwise create error
   MESSAGE(FATAL_ERROR "GADGETRON_CMAKE_MODULES cannot be found. 
   Try to set GADGETRON_HOME environment variable.")
endif(NOT GADGETRON_CMAKE_MODULES)

message("GADGETRON_HOME is ${GADGETRON_HOME}")

find_path(ISMRMRD_CMAKE_MODULES ISMRMRDConfig.cmake  HINTS
  $ENV{GADGETRON_HOME}/lib/cmake/ISMRMRD 
  ${GADGETRON_HOME}/lib/cmake/ISMRMRD 
  /usr/local/lib/cmake/ISMRMRD)

if (NOT ISMRMRD_CMAKE_MODULES)
   #TODO: Check if path found otherwise create error
   MESSAGE(FATAL_ERROR "ISMRMRD_CMAKE_MODULES cannot be found. 
   Try to set GADGETRON_HOME environment variable.")
endif(NOT ISMRMRD_CMAKE_MODULES)

set(CMAKE_MODULE_PATH ${GADGETRON_CMAKE_MODULES} ${ISMRMRD_CMAKE_MODULES} ${CMAKE_SOURCE_DIR}/cmake ${GADGETRON_HOME}/share/gadgetron/cmake)
list(APPEND CMAKE_MODULE_PATH ${GADGETRON_HOME}/lib/cmake/ISMRMRD)
message("CMAKE_MODULE_PATH is " ${CMAKE_MODULE_PATH})

###############################################################

set(CMAKE_INSTALL_PREFIX ${GADGETRON_HOME})
LINK_DIRECTORIES(${GADGETRON_LIB_DIR} ${GADGETRON_HOME}/lib)

###############################################################

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# necessary for Windows and RHEL <=6 systems

if(WIN32)
    add_definitions( -DHAVE_SNPRINTF) #Python decides to overwrite snprintf if this is not defined, because ofcourse it does.
endif()
# We actually only use system and thread explicitly, but they require linking in date_time and chrono
if (WIN32)
    find_package(Boost 1.65.0 COMPONENTS system date_time chrono program_options filesystem timer exception REQUIRED )
else()
    find_package(Boost 1.65.0 COMPONENTS system date_time chrono program_options filesystem timer REQUIRED )
endif()

if (NOT DEFINED Boost_VERSION_STRING)
    set(Boost_VERSION_STRING "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
endif()

if (Boost_FOUND)
    message("Found Boost : ${Boost_LIBRARIES}")
endif()

###############################################################

find_package(PythonLibs 3  REQUIRED)

find_package(NumPy REQUIRED)
message("Found numpy : ${NUMPY_INCLUDE_DIRS} ${NUMPY_LIBRARIES} ${PYTHON_INCLUDE_DIRS} ${PYTHON_LIBRARIES}")

add_library(python INTERFACE)
set_property(TARGET python PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
message("Boost_VERSION ${Boost_VERSION_STRING}")
if (Boost_VERSION_STRING VERSION_LESS 1.67.0)
    find_package(Boost 1.65.0 COMPONENTS python3 REQUIRED)
    set(Boost_PYTHON3_TARGET Boost::python3)
    else()
    string(REGEX MATCH "^3\\.([0-9]+)\\.[0-9]+" PYTHON_MINOR_VERSION ${PYTHONLIBS_VERSION_STRING} )
    set(PYTHON_MINOR_VERSION ${CMAKE_MATCH_1})
    find_package(Boost 1.65.0 COMPONENTS "python3${PYTHON_MINOR_VERSION}" REQUIRED)
    set(Boost_PYTHON3_FOUND TRUE)
    set(Boost_PYTHON3_TARGET Boost::python3${PYTHON_MINOR_VERSION})
endif()
set_property(TARGET python PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
target_link_libraries(python INTERFACE ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES} ${Boost_PYTHON3_TARGET})

if (Boost_FOUND)
    add_library(Boost INTERFACE)
    set_property(TARGET Boost PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BOOST_INCLUDE_DIRS})
    target_link_libraries(Boost INTERFACE ${Boost_LIBRARIES})
endif()

###############################################################
###############################################################

IF (WIN32)
    set(CMAKE_DEBUG_SUFFIX d CACHE STRING "the debug suffix")
ENDIF(WIN32)

message("GADGETRON_INCLUDE_DIR is ${GADGETRON_INCLUDE_DIR}")

include_directories(
    ${GADGETRON_INCLUDE_DIR}
    ${GADGETRON_HOME}/include/gadgetron
    ${GADGETRON_HOME}/include/ismrmrd
    #${ISMRMRD_INCLUDE_DIR}
    #${Boost_INCLUDE_DIR}
  )

    # install python gadgets
    set(gadgetron_Bartholomew_python_gadgets_files 
            python/cine_auto_contouring.py 
            python/IsmrmredImageArray_cine_auto_contouring.py
            python/segmentation_settings.py
            python/segmentation_tools.py
        )

    source_group(gadgets FILES ${gadgetron_Bartholomew_python_gadgets_files})

    install(FILES
        ${gadgetron_Bartholomew_python_gadgets_files}
        python/optimised_graph.pb
        DESTINATION ${GADGETRON_INSTALL_PYTHON_MODULE_PATH} COMPONENT main)

    # add the demo
    include_directories(${CMAKE_SOURCE_DIR}
                        ${ISMRMRD_INCLUDE_DIR})

    add_executable(gadgetron_Bartholomew_demo 
                    demo.cpp
                    ${gadgetron_Bartholomew_python_gadgets_files} )

    target_link_libraries(gadgetron_Bartholomew_demo 
                        gadgetron_toolbox_python
                        gadgetron_toolbox_log
                        )

    set_target_properties(gadgetron_Bartholomew_demo PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

    install(TARGETS gadgetron_Bartholomew_demo DESTINATION bin COMPONENT main)
