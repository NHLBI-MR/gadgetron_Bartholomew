cmake_minimum_required(VERSION 2.6)

project(gadgetron_Bartholomew)

if (WIN32)
    add_definitions(-D__BUILD_GADGETRON_BARTHOLOMEW__)
endif ()

# --- Provide good defaults for searching for packages (i.e. ismrmrd)
set(CMAKE_PREFIX_PATH "")
if(CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "/usr/local")
endif()
if(EXISTS $ENV{CMAKE_PREFIX_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif()
if(EXISTS $ENV{ISMRMRD_HOME})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif()
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
message(STATUS "Looking for packages in : ${CMAKE_PREFIX_PATH}")
# ---

# CMake 3.0 ``MACOSX_RPATH`` is enabled by default.
if (APPLE)
    cmake_policy(SET CMP0042 NEW)
endif()

# Require C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# check the compiler version
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (NOT CMAKE_CXX_COMPILER_VERSION) #Compiler version is not set on Ubuntu 12.02 (gcc 4.6)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
  endif()
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    message("CXX COMPILER VERSION: ${CMAKE_CXX_COMPILER_ID} : ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "Gadgetron requires GCC version >= 4.8")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
    message(FATAL_ERROR "Gadgetron requires Clang version >= 3.5")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "Gadgetron requires MSVC 2013 or above ")
  endif()
else()
  message(WARNING "Unsupported compiler!")
endif()

#VERSIONING AND INSTALL PATHS
set(GADGETRON_VERSION_MAJOR 3)
set(GADGETRON_VERSION_MINOR 15)
set(GADGETRON_VERSION_PATCH 0) 
set(GADGETRON_VERSION_STRING ${GADGETRON_VERSION_MAJOR}.${GADGETRON_VERSION_MINOR}.${GADGETRON_VERSION_PATCH})
set(GADGETRON_SOVERSION ${GADGETRON_VERSION_MAJOR}.${GADGETRON_VERSION_MINOR})
find_package(Git)
if (GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE SASHA_HC_GIT_SHA1_HASH)
  string(STRIP ${SASHA_HC_GIT_SHA1_HASH} SASHA_HC_GIT_SHA1_HASH)
else()
  set(SASHA_HC_GIT_SHA1_HASH "NA")
endif()

set(GADGETRON_INSTALL_CMAKE_PATH share/gadgetron/cmake)
set(GADGETRON_INSTALL_CONFIG_PATH share/gadgetron/config)
set(GADGETRON_INSTALL_MATLAB_PATH share/gadgetron/matlab)
set(GADGETRON_INSTALL_PYTHON_MODULE_PATH share/gadgetron/python)
set(GADGETRON_INSTALL_SCHEMA_PATH schema)
set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)

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

# build options for OpenMP support
find_package(OpenMP)
OPTION(USE_OPENMP "Use OpenMP" On)
if (OPENMP_FOUND)
    if(USE_OPENMP) 
        message("OpenMP multithreading enabled")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        ADD_DEFINITIONS(-DUSE_OMP)
    else (USE_OPENMP)
        message("OpenMP multithreading is supported, but disabled")
    endif(USE_OPENMP) 
else (OPENMP_FOUND)
  message("OpenMP multithreading not supported")
endif (OPENMP_FOUND)

if (WIN32)
    ADD_DEFINITIONS(-DWIN32 -D_WIN32 -D_WINDOWS)
#    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
    ADD_DEFINITIONS(-D_VARIADIC_MAX=10) #to fix compiler limitations in Visual Studio Express
    ADD_DEFINITIONS("/wd4251") #disable warnings, 4251: needs to have dll-interface to be used by clients
    ADD_DEFINITIONS("/wd4344") #disable warnings, 4344: behavior change: use of explicit template arguments
    ADD_DEFINITIONS("/wd4996") #disable warnings, 4996: the POSIX name for this item is deprecated. Instead, use the ISO C++ conformant name
    if ( HAS_64_BIT )
        ADD_DEFINITIONS(-DWIN64 -D_WIN64)
    endif ( HAS_64_BIT )
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /arch:AVX")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /bigobj")
    SET (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /INCREMENTAL:NO")
    SET (CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} /INCREMENTAL:NO")
    SET (CMAKE_EXE_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    SET (CMAKE_SHARED_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    SET (CMAKE_STATIC_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    SET (CMAKE_MODULE_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    # The two flags below is to fix Windows problems in relation to multiple defined operators new/delete and some constructors that are defined in our headers
    #SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE") 
    #SET (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE") 
else (WIN32)
    if (UNIX)
        if (APPLE)
        else (APPLE)
            SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")
        endif (APPLE)
    endif (UNIX)
endif (WIN32)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${GADGETRON_HOME}/lib/cmake/ISMRMRD)

# whether to suppress compilation warnings
OPTION(BUILD_SUPPRESS_WARNINGS "Build package while suppressing warnings" Off)
if (BUILD_SUPPRESS_WARNINGS)
    if (WIN32)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    elseif (WIN32)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w")
    endif (WIN32)
endif (BUILD_SUPPRESS_WARNINGS)

# whether to suppress compilation warnings
OPTION(BUILD_FOR_TIMING "Build package while timing purposes" Off)
if (BUILD_FOR_TIMING)
    ADD_DEFINITIONS(-DBUILD_WITH_TIMING)
endif (BUILD_FOR_TIMING)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# necessary for Windows and RHEL <=6 systems
set(Boost_NO_BOOST_CMAKE ON)

if(WIN32)
  add_definitions( -DBOOST_ALL_NO_LIB )
  add_definitions( -DBOOST_ALL_DYN_LINK )
endif(WIN32)

###############################################################
#Bootstrap search for libraries (We need to find cmake modules in Gadgetron)
###############################################################

message("GADGETRON_HOME is ${GADGETRON_HOME}")

find_path(GADGETRON_CMAKE_MODULES FindGadgetron.cmake HINTS
  $ENV{GADGETRON_HOME}/share/gadgetron/cmake
  ${GADGETRON_HOME}/share/gadgetron/cmake
  /usr/local/share/gadgetron/cmake)

if (NOT GADGETRON_CMAKE_MODULES)
   #TODO: Check if path found otherwise create error
   MESSAGE(FATAL_ERROR "GADGETRON_CMAKE_MODULES cannot be found. 
   Try to set GADGETRON_HOME environment variable.")
endif(NOT GADGETRON_CMAKE_MODULES)

find_path(ISMRMRD_CMAKE_MODULES ISMRMRDConfig.cmake  HINTS
  $ENV{GADGETRON_HOME}/lib/cmake/ISMRMRD 
  ${GADGETRON_HOME}/lib/cmake/ISMRMRD 
  /usr/local/lib/cmake/ISMRMRD)

if (NOT ISMRMRD_CMAKE_MODULES)
   #TODO: Check if path found otherwise create error
   MESSAGE(FATAL_ERROR "ISMRMRD_CMAKE_MODULES cannot be found. 
   Try to set GADGETRON_HOME environment variable.")
endif(NOT ISMRMRD_CMAKE_MODULES)

set(CMAKE_MODULE_PATH ${GADGETRON_CMAKE_MODULES} ${ISMRMRD_CMAKE_MODULES} ${CMAKE_SOURCE_DIR}/cmake)
###############################################################
find_package(Gadgetron REQUIRED)

set(CMAKE_INSTALL_PREFIX ${GADGETRON_HOME})
LINK_DIRECTORIES(${GADGETRON_LIB_DIR})

# We actually only use system and thread explicitly, but they require linking in date_time and chrono
#if (WIN32)
#  find_package(Boost COMPONENTS system thread date_time chrono program_options filesystem REQUIRED)
#else(WIN32)
#  find_package(Boost COMPONENTS system thread program_options filesystem REQUIRED)
#endif(WIN32)

#find_package(ISMRMRD REQUIRED)
#if (ISMRMRD_FOUND)
#    message("ISMRMRD Found")
#else()
#    message("ISMRMRD not found")
#endif()

find_package(PythonLibs 2)
string(REPLACE "." ";" PYTHON_VERSION_MAJOR ${PYTHONLIBS_VERSION_STRING})
list(GET PYTHON_VERSION_MAJOR 0 PYVER)
add_definitions(-DPYVER=${PYVER})

find_package(NumPy)
if(NOT PYTHONLIBS_FOUND)
    message(STATUS "Python Libraries/Headers NOT found, NOT compiling Python Gadgets")
endif()

###############################################################
###############################################################

IF (WIN32)
    set(CMAKE_DEBUG_SUFFIX d CACHE STRING "the debug suffix")
ENDIF(WIN32)

include_directories(
    ${GADGETRON_INCLUDE_DIR}
    #${ISMRMRD_INCLUDE_DIR}
    #${Boost_INCLUDE_DIR}
  )

if (PYTHONLIBS_FOUND)

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
                        optimized gadgetron_toolbox_python  debug gadgetron_toolbox_python${CMAKE_DEBUG_SUFFIX}
                        optimized gadgetron_toolbox_log  debug gadgetron_toolbox_log${CMAKE_DEBUG_SUFFIX}
                        #${ISMRMRD_LIBRARIES}
                        )

    set_target_properties(gadgetron_Bartholomew_demo PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

    install(TARGETS gadgetron_Bartholomew_demo DESTINATION bin COMPONENT main)
endif ()
