set(HDF5_PREFER_PARALLEL True)
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF REQUIRED netcdf)

add_executable(trace trace.cpp image.cpp input.cpp)
target_compile_features(trace PRIVATE cxx_std_17)
target_include_directories(trace PRIVATE ../../external/ ${NETCDF_INCLUDE_DIRS})
target_link_directories(trace PRIVATE ${NETCDF_LIBRARY_DIRS})
target_link_libraries(trace PRIVATE ${NETCDF_LIBRARIES}
                                    ${HDF5_HL_LIBRARIES}
                                    ${HDF5_C_LIBRARIES}
                                    MPI::MPI_CXX)

