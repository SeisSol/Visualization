/**
 * @file
 * This file is part of SeisSol.
 *
 * @author Sebastian Rettenberger (sebastian.rettenberger AT tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger)
 *
 * @section LICENSE
 * Copyright (c) 2016, SeisSol Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>

#include <netcdf.h>

#include <hdf5.h>

#include "utils/args.h"
#include "utils/logger.h"

#include "KDTree.h"

const static int FACE2NODES[4][3] = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};

template<typename T>
static void checkH5Err(T status)
{
	if (status < 0)
		logError() << "An HDF5 error occurred";
}

static void checkNcError(int error)
{
	if (error != NC_NOERR)
		logError() << "An netCDF error occurred:" << nc_strerror(error);
}

struct Support {
  // limits[0][:] = (min x, max x)
  // limits[1][:] = (min y, max y)
  // limits[2][:] = (min y, max y)
  double limits[3][2];

  Support() {
    limits[0][0] = std::numeric_limits<double>::infinity();
    limits[0][1] = -std::numeric_limits<double>::infinity();
    limits[1][0] = std::numeric_limits<double>::infinity();
    limits[1][1] = -std::numeric_limits<double>::infinity();
    limits[2][0] = std::numeric_limits<double>::infinity();
    limits[2][1] = -std::numeric_limits<double>::infinity();
  }

  double operator()(int splitdim, int side) const {
    return limits[splitdim][side];
  }
};

struct Action {
	std::vector<Point> points;

	double operator()(Point& p) {
		points.push_back(p);
	}
};

int main(int argc, char* argv[])
{
	utils::Args args;
// 	args.addOption("level", 'l', "gzip compressen level [0-9]", utils::Args::Required, false);
	args.addAdditionalOption("mesh", "mesh file");
	args.addAdditionalOption("output", "output file");

	switch (args.parse(argc, argv)) {
	case utils::Args::Help:
		return 0;
	case utils::Args::Error:
		return 1;
	}

	const unsigned int direction = 0;

	std::string mesh = args.getAdditionalArgument<const char*>("mesh");
	std::string output = args.getAdditionalArgument<const char*>("output");

	// Read the fault vertices
	logInfo() << "Build fault K-D-tree...";
	int ncFile;
	checkNcError(nc_open(mesh.c_str(), NC_NOWRITE, &ncFile));

	int ncDimPart;
	checkNcError(nc_inq_dimid(ncFile, "partitions", &ncDimPart));
	size_t partitions;
	checkNcError(nc_inq_dimlen(ncFile, ncDimPart, &partitions));

	int ncDimElem;
	checkNcError(nc_inq_dimid(ncFile, "elements", &ncDimElem));
	size_t maxElements;
	checkNcError(nc_inq_dimlen(ncFile, ncDimElem, &maxElements));

	int ncDimVrtx;
	checkNcError(nc_inq_dimid(ncFile, "vertices", &ncDimVrtx));
	size_t maxVertices;
	checkNcError(nc_inq_dimlen(ncFile, ncDimVrtx, &maxVertices));

	int ncElemSize;
	checkNcError(nc_inq_varid(ncFile, "element_size", &ncElemSize));

	int ncElemVertices;
	checkNcError(nc_inq_varid(ncFile, "element_vertices", &ncElemVertices));

	int ncElemBoundaries;
	checkNcError(nc_inq_varid(ncFile, "element_boundaries", &ncElemBoundaries));

	int ncVrtxSize;
	checkNcError(nc_inq_varid(ncFile, "vertex_size", &ncVrtxSize));

	int ncVrtxCoords;
	checkNcError(nc_inq_varid(ncFile, "vertex_coordinates", &ncVrtxCoords));

	int* elementSize = new int[partitions];
	checkNcError(nc_get_var_int(ncFile, ncElemSize, elementSize));

	int* vertexSize = new int[partitions];
	checkNcError(nc_get_var_int(ncFile, ncVrtxSize, vertexSize));

	int* elementVertices = new int[maxElements*4];
	int* elementBoundaries = new int[maxElements*4];

	double* vertexCoordinates = new double[maxVertices*3];

	std::unordered_set<Point> faultPoints;

	unsigned long totalElements = 0;

	double minfault = std::numeric_limits<double>::infinity();
	double maxfault = -std::numeric_limits<double>::infinity();

	double minfault2 = std::numeric_limits<double>::infinity();
	double maxfault2 = -std::numeric_limits<double>::infinity();

	double outsideborder1 = 0;
	double outsideborder2 = 0;

	for (size_t p = 0; p < partitions; p++) {
		totalElements += elementSize[p];

		size_t offset[3] = {p, 0, 0};
		size_t size[3] = {1, static_cast<size_t>(elementSize[p]), 4};

		checkNcError(nc_get_vara_int(ncFile, ncElemVertices, offset, size, elementVertices));
		checkNcError(nc_get_vara_int(ncFile, ncElemBoundaries, offset, size, elementBoundaries));

		size[1] = vertexSize[p]; size[2] = 3;
// 		logInfo() << "size" << size[0] << size[1] << size[2];
		checkNcError(nc_get_vara_double(ncFile, ncVrtxCoords, offset, size, vertexCoordinates));

		for (unsigned int i = 0; i < elementSize[p]; i++) {
			for (unsigned int j = 0; j < 4; j++) {
				if (elementBoundaries[i*4+j] == 3) {
					// Is a fault boundary
					for (unsigned int k = 0; k < 3; k++) {
						Point point;
						memcpy(point.coords, &vertexCoordinates[elementVertices[i*4 + FACE2NODES[j][k]]*3], sizeof(double)*3);
						faultPoints.insert(point);

						minfault = std::min(minfault, point.coords[direction]);
						maxfault = std::max(maxfault, point.coords[direction]);

						if (point.coords[1-direction] < minfault2) {
							minfault2 = point.coords[1-direction];
							outsideborder1 = point.coords[direction];
						}
						if (point.coords[1-direction] > maxfault2) {
							maxfault2 = point.coords[1-direction];
							outsideborder2 = point.coords[direction];
						}
					}
				}
			}
		}
	}

	logInfo() << "Cells <" << minfault2 << "and >" << maxfault2 << "have the be handled manually";
	logInfo() << "A good choice might be" << outsideborder1 << "reps." << outsideborder2;

	bool split[3] = {true, true, true};
	split[direction] = false;
	KDTree kdtree(faultPoints, 4, split);

	faultPoints.clear(); // Free memory

	unsigned int* isLeft = new unsigned int[totalElements];
	std::fill(isLeft, isLeft+totalElements, 1);

	unsigned long globElement = 0;
	for (size_t p = 0; p < partitions; p++) {
		logInfo() << "Processing partition" << utils::nospace << p << "...";

		size_t offset[3] = {p, 0, 0};
		size_t size[3] = {1, static_cast<size_t>(elementSize[p]), 4};

		checkNcError(nc_get_vara_int(ncFile, ncElemVertices, offset, size, elementVertices));

		size[1] = vertexSize[p]; size[2] = 3;
		checkNcError(nc_get_vara_double(ncFile, ncVrtxCoords, offset, size, vertexCoordinates));

		for (unsigned int i = 0; i < elementSize[p]; i++) {
			double avg = 0;

			Support sup;
			for (unsigned int j = 0; j < 4; j++) {
				sup.limits[0][0] = std::min(sup.limits[0][0], vertexCoordinates[elementVertices[i*4+j]*3]);
				sup.limits[0][1] = std::max(sup.limits[0][1], vertexCoordinates[elementVertices[i*4+j]*3]);
				sup.limits[1][0] = std::min(sup.limits[1][0], vertexCoordinates[elementVertices[i*4+j]*3+1]);
				sup.limits[1][1] = std::max(sup.limits[1][1], vertexCoordinates[elementVertices[i*4+j]*3+1]);
				sup.limits[2][0] = std::min(sup.limits[1][0], vertexCoordinates[elementVertices[i*4+j]*3+2]);
				sup.limits[2][1] = std::max(sup.limits[1][1], vertexCoordinates[elementVertices[i*4+j]*3+2]);

				avg += vertexCoordinates[elementVertices[i*4+j]*3 + direction];
			}

			avg /= 4;

			if (avg < minfault) {
				// Do nothing
			} else if (avg > maxfault) {
				isLeft[globElement] = 0;
			} else {
				Action act;
				kdtree.search(sup, act);

				bool hasPoint = false;

				for (std::vector<Point>::const_iterator it = act.points.begin();
						it != act.points.end(); it++) {
					if (it->coords[1-direction] >= sup.limits[1-direction][0] && it->coords[1-direction] <= sup.limits[1-direction][1]
							&& it->z >= sup.limits[2][0] && it->z <= sup.limits[2][1]) {
						hasPoint = true;

						if (it->coords[direction] < avg) {
							isLeft[globElement] = 0;
							break;
						}
					}
				}

				if (!hasPoint) {
					if (avg > minfault) {
						isLeft[globElement] = 0;
					}
				}
			}

			globElement++;
		}
	}

	checkNcError(nc_close(ncFile));

	delete [] elementVertices;
	delete [] elementBoundaries;
	delete [] vertexCoordinates;
	delete [] elementSize;
	delete [] vertexSize;

	hid_t h5File = H5Fopen(output.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	checkH5Err(h5File);

	hsize_t chunkSize = 256*1024*1024 / sizeof(unsigned int);

	hid_t h5var = H5Dopen(h5File, "/is_left", H5P_DEFAULT);
	hid_t h5space;
	if (h5var >= 0) {
		logWarning() << "Overwriting old dataset from HDF5 file";

		h5space = H5Dget_space(h5var);
		checkH5Err(h5space);
		int ndims = H5Sget_simple_extent_ndims(h5space);
		checkH5Err(ndims);
		if (ndims != 1)
			logError() << "Old dataset has wrong dimension";

		hsize_t extent;
		checkH5Err(H5Sget_simple_extent_dims(h5space, &extent, 0L));
		if (extent != totalElements)
			logError() << "Old dataset has wrong size";
	} else {
		// Create new dataset
		hsize_t dim = totalElements;
		h5space = H5Screate_simple(1, &dim, 0L);
		checkH5Err(h5space);
		hid_t h5pcreate = H5Pcreate(H5P_DATASET_CREATE);
		checkH5Err(h5pcreate);
		hsize_t chunkDim = std::min(chunkSize, dim);
		checkH5Err(H5Pset_chunk(h5pcreate, 1, &chunkDim));
// 		checkH5Err(H5Pset_szip(h5pcreate, H5_SZIP_NN_OPTION_MASK, 4));
		checkH5Err(H5Pset_deflate(h5pcreate, 5));
		h5var = H5Dcreate(h5File, "/is_left", H5T_STD_U32LE, h5space,
			H5P_DEFAULT, h5pcreate, H5P_DEFAULT);
		checkH5Err(h5var);
		checkH5Err(H5Pclose(h5pcreate));
	}

	hsize_t dim = totalElements;
	hid_t h5memspace = H5Screate_simple(1, &dim, 0L);
	checkH5Err(h5memspace);

	checkH5Err(H5Sselect_all(h5memspace));

	checkH5Err(H5Dwrite(h5var, H5T_NATIVE_UINT32, h5memspace, h5space, H5P_DEFAULT, isLeft));

	checkH5Err(H5Sclose(h5memspace));
	checkH5Err(H5Dclose(h5var));
	checkH5Err(H5Sclose(h5space));

	delete [] isLeft;

	checkH5Err(H5Fclose(h5File));

	return 0;
}
