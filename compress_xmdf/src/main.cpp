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
#include <string>

#include <hdf5.h>

#include "utils/args.h"
#include "utils/logger.h"
#include "utils/stringutils.h"

template<typename T>
static void checkH5Err(T status)
{
	if (status < 0)
		logError() << "An HDF5 error occurred";
}

static void compressData(unsigned int ndims, hsize_t offset[], hsize_t size[],
	hid_t h5invar, hid_t h5inspace, hid_t h5outvar, hid_t h5outspace,
	hid_t h5native_type, void* buffer)
{
	hid_t h5memspace = H5Screate_simple(ndims, size, 0L);
	checkH5Err(h5memspace);

	checkH5Err(H5Sselect_hyperslab(h5inspace, H5S_SELECT_SET, offset, 0L, size, 0L));
	checkH5Err(H5Dread(h5invar, h5native_type, h5memspace, h5inspace, H5P_DEFAULT, buffer));

	checkH5Err(H5Sselect_hyperslab(h5outspace, H5S_SELECT_SET, offset, 0L, size, 0L));
	checkH5Err(H5Dwrite(h5outvar, h5native_type, h5memspace, h5outspace, H5P_DEFAULT, buffer));

	checkH5Err(H5Sclose(h5memspace));
}

template<typename T>
static void compressDataset(hid_t h5ifile, hid_t h5ofile,
	const char* varname, hid_t h5native_type, hid_t h5type,
	void* buffer, unsigned long bufferSize)
{
	// Open original var
	hid_t h5ivar = H5Dopen(h5ifile, varname, H5P_DEFAULT);
	checkH5Err(h5ivar);

	// Get dimension information
	hid_t h5ispace = H5Dget_space(h5ivar);
	checkH5Err(h5ispace);

	int ndims = H5Sget_simple_extent_ndims(h5ispace);
	checkH5Err(ndims);
	if (ndims > 2)
		logError() << "Dimension > 2 are not supported";

	hsize_t extent[2];
	checkH5Err(H5Sget_simple_extent_dims(h5ispace, extent, 0L));
	hsize_t nelements = extent[0];
	unsigned int dim2 = 1;
	if (ndims > 1)
		dim2 = extent[1];

	// Create new dataset
	hsize_t chunkSize = bufferSize / dim2 / sizeof(T);

	hsize_t dims[2] = {nelements, dim2}; // Change this for other elements
	hid_t h5ospace = H5Screate_simple(ndims, dims, 0L);
	checkH5Err(h5ospace);
	hid_t h5opcreate = H5Pcreate(H5P_DATASET_CREATE);
	checkH5Err(h5opcreate);
	hsize_t chunkDims[2] = {std::min(chunkSize, nelements), dim2};
	checkH5Err(H5Pset_chunk(h5opcreate, ndims, chunkDims));
// 	checkH5Err(H5Pset_szip(h5opcreate, H5_SZIP_NN_OPTION_MASK, 4));
	checkH5Err(H5Pset_deflate(h5opcreate, 6));
	hid_t h5ovar = H5Dcreate(h5ofile, varname, h5type, h5ospace,
		H5P_DEFAULT, h5opcreate, H5P_DEFAULT);
	checkH5Err(h5ovar);
	checkH5Err(H5Pclose(h5opcreate));

	// Transfer data
	unsigned long pos = 0;
	while (pos < nelements) {
		const unsigned long left = nelements - pos;
		if (left < chunkSize)
			chunkSize = left;

		hsize_t offset[2] = {pos, 0};
		hsize_t size[2] = {chunkSize, static_cast<hsize_t>(dim2)};

		compressData(ndims, offset, size,
			h5ivar, h5ispace, h5ovar, h5ospace,
			h5native_type, buffer);

		pos += chunkSize;
	}

	checkH5Err(H5Sclose(h5ispace));
	checkH5Err(H5Dclose(h5ivar));
	checkH5Err(H5Dclose(h5ovar));
	checkH5Err(H5Sclose(h5ospace));
}

template<typename T>
static void compressTimeDataset(hid_t h5ifile, hid_t h5ofile,
	const char* varname, hid_t h5native_type, hid_t h5type,
	void* buffer, unsigned long bufferSize)
{
	// Open original var
	hid_t h5ivar = H5Dopen(h5ifile, varname, H5P_DEFAULT);
	checkH5Err(h5ivar);

	// Get dimension information
	hid_t h5ispace = H5Dget_space(h5ivar);
	checkH5Err(h5ispace);

	int ndims = H5Sget_simple_extent_ndims(h5ispace);
	checkH5Err(ndims);
	if (ndims > 2)
		logError() << "Dimension > 1 are not supported for time datasets";

	hsize_t extent[2];
	checkH5Err(H5Sget_simple_extent_dims(h5ispace, extent, 0L));
	hsize_t timesteps = extent[0];
	hsize_t nelements = extent[1];

	// Create new dataset
	hsize_t chunkSize = bufferSize / sizeof(T);

	hsize_t dims[2] = {timesteps, nelements}; // Change this for other elements
	hid_t h5ospace = H5Screate_simple(ndims, dims, 0L);
	checkH5Err(h5ospace);
	hid_t h5opcreate = H5Pcreate(H5P_DATASET_CREATE);
	checkH5Err(h5opcreate);
	hsize_t chunkDims[2] = {1, std::min(chunkSize, nelements)};
	checkH5Err(H5Pset_chunk(h5opcreate, ndims, chunkDims));
// 	checkH5Err(H5Pset_szip(h5opcreate, H5_SZIP_NN_OPTION_MASK, 4));
	checkH5Err(H5Pset_deflate(h5opcreate, 6));
	hid_t h5ovar = H5Dcreate(h5ofile, varname, h5type, h5ospace,
		H5P_DEFAULT, h5opcreate, H5P_DEFAULT);
	checkH5Err(h5ovar);
	checkH5Err(H5Pclose(h5opcreate));

	// Transfer data
	for (unsigned long t = 0; t < timesteps; t++) {
		unsigned long pos = 0;
		while (pos < nelements) {
			const unsigned long left = nelements - pos;
			if (left < chunkSize)
				chunkSize = left;

			hsize_t offset[2] = {t, pos};
			hsize_t size[2] = {1, chunkSize};

			compressData(ndims, offset, size,
				h5ivar, h5ispace, h5ovar, h5ospace,
				h5native_type, buffer);

			pos += chunkSize;
		}
	}

	checkH5Err(H5Sclose(h5ispace));
	checkH5Err(H5Dclose(h5ivar));
	checkH5Err(H5Dclose(h5ovar));
	checkH5Err(H5Sclose(h5ospace));
}

int main(int argc, char* argv[])
{
	utils::Args args;
	args.addAdditionalOption("input", "input file");
	args.addAdditionalOption("output", "output file", false);

	switch (args.parse(argc, argv)) {
	case utils::Args::Help:
		return 0;
	case utils::Args::Error:
		return 1;
	}

	std::string input = args.getAdditionalArgument<const char*>("input");
	std::string output;
	if (args.isSetAdditional("output"))
		output = args.getAdditionalArgument<const char*>("output");
	else {
		output = input;
		utils::StringUtils::replaceLast(output, ".h5", "");
		output += "_compressed.h5";
	}

	hid_t h5ifile = H5Fopen(input.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	checkH5Err(h5ifile);

	const char* varnames[] = {
		"/connect", "/geometry", "/partition",
		"/sigma_xx", "/sigma_yy", "/sigma_zz",
		"/sigma_xy", "/sigma_yz", "/sigma_xz",
		"/u", "/v", "/w"
	};

	// Get the size
	hid_t h5ivar = H5Dopen(h5ifile, varnames[0], H5P_DEFAULT);
	checkH5Err(h5ivar);
	hid_t h5ispace = H5Dget_space(h5ivar);
	checkH5Err(h5ispace);
	hsize_t extent[2];
	checkH5Err(H5Sget_simple_extent_dims(h5ispace, extent, 0L));
	unsigned long numElements = extent[0];
	checkH5Err(H5Sclose(h5ispace));
	checkH5Err(H5Dclose(h5ivar));

	h5ivar = H5Dopen(h5ifile, varnames[1], H5P_DEFAULT);
	h5ispace = H5Dget_space(h5ivar);
	checkH5Err(H5Sget_simple_extent_dims(h5ispace, extent, 0L));
	unsigned long numVertices = extent[0];
	checkH5Err(H5Sclose(h5ispace));
	checkH5Err(H5Dclose(h5ivar));

	logInfo() << "Found dataset with" << numElements << "elements and" << numVertices << "vertices";

	const unsigned long bufferSize = 256*1024*1024;
	char* buffer = new char[bufferSize];

	hid_t h5ofile = H5Fcreate(output.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	checkH5Err(h5ofile);

	// Connect
	logInfo() << "Compressing connectivity...";
	compressDataset<unsigned long>(h5ifile, h5ofile,
		varnames[0], H5T_NATIVE_UINT64, H5T_STD_U64LE,
		buffer, bufferSize);

	// Connect
	logInfo() << "Compressing geometry...";
	compressDataset<float>(h5ifile, h5ofile,
		varnames[1], H5T_NATIVE_FLOAT, H5T_IEEE_F32LE,
		buffer, bufferSize);

	// Partition
	logInfo() << "Compressing partition...";
	compressDataset<unsigned int>(h5ifile, h5ofile,
		varnames[2], H5T_NATIVE_UINT32, H5T_STD_U32LE,
		buffer, bufferSize);

	for (unsigned int i = 3; i < 12; i++) {
		logInfo() << "Compressing" << utils::nospace << &varnames[i][1] << "...";
		compressTimeDataset<float>(h5ifile, h5ofile,
			varnames[i], H5T_NATIVE_FLOAT, H5T_IEEE_F32LE,
			buffer, bufferSize);
	}

	checkH5Err(H5Fclose(h5ifile));
	checkH5Err(H5Fclose(h5ofile));

	return 0;
}