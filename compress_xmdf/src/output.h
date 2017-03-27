/**
 * @file
 * This file is part of SeisSol.
 *
 * @author Sebastian Rettenberger (sebastian.rettenberger AT tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger)
 *
 * @section LICENSE
 * Copyright (c) 2017, SeisSol Group
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

#ifndef OUTPUT_H
#define OUTPUT_H

#include <hdf5.h>

#include "utils/logger.h"

#include "hdf5_helper.h"

class OutputVar
{
private:
	hid_t m_var;
	hid_t m_space;
	
	unsigned int m_ndims;
	
public:
	OutputVar(hid_t file, const char* name, hid_t type, unsigned int timesteps, size_t nElements, unsigned int dim2,
			unsigned int compressionLevel)
	{
		if (timesteps > 0 && dim2 > 0)
			logError() << "Time data set with multiple dimensions is not supported.";
		
		m_ndims = 2;
		if (timesteps == 0 && nElements == 0)
			m_ndims = 1;
		
		bool hasTime = true;
		if (timesteps == 0) {
			timesteps = 1;
			hasTime = false;
		}
		if (dim2 == 0)
			dim2 = 1;
		
		// Create new dataset
		hsize_t chunkSize = CHUNK_SIZE / dim2 / sizeofType(type);
		
		hsize_t dims[2];
		if (hasTime) {
			dims[0] = timesteps;
			dims[1] = nElements;
		} else {
			dims[0] = nElements;
			dims[1] = dim2;
		}
		
		m_space = H5Screate_simple(m_ndims, dims, 0L);
		checkH5Err(m_space);
		hid_t h5pcreate = H5Pcreate(H5P_DATASET_CREATE);
		checkH5Err(h5pcreate);
		hsize_t chunkDims[2];
		if (hasTime) {
			chunkDims[0] = 1;
			chunkDims[1] = std::min(chunkSize, static_cast<hsize_t>(nElements));
		} else {
			chunkDims[0] = std::min(chunkSize, static_cast<hsize_t>(nElements));
			chunkDims[1] = dim2;
		}
		checkH5Err(H5Pset_chunk(h5pcreate, m_ndims, chunkDims));
// 		checkH5Err(H5Pset_szip(h5opcreate, H5_SZIP_NN_OPTION_MASK, 4));
		checkH5Err(H5Pset_deflate(h5pcreate, compressionLevel));
		m_var = H5Dcreate(file, name, type, m_space,
			H5P_DEFAULT, h5pcreate, H5P_DEFAULT);
		checkH5Err(m_var);
		checkH5Err(H5Pclose(h5pcreate));
	}
	
	~OutputVar()
	{
		checkH5Err(H5Dclose(m_var));
		checkH5Err(H5Sclose(m_space));
	}
	
	void write(void* buffer, hid_t nativeType, hsize_t offset[2], hsize_t size[2])
	{
		hid_t memspace = H5Screate_simple(m_ndims, size, 0L);
		checkH5Err(memspace);

		checkH5Err(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, offset, 0L, size, 0L));
		checkH5Err(H5Dwrite(m_var, nativeType, memspace, m_space, H5P_DEFAULT, buffer));

		checkH5Err(H5Sclose(memspace));
	}
	
public:
	static const size_t CHUNK_SIZE = 256*1024*1024;
};

#endif // OUTPUT_H