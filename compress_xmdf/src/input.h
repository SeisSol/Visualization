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

#ifndef INPUT_H
#define INPUT_H

#include <fcntl.h>
#include <glob.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <sys/stat.h>

#include <hdf5.h>

#include "utils/stringutils.h"

#include "hdf5_helper.h"
#include "output.h"

struct Variable
{
	std::string name;
	unsigned int timesteps;
};

class Input
{
protected:
	size_t m_numElements;
	size_t m_numVertices;
	
	unsigned int m_verticesPerElement;
	
	char* m_buffer;
	
public:
	Input()
		: m_numElements(0), m_numVertices(0), m_verticesPerElement(0)
	{
		m_buffer = new char[OutputVar::CHUNK_SIZE];
	}
	
	virtual ~Input()
	{
		delete [] m_buffer;
	}
	
	size_t numElements() const
	{
		return m_numElements;
	}
	
	size_t numVertices() const
	{
		return m_numVertices;
	}
	
	unsigned int verticesPerElement() const
	{
		return m_verticesPerElement;
	}

	virtual std::vector<Variable> getVarList() = 0;
	
	virtual void writeVariable(const Variable &variable, hid_t nativeType, bool isVertex, OutputVar &writer) = 0;
};


class HDF5Input : public Input
{
private:
	hid_t m_file;

public:
	HDF5Input(const std::string &xdmfFile)
	{
		std::string h5File(xdmfFile);
		utils::StringUtils::replaceLast(h5File, ".xdmf", ".h5");

		logInfo() << "Reading heavy data from" << h5File;

		m_file = H5Fopen(h5File.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		checkH5Err(m_file);
	}

	virtual ~HDF5Input()
	{
		checkH5Err(H5Fclose(m_file));
	}

	std::vector<Variable> getVarList()
	{
		std::vector<Variable> variables;

		hsize_t idx = 0;
		checkH5Err(H5Literate_by_name(m_file, "/", H5_INDEX_NAME, H5_ITER_NATIVE, &idx, extractVars, &variables, H5P_DEFAULT));

		for (std::vector<Variable>::iterator it = variables.begin();
			it != variables.end(); ++it) {
			
			hid_t var = H5Dopen(m_file, ("/"+it->name).c_str(), H5P_DEFAULT);
			checkH5Err(var);
			
			hid_t space = H5Dget_space(var);
			checkH5Err(space);
		
			hsize_t size[2];
			int ndims = H5Sget_simple_extent_dims(space, size, 0L);
			checkH5Err(ndims);
			
			if (it->name == "connect") {
				m_numElements = size[0];
				m_verticesPerElement = size[1];
			} else if (it->name == "geometry") {
				m_numVertices = size[0];
			} else {
				if (ndims > 1)
					it->timesteps = size[0];
				else
					it->timesteps = 0;
			}
			
			checkH5Err(H5Sclose(space));
			checkH5Err(H5Dclose(var));
		}

		return variables;
	}
	
	void writeVariable(const Variable &variable, hid_t nativeType, bool isVertex, OutputVar &writer)
	{
		unsigned int timesteps = std::max(variable.timesteps, 1u);
		
		hid_t var = H5Dopen(m_file, variable.name.c_str(), H5P_DEFAULT);
		checkH5Err(var);
			
		hid_t space = H5Dget_space(var);
		checkH5Err(space);
		
		hsize_t extent[2];
		int ndims = H5Sget_simple_extent_dims(space, extent, 0L);
		checkH5Err(ndims);
		if (ndims > 2)
			logError() << "Dimension > 2 are not supported";
		
		unsigned int dim2 = 1;
		hsize_t nElements;
		
		if (variable.timesteps == 0) {
			nElements = extent[0];
			if (ndims > 1)
				dim2 = extent[1];
		} else {
			nElements = extent[1];
		}

		hsize_t chunkSize = OutputVar::CHUNK_SIZE / dim2 / sizeofType(nativeType);
	
		// Write data
		for (unsigned int t = 0; t < timesteps; t++) {
			unsigned long pos = 0;
			while (pos < nElements) {
				const unsigned long left = nElements - pos;
				hsize_t tmpChunkSize = chunkSize;

				if (left < tmpChunkSize)
					tmpChunkSize = left;

				hsize_t offset[2];
				hsize_t size[2];
				if (variable.timesteps > 0) {
					offset[0] = t;
					offset[1] = pos;
					size[0] = 1;
					size[1] = tmpChunkSize;
				} else {
					offset[0] = pos;
					offset[1] = 0;
					size[0] = tmpChunkSize;
					size[1] = dim2;
				}
				
				hid_t memspace = H5Screate_simple(ndims, size, 0L);
				checkH5Err(memspace);
				
				checkH5Err(H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, 0L, size, 0L));
				checkH5Err(H5Dread(var, nativeType, memspace, space, H5P_DEFAULT, m_buffer));
				
				checkH5Err(H5Sclose(memspace));

				writer.write(m_buffer, nativeType, offset, size);

				pos += chunkSize;
			}
		}

	}

private:
	static herr_t extractVars(hid_t group, const char* name, const H5L_info_t* info, void* op_data)
	{
		std::vector<Variable> *variables = static_cast<std::vector<Variable>*>(op_data);

		Variable var;
		var.name = name;
		var.timesteps = false;
		
		variables->push_back(var);

		return 0;
	}
};

class BinaryInput : public Input
{
private:
	std::string m_fileBase;
	std::string m_fileRegex;
	
	char* m_readBuffer;
	
public:
	BinaryInput(const std::string &xdmfFile)
		: m_fileBase(xdmfFile)
	{
		utils::StringUtils::replaceLast(m_fileBase, ".xdmf", "_");
		m_fileRegex = m_fileBase + "*.bin";
		
		m_readBuffer = new char[2*OutputVar::CHUNK_SIZE];

		logInfo() << "Reading heavy data from" << m_fileRegex.c_str();
	}
	
	virtual ~BinaryInput()
	{
		delete [] m_readBuffer;
	}

	std::vector<Variable> getVarList()
	{
		std::vector<Variable> variables;
		
		glob_t globResult;
		glob(m_fileRegex.c_str(), 0, 0L, &globResult);
		
		for (size_t i = 0; i < globResult.gl_pathc; i++) {
			std::string file(globResult.gl_pathv[i]);
			
			Variable var;
			var.name = file.substr(m_fileRegex.size()-5, file.size()-m_fileRegex.size()+1);
			
			if (var.name == "geometry") {
				int fd = openByVar(var.name);
				m_numVertices = getFileSize(fd) / (3 * sizeof(double));
				close(fd);
			} else if (var.name == "partition") {
				int fd = openByVar(var.name);
				m_numElements = getFileSize(fd) / sizeof(int);
				close(fd);
			}
			
			var.timesteps = 0;
			
			variables.push_back(var);
		}
		
		globfree(&globResult);
		
		// Get the timesteps and the vertices per element
		for (std::vector<Variable>::iterator it = variables.begin();
			it != variables.end(); ++it) {
			
			if (it->name == "connect") {
				int fd = openByVar(it->name);
				m_verticesPerElement = getFileSize(fd) / (m_numElements * sizeof(unsigned long));
				close(fd);
			} else if (it->name == "geometry" || it->name == "partition") {
				// Do nothing
			} else {
				int fd = openByVar(it->name);
				it->timesteps = getFileSize(fd) / (m_numElements * sizeof(double));
				close(fd);
				
				if (it->timesteps == 1)
					it->timesteps = 0;
			}
		}
		
		return variables;
	}
	
	void writeVariable(const Variable &variable, hid_t nativeType, bool isVertex, OutputVar &writer)
	{
		
		int fd = openByVar(variable.name);
		
		size_t typeSize = sizeofType(nativeType);
		if (doCompression(nativeType))
			typeSize *= 2;
		
		size_t fileSize = getFileSize(fd);
		lseek(fd, 0, SEEK_SET);
		
		unsigned int dim2 = 1;
		hsize_t nElements = isVertex ? m_numVertices : m_numElements;
		
		if (variable.timesteps == 0) {
				dim2 = fileSize / (nElements * typeSize);
		}
		
		unsigned int timesteps = std::max(variable.timesteps, 1u);

		hsize_t chunkSize = OutputVar::CHUNK_SIZE / dim2 / sizeofType(nativeType);
	
		// Write data
		for (unsigned int t = 0; t < timesteps; t++) {
			unsigned long pos = 0;
			while (pos < nElements) {
				const unsigned long left = nElements - pos;
				hsize_t tmpChunkSize = chunkSize;

				if (left < tmpChunkSize)
					tmpChunkSize = left;

				hsize_t offset[2];
				hsize_t size[2];
				if (variable.timesteps > 0) {
					offset[0] = t;
					offset[1] = pos;
					size[0] = 1;
					size[1] = tmpChunkSize;
				} else {
					offset[0] = pos;
					offset[1] = 0;
					size[0] = tmpChunkSize;
					size[1] = dim2;
				}
				
				size_t readSize = tmpChunkSize * dim2 * sizeofType(nativeType);
				if (doCompression(nativeType))
					readSize *= 2;
				
				if (read(fd, m_readBuffer, readSize) != readSize)
					logError() << "Could not read data" << readSize;
				
				void* buffer;
				if (doCompression(nativeType)) {
					compress();
					buffer = m_buffer;
				} else {
					buffer = m_readBuffer;
				}

				writer.write(buffer, nativeType, offset, size);

				pos += chunkSize;
			}
		}
		
		if (variable.timesteps > 0) {
		} else {
		}
		
		close(fd);
	}
	
private:
	int openByVar(const std::string &var)
	{
		int fd = open((m_fileBase+var+".bin").c_str(), 0);
		if (fd < 0)
			logError() << "Could not open file" << (m_fileBase+var+".bin").c_str();
		
		return fd;
	}
	
	void compress()
	{
		for (unsigned int i = 0; i < OutputVar::CHUNK_SIZE / sizeof(float); i++) {
			reinterpret_cast<float*>(m_buffer)[i] = reinterpret_cast<double*>(m_readBuffer)[i];
		}
	}
	
private:
	static size_t getFileSize(int fd)
	{
		size_t len = lseek(fd, 0, SEEK_END);
		return len;
	}
	
	static bool doCompression(hid_t type)
	{
		if (type == H5T_NATIVE_FLOAT)
			return true;
		
		return false;
	}
};

#endif // INPUT_H