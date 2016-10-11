/**
 * @file
 * This file is part of SeisSol.
 *
 * @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 **/

#include "KDTree.h"

#include <algorithm>
#include <cstring>
#include <cmath>

KDTree::KDTree(std::unordered_set<Point> const& points, int maxLeafSize, bool split[3])
	: maxLeafN(maxLeafSize)
{
	memcpy(this->split, split, sizeof(bool)*3);

	int n = points.size();

	// Copy data and change row-major to column-major storage.
	data = new Point[n];
	Point* next = data;
	for (std::unordered_set<Point>::const_iterator it = points.begin();
			it != points.end(); it++) {
		memcpy(next->coords, it->coords, sizeof(double)*3);
		next++;
	}

	idx = new int[n];
	for (int i = 0; i < n; ++i) {
		idx[i] = i;
	}

	int maxHeight = 1 + ceil(log2(n / static_cast<double>(maxLeafN)));
	int maxNodes = (1 << maxHeight) - 1;
	nodes = new Node[maxNodes];
	nodes[0].start = 0;
	nodes[0].n = n;

	buildTree(0, 0);
}

KDTree::~KDTree()
{
	delete[] data;
	delete[] idx;
	delete[] nodes;
}

void KDTree::swap(int i, int j)
{
	if (i != j) {
		std::swap(idx[i], idx[j]);
    std::swap(data[i], data[j]);
	}
}

int KDTree::partition(int left, int right, int pivotIdx, int splitdim)
{
	double pivot = data[pivotIdx].coords[splitdim];
	int st = left;
	for (int i = left; i < right; ++i) {
		if (data[i].coords[splitdim] < pivot) {
			swap(st, i);
			++st;
		}
	}
	swap(st, right);
	return st;
}

void KDTree::buildTree(int k, int splitdim)
{
	Node& node = nodes[k];
	node.splitdim = splitdim;
	if (node.n > maxLeafN) {
		int half = (node.n % 2 != 0) ? (node.n + 1)/2 : node.n/2;
		int l = node.start;
		int r = l + node.n - 1;
		int median_idx = l + half;
		if (l != r) {
			int pivotIdx;
			while (true) {
				pivotIdx = r;
				pivotIdx = partition(l, r, pivotIdx, splitdim);
				if (median_idx == pivotIdx) {
					break;
				} else if (median_idx < pivotIdx) {
					r = pivotIdx - 1;
				} else {
					l = pivotIdx + 1;
				}
			}
		}
		node.pivot = data[median_idx].coords[splitdim];

		Node& left = nodes[leftChild(k)];
		left.start = node.start;
		left.n = half;

		Node& right = nodes[rightChild(k)];
		right.start = median_idx;
		right.n = node.n - half;

		int nextSplitdim = splitdim;
		do {
			nextSplitdim = (nextSplitdim + 1) % 3;
		} while (!split[nextSplitdim]);
		buildTree(leftChild(k), nextSplitdim);
		buildTree(rightChild(k), nextSplitdim);
	} else {
		node.isLeaf = true;
	}
}
