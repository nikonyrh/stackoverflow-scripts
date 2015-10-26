#include <unordered_set>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "Utils.hpp"


typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;

// Maybe there is a simple 64-bit solution out there?
__host__ __device__ inline int hammingWeight(uint32_t v)
{
	v = v - ((v>>1) & 0x55555555);
	v = (v & 0x33333333) + ((v>>2) & 0x33333333);

	return ((v + (v>>4) & 0xF0F0F0F) * 0x1010101) >> 24;
}

__host__ __device__ inline int hammingDistance(const uint64_t a, const uint64_t b)
{
	const uint64_t delta = a ^ b;
	return hammingWeight(delta & 0xffffffffULL) + hammingWeight(delta >> 32);
}

struct HammingDistanceFilter
{
	const uint64_t _target, _maxDistance;

	HammingDistanceFilter(const uint64_t target, const uint64_t maxDistance) :
			_target(target), _maxDistance(maxDistance) {
	}

	__host__ __device__ bool operator()(const uint64_t hash) {
		return hammingDistance(_target, hash) <= _maxDistance;
	}
};

void findHashes(
	const thrust::host_vector<uint64_t> hashesCpu,
	const std::unordered_set<uint64_t> hashesToSearch,
	const int maxDistance
) {
	/*std::cout << hashesToSearch.size() << " hashes to find form " <<
		hashesCpu.size() << ", " << maxDistance << " max distance" << std::endl;*/

	thrust::device_vector<uint64_t> hashesGpu = hashesCpu;
	thrust::device_vector<uint64_t> matchesGpu(hashesCpu.size());
	
	std::vector<double> durations;
	std::vector<size_t> nMatches;
	
	for (auto it = hashesToSearch.begin(); it != hashesToSearch.end(); ++it) {
		CHRTimer timer;
		timer.Start();
		matchesGpu.clear();
		
		const auto matchesGpuEnd = thrust::copy_if(
			hashesGpu.cbegin(), hashesGpu.cend(), matchesGpu.begin(), HammingDistanceFilter(*it, maxDistance)
		);
		thrust::system::cuda::detail::synchronize();
		
		nMatches.push_back(matchesGpuEnd - matchesGpu.begin());

		thrust::host_vector<uint64_t> matchesCpu(nMatches.back());
		thrust::copy(matchesGpu.begin(), matchesGpuEnd, matchesCpu.begin());
		thrust::system::cuda::detail::synchronize();
		
		timer.Stop();
		durations.push_back(timer.GetElapsedAsSeconds() * 1000.0);

#if 0
		std::set<uint64_t> uniqueMatches(matchesCpu.begin(), matchesCpu.end());
		std::cout << nMatches.back() << " matches (" << uniqueMatches.size() << " unique) for\n    " << Utils::toBinary(*it) << "\n";

		for (auto it2 = uniqueMatches.begin(); it2 != uniqueMatches.end(); ++it2) {
			std::cout << "  - " << Utils::toBinary(*it2) << "\n";
		}
		std::cout << std::endl;
#endif
	}
	
	std::sort(durations.begin(), durations.end());
	std::sort(nMatches.begin(), nMatches.end());

	const int percentiles[] = {5, 25, 50, 75, 95, 99, -1};
	char buffer[2048];

#if 0
	for (int i = 0; percentiles[i] >= 0; ++i) {
		const int j = (int)(0.5 + percentiles[i] * 0.01 * (hashesToSearch.size() - 1));
		sprintf_s(buffer, 2048, "%3dth: %9.6f sec, %8d matches\n", percentiles[i], durations[j], nMatches[j]);
		std::cout << buffer;
	}
#else
	char *bufferPtr = buffer;
	bufferPtr += sprintf_s(bufferPtr, 2048, "%5.2f", hashesCpu.size() * 0.000001);
	
	for (int i = 0; percentiles[i] >= 0; ++i) {
		const int j = (int)(0.5 + percentiles[i] * 0.01 * (hashesToSearch.size() - 1));
		bufferPtr += sprintf_s(bufferPtr, 2048, " %9.6f", durations[j]);
	}

	for (int i = 0; percentiles[i] >= 0; ++i) {
		const int j = (int)(0.5 + percentiles[i] * 0.01 * (hashesToSearch.size() - 1));
		bufferPtr += sprintf_s(bufferPtr, 2048, " %7d", nMatches[j]);
	}
	std::cout << buffer;
#endif

	std::cout << std::endl;
}

/*
	Exmple parameters: 500 1 2 3 4 5 to run 500 searaches from 1, 2, 3, 4 and 5 million 64-bit hashes
*/
int main(int argc, char **argv)
{
    const int nSearches = argc >= 2 ? atoi(argv[1]) : 50;

	std::vector<int> nHashes;

	if (argc >= 3) {
		for (int i = 2; i < argc; ++i) {
			nHashes.push_back(atoi(argv[i]) * 1000000);
		}
	}
	else {
		nHashes.push_back(100000);
	}

	thrust::host_vector<uint64_t> hashesCpu(nHashes.back());
	std::unordered_set<uint64_t> hashesToSearch(nSearches);

	{
		// Use py/generate_to_file.py to generate this file
		std::ifstream numbers("numbers.txt");

		for (int i = 0; i < nHashes.back(); ++i) {
			numbers >> hashesCpu[i];
		}
	}
	
    for (int i = 0; hashesToSearch.size() < nSearches && i < hashesCpu.size(); ++i) {
		hashesToSearch.insert(hashesCpu[i]);
	}
	
	for (auto it = nHashes.begin(); it != nHashes.end(); ++it) {
		findHashes(thrust::host_vector<uint64_t>(hashesCpu.begin(), hashesCpu.begin() + *it), hashesToSearch, 8);
	}
	
    return 0;
}
