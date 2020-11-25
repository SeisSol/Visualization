#include "image.h"
#include "input.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <time.h>
#include <utility>
#include <vector>

bool operator<(timespec const& a, timespec const& b) {
    if (a.tv_sec == b.tv_sec) {
        return a.tv_nsec < b.tv_nsec;
    }
    return a.tv_sec < b.tv_sec;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./trace <prefix> <output> [resolution in nsec]" << std::endl;
        return -1;
    }
    const auto prefix = std::string(argv[1]);
    char const* output = argv[2];
    long resolution = argc >= 4 ? atoi(argv[3]) : 1000000000L;

    const std::vector<std::string> names{"computeLocalIntegration.nc",
                                         "computeNeighboringIntegration.nc",
                                         "computeDynamicRupture.nc"};
    std::vector<std::pair<std::vector<int>, std::vector<Sample>>> stats;
    for (auto const& name : names) {
        const auto f = prefix + name;
        stats.emplace_back(read(f.c_str()));
    }

    auto cmp_begin = [](auto const& a, auto const& b) { return a.begin < b.begin; };
    auto cmp_end = [](auto const& a, auto const& b) { return a.end < b.end; };
    auto cmp_subRegion = [](auto const& a, auto const& b) { return a.subRegion < b.subRegion; };

    timespec t_min{std::numeric_limits<time_t>::max(), std::numeric_limits<long>::max()};
    timespec t_max{0, 0};
    unsigned subRegion_max = 0;

    for (const auto& st : stats) {
        auto my_t_min = std::min_element(st.second.begin(), st.second.end(), cmp_begin)->begin;
        t_min = std::min(t_min, my_t_min);
        auto my_t_max = std::max_element(st.second.begin(), st.second.end(), cmp_end)->end;
        t_max = std::max(t_max, my_t_max);
        auto my_subRegion_max =
            std::max_element(st.second.begin(), st.second.end(), cmp_subRegion)->subRegion;
        subRegion_max = std::max(subRegion_max, my_subRegion_max);
    }

    auto im = Image(stats[0].first.size() - 1, subRegion_max + 1, t_min, t_max, resolution);
    for (const auto& st : stats) {
        im.add(st.first, st.second);
    }
    im.write(output);

    return 0;
}
