#include "input.h"

#include <netcdf.h>
#include <stdio.h>
#include <stdlib.h>

void check_err(const int stat, const int line, const char* file) {
    if (stat != NC_NOERR) {
        (void)fprintf(stderr, "line %d of %s: %s\n", line, file, nc_strerror(stat));
        fflush(stderr);
        exit(1);
    }
}

std::pair<std::vector<int>, std::vector<Sample>> read(char const* file_name) {
    int ncid;

    /* dimension ids */
    int rank_dim;
    int sample_dim;

    /* dimension lengths */
    size_t rank_len;
    size_t sample_len;

    /* variable ids */
    int offset_id;
    int sample_id;

    size_t start[] = {0};
    size_t count[1];

    check_err(nc_open(file_name, 0, &ncid), __LINE__, __FILE__);

    /* get dimensions */
    check_err(nc_inq_dimid(ncid, "rank", &rank_dim), __LINE__, __FILE__);
    check_err(nc_inq_dimlen(ncid, rank_dim, &rank_len), __LINE__, __FILE__);
    check_err(nc_inq_dimid(ncid, "sample", &sample_dim), __LINE__, __FILE__);
    check_err(nc_inq_dimlen(ncid, sample_dim, &sample_len), __LINE__, __FILE__);

    auto offset = std::vector<int>(rank_len);
    auto sample = std::vector<Sample>(sample_len);

    /* get variable ids */
    check_err(nc_inq_varid(ncid, "offset", &offset_id), __LINE__, __FILE__);
    check_err(nc_inq_varid(ncid, "sample", &sample_id), __LINE__, __FILE__);

    count[0] = rank_len;
    check_err(nc_get_vara(ncid, offset_id, start, count, offset.data()), __LINE__, __FILE__);
    count[0] = sample_len;
    check_err(nc_get_vara(ncid, sample_id, start, count, sample.data()), __LINE__, __FILE__);

    check_err(nc_close(ncid), __LINE__, __FILE__);

    return {offset, sample};
}
