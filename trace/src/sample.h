#ifndef SAMPLE_20201125_H
#define SAMPLE_20201125_H

#include <time.h>

struct Sample {
    timespec begin;
    timespec end;
    unsigned int loopLength;
    unsigned int subRegion;
};

#endif // SAMPLE_20201125_H
