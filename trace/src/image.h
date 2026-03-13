#ifndef IMAGE_20201125_H
#define IMAGE_20201125_H

#include "sample.h"

#include <time.h>
#include <tuple>
#include <vector>

inline long long difftime(const timespec& start, const timespec& end) {
  return 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
}

class Image {
  public:
  static constexpr int Comp = 3;
  static constexpr int Offset = 5;
  static constexpr unsigned int Colors[] = {0xff0000, 0x00ff00, 0x0000ff};

  Image(int nranks,
        int nsubregions,
        const timespec& t_min,
        const timespec& t_max,
        long resolution = 1000000000L);
  ~Image();
  void add(const std::vector<int>& offset, const std::vector<Sample>& sample);
  void write(const char* file_name);

  private:
  std::tuple<long long, double, long long, double> time_slots(const timespec& begin,
                                                              const timespec& end);
  unsigned char* pixel(int rank, long long time_slot);
  void add_pixels(int rank, const timespec& begin, const timespec& end, int subRegion);

  int nsubregions_;
  timespec t_min_, t_max_;
  long long resolution_;
  int region_ = 0;

  int width_, height_, stride_;
  unsigned char* px_;
};

#endif // IMAGE_20201125_H
