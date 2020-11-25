#include "image.h"
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <exception>

Image::Image(int nranks, int nsubregions, timespec const& t_min, timespec const& t_max,
             long resolution)
    : nsubregions_(nsubregions), t_min_(t_min), t_max_(t_max), resolution_(resolution) {
    auto time_diff = difftime(t_min, t_max);
    std::size_t time_slots = 1 + (time_diff - 1) / resolution;

    std::size_t root = sqrt(time_slots / static_cast<double>(nranks + Offset));
    width_ = 1 + (time_slots - 1) / (root);
    height_ = nranks * root + Offset * (root - 1);
    stride_ = width_ * (nranks + Offset);

    auto npixels = width_ * height_ * Comp;
    px_ = new unsigned char[npixels];
    memset(px_, 0, npixels * sizeof(unsigned char));

    for (int r = 0; r < root - 1; ++r) {
        for (int o = 0; o < nranks; ++o) {
            for (int c = 0; c < Comp * width_; ++c) {
                px_[Comp * (r * stride_ + o * width_) + c] = 255;
            }
        }
    }
}

Image::~Image() { delete[] px_; }

std::tuple<long long, double, long long, double> Image::time_slots(timespec const& begin,
                                                                   timespec const& end) {
    auto d1 = difftime(t_min_, begin);
    auto d2 = difftime(t_min_, end);
    long long t1 = d1 / resolution_;
    long long t2 = 1 + (d2 - 1) / resolution_;
    double w1 = static_cast<double>(d1) / resolution_ - t1;
    double w2 = t2 - static_cast<double>(d2) / resolution_;
    return {t1, w1, t2, w2};
}

unsigned char* Image::pixel(int rank, long long time_slot) {
    auto const block_row = time_slot / width_;
    auto const block_begin = time_slot % width_;
    return px_ + Comp * (block_row * stride_ + rank * width_ + block_begin);
}

void Image::add_pixels(int rank, timespec const& begin, timespec const& end, int subRegion) {
    if (region_ >= sizeof(Colors) / sizeof(int)) {
        throw std::runtime_error("OMG! Not enough colours.");
    }
    auto [t1, w1, t2, w2] = time_slots(begin, end);
    for (auto t = t1; t <= t2; ++t) {
        double w = t == t1 ? w1 : (t == t2 ? w2 : 1.0);
        unsigned char* px = pixel(rank, t);
        for (int c = 0; c < Comp; ++c) {
            int channel = 8 * (Comp - c - 1);
            unsigned char col = (Colors[region_] & (0xff << channel)) >> channel;
            col *= (nsubregions_ - subRegion) / static_cast<double>(nsubregions_);
            px[c] = w * col + (1.0 - w) * px[c];
        }
    }
}

void Image::add(std::vector<int> const& offset, std::vector<Sample> const& sample) {
    for (std::size_t i = 0; i < offset.size() - 1; ++i) {
        for (int j = offset[i]; j < offset[i + 1]; ++j) {
            add_pixels(i, sample[j].begin, sample[j].end, sample[j].subRegion);
        }
    }
    ++region_;
}

void Image::write(char const* file_name) {
    stbi_write_png(file_name, width_, height_, Comp, px_, width_ * Comp);
}
