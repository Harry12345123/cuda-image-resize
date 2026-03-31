#include <opencv2/opencv.hpp>
#include <iostream>
void resize_cuda(const unsigned char *in, unsigned char *out, int in_w, int in_h, int out_w, int out_h, int channels);

int main(int argc, char **argv)
{
    std::string input = argc > 1 ? argv[1] : "input.jpg";
    cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
    if (img.empty())
        return -1;
    int out_w = img.cols / 2;
    int out_h = img.rows / 2;
    cv::Mat out(out_h, out_w, img.type());
    resize_cuda(img.data, out.data, img.cols, img.rows, out_w, out_h, img.channels());
    cv::imwrite("output_resize.jpg", out);
    return 0;
}