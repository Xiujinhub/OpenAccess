#ifndef RETINEX_H
#define RETINEX_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mmintrin.h>   //mmx
#include <xmmintrin.h>  //sse
#include <emmintrin.h>  //sse2
#include <pmmintrin.h>  //sse3

class Retinex
{
public:
    Retinex();
    ~Retinex();
    
    /**retinex C++代码实现**/
    cv::Mat single_scale_retinex(const cv::Mat& src, double sigma);//单尺度去雾算法
    void retinex_process(const cv::Mat& src, cv::Mat& dst, double sigma);//
    void multi_retinex_process(const cv::Mat& src, cv::Mat& dst, std::vector<double>& sigma, std::vector<double>& w);//多尺度
    void color_restoration(const cv::Mat& src, cv::Mat& dst, double a, double b);//颜色恢复
    void multi_retinex_color_restoration_process(const cv::Mat& src, cv::Mat& dst, std::vector<double>& sigma, std::vector<double>& w, double G, double t, double a, double b);

};
#endif // RETINEX_H
