#include <iostream>
#include <retinex.h>


#include <windows.h>   // Windows平台编码处理
using namespace std;

int main()
{
  img = cv::imread("image/00003.jpg");
    if (img.empty()) {
        return -1;
    }

    std::vector<double> sigma = { 1, 3, 5 };
    std::vector<double> w(3, 1 / 3.0);
    retinex.multi_retinex_color_restoration_process(img, img, sigma, w, 5, 25, 125, 46);
    cv::imshow("retinex_result", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
    
}
