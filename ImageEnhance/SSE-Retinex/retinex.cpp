/***Retinex***/

cv::Mat Retinex::single_scale_retinex(const cv::Mat& src, double sigma)
{
    cv::Mat gaussImg, log_src(src.size(), CV_32F);
    cv::Mat difference(src.size(), CV_32F);
    // 对输入图像进行高斯模糊处理
    cv::GaussianBlur(src, gaussImg, cv::Size(0, 0), sigma);
    // 将输入和模糊后的图像转换为32位浮点类型
    gaussImg.convertTo(gaussImg, CV_32F);
    src.convertTo(log_src, CV_32F);
// 使用SSE加速逐元素操作
    #pragma omp parallel for
    for (int i = 0; i < gaussImg.rows; i++)
    {
        for (int j = 0; j < gaussImg.cols; j += 4)  // 每次处理4个像素
        {
            // 使用SSE加载4个像素数据
            __m128 src_vals = _mm_loadu_ps(&log_src.at<float>(i, j));  // 加载原图数据
            __m128 gauss_vals = _mm_loadu_ps(&gaussImg.at<float>(i, j));  // 加载高斯模糊图数据

            // 对原图和高斯图进行log运算
            __m128 log_src_vals = _mm_add_ps(src_vals, _mm_set1_ps(1.0f));  // log(src + 1)
            __m128 log_gauss_vals = _mm_add_ps(gauss_vals, _mm_set1_ps(1.0f));  // log(gaussImg + 1)

            log_src_vals = _mm_log_ps(log_src_vals);  // 计算log
            log_gauss_vals = _mm_log_ps(log_gauss_vals);  // 计算log

            // 计算差值
            __m128 diff_vals = _mm_sub_ps(log_src_vals, log_gauss_vals);

            // 将计算结果存储回矩阵
            _mm_storeu_ps(&difference.at<float>(i, j), diff_vals);  // 存储差分结果
        }
    }
    gaussImg.release();
    log_src.release();
    return difference;
}

void Retinex::retinex_process(const cv::Mat& src, cv::Mat& dst, double sigma)
{
    std::vector<cv::Mat> channels;
    std::vector<cv::Mat> channels_dst;
    cv::split(src, channels);
#pragma omp parallel
    for (int i = 0; i < channels.size(); i++)
    {
        channels_dst.push_back(single_scale_retinex(channels[i], sigma));
    }
    cv::merge(channels_dst, dst);
}

void Retinex::multi_retinex_process(const cv::Mat& src, cv::Mat& dst, std::vector<double>& sigma, std::vector<double>& w)
{
    std::vector<cv::Mat> temp(sigma.size());
#pragma omp parallel for
    for (int i = 0; i < sigma.size(); i++)
    {
        retinex_process(src, temp[i], sigma[i]);
    }
    // 用加权平均法合成多尺度Retinex图像
    dst = cv::Mat::zeros(src.size(), CV_32FC3);
    for (int i = 0; i < sigma.size(); i++)
    {
        dst += w[i] * temp[i];
    }
}

void Retinex::color_restoration(const cv::Mat& src, cv::Mat& dst, double a, double b)
{
    // 将输入图像转换为浮动点类型
    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);
    // 计算通道的sum值
    cv::Mat sum = cv::Mat::zeros(src.size(), CV_32F);
    std::vector<cv::Mat> channels(3);
    cv::split(srcFloat, channels);
    // 计算每个通道的sum
    for (int i = 0; i < 3; ++i)
    {
        sum += channels[i];
    }
#pragma omp parallel
    dst = cv::Mat(src.size(), CV_32FC3);
    for (int i = 0; i < 3; ++i)
    {
        // 使用 log 函数进行颜色恢复
        cv::Mat channel_log;
        cv::log(a * channels[i] + 1.0, channel_log);  // 防止对零取对数

        cv::Mat sum_log;
        cv::log(sum + 1.0, sum_log);  // 防止对零取对数

        channels[i] = b * (channel_log - sum_log);
    }
    cv::merge(channels, dst);
}

void Retinex::multi_retinex_color_restoration_process(const cv::Mat& src, cv::Mat& dst, std::vector<double>& sigma, std::vector<double>& w, double G, double t, double a, double b)
{
    dst.convertTo(dst, CV_32FC3);
    cv::Mat multiRSImg(src.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    multi_retinex_process(src, multiRSImg, sigma, w);
    cv::Mat colorResImg(src.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    color_restoration(src, colorResImg, a, b);
// 并行化颜色恢复的多尺度Retinex计算
#pragma omp parallel
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            for (int c = 0; c < 3; c++)  // 遍历每个通道
            {
                dst.at<cv::Vec3f>(i, j)[c] = cv::saturate_cast<uchar>(G * ((multiRSImg.at<cv::Vec3f>(i, j)[c] * colorResImg.at<cv::Vec3f>(i, j)[c]) + t));
            }
        }
    }
    // 归一化结果
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
}
