
/*** SSE-Retinex ***/
cv::Mat Retinex::single_scale_retinex(const cv::Mat& src, double sigma)
{
    cv::Mat gaussImg;
    cv::GaussianBlur(src, gaussImg, cv::Size(0, 0), sigma);

    cv::Mat srcFloat, gaussFloat;
    src.convertTo(srcFloat, CV_32F);
    gaussImg.convertTo(gaussFloat, CV_32F);

    cv::Mat difference(src.size(), CV_32F);

#pragma omp parallel for
    for (int i = 0; i < srcFloat.rows; i++)
    {
        // 使用行指针提高效率
        const float* src_row = srcFloat.ptr<float>(i);
        const float* gauss_row = gaussFloat.ptr<float>(i);
        float* diff_row = difference.ptr<float>(i);

        int j = 0;
        // 使用SSE一次处理4个像素（替换AVX2）
        for (; j <= srcFloat.cols - 4; j += 4)
        {
            // 使用SSE指令加载数据
            __m128 src_vals = _mm_loadu_ps(src_row + j);
            __m128 gauss_vals = _mm_loadu_ps(gauss_row + j);
            __m128 one = _mm_set1_ps(1.0f);

            // 计算 log(src + 1)
            __m128 log_src_vals = _mm_log_ps(_mm_add_ps(src_vals, one));

            // 计算 log(gauss + 1)
            __m128 log_gauss_vals = _mm_log_ps(_mm_add_ps(gauss_vals, one));

            // 计算差值
            __m128 diff_vals = _mm_sub_ps(log_src_vals, log_gauss_vals);
            _mm_storeu_ps(diff_row + j, diff_vals);
        }

        // 处理剩余像素
        for (; j < srcFloat.cols; j++)
        {
            diff_row[j] = std::log(src_row[j] + 1.0f) - std::log(gauss_row[j] + 1.0f);
        }
    }

    return difference;
}


void Retinex::retinex_process(const cv::Mat& src, cv::Mat& dst, double sigma)
{
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    // 预分配空间避免多线程竞争
    std::vector<cv::Mat> channels_dst(channels.size());

#pragma omp parallel for
    for (int i = 0; i < channels.size(); i++)
    {
        channels_dst[i] = single_scale_retinex(channels[i], sigma);
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

    dst = cv::Mat::zeros(src.size(), CV_32FC3);
    for (int i = 0; i < sigma.size(); i++)
    {
        dst += w[i] * temp[i];
    }
}

void Retinex::color_restoration(const cv::Mat& src, cv::Mat& dst, double a, double b)
{
    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    // 计算三通道总和
    cv::Mat sum = cv::Mat::zeros(src.size(), CV_32F);

#pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        const cv::Vec3f* src_row = srcFloat.ptr<cv::Vec3f>(i);
        float* sum_row = sum.ptr<float>(i);

        for (int j = 0; j < src.cols; j++)
        {
            sum_row[j] = src_row[j][0] + src_row[j][1] + src_row[j][2] + FLT_MIN;
        }
    }

    dst.create(src.size(), CV_32FC3);

#pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        const cv::Vec3f* src_row = srcFloat.ptr<cv::Vec3f>(i);
        const float* sum_row = sum.ptr<float>(i);
        cv::Vec3f* dst_row = dst.ptr<cv::Vec3f>(i);

        for (int j = 0; j < src.cols; j++)
        {
            float log_sum = std::log(sum_row[j]);

            dst_row[j][0] = b * (std::log(a * src_row[j][0] + 1.0f) - log_sum);
            dst_row[j][1] = b * (std::log(a * src_row[j][1] + 1.0f) - log_sum);
            dst_row[j][2] = b * (std::log(a * src_row[j][2] + 1.0f) - log_sum);
        }
    }
}

void Retinex::multi_retinex_color_restoration_process(const cv::Mat& src, cv::Mat& dst,
                                                      std::vector<double>& sigma, std::vector<double>& w,
                                                      double G, double t, double a, double b)
{
    cv::Mat multiRSImg;
    multi_retinex_process(src, multiRSImg, sigma, w);

    cv::Mat colorResImg;
    color_restoration(src, colorResImg, a, b);

    dst.create(src.size(), CV_32FC3);

#pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        const cv::Vec3f* multi_row = multiRSImg.ptr<cv::Vec3f>(i);
        const cv::Vec3f* color_row = colorResImg.ptr<cv::Vec3f>(i);
        cv::Vec3f* dst_row = dst.ptr<cv::Vec3f>(i);

        for (int j = 0; j < src.cols; j++)
        {
            dst_row[j][0] = G * (multi_row[j][0] * color_row[j][0] + t);
            dst_row[j][1] = G * (multi_row[j][1] * color_row[j][1] + t);
            dst_row[j][2] = G * (multi_row[j][2] * color_row[j][2] + t);
        }
    }

    // 归一化结果
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
}
