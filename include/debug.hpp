#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace delay_estimator
{

class SearchPlot
{
public:
    explicit SearchPlot(cv::Size size)
        : _size(size)
    {
    }

    void setAxis(double min_x, double max_x, double min_y, double max_y)
    {
        _min = {min_x, min_y};
        _max = {max_x, max_y};
    }

    void add(const std::string& name, const std::vector<double>& x, const std::vector<double>& y)
    {
        Curve curve{name, {}, {}};

        for (std::size_t i = 0; i < std::min(x.size(), y.size()); ++i)
        {
            if (std::isfinite(x[i]) && std::isfinite(y[i]))
            {
                curve.x.push_back(x[i]);
                curve.y.push_back(y[i]);
            }
        }

        _curves.push_back(std::move(curve));
    }

    void draw()
    {
        cv::Mat image(_size, CV_8UC3, cv::Scalar::all(255));
        bool grid = true;

        for (std::size_t i = 0; i < _curves.size(); ++i)
        {
            grid = !draw(image, _curves[i], _colors[i % _colors.size()], static_cast<int>(i), grid) && grid;
        }

        _curves.clear();
        std::lock_guard<std::mutex> lock(_mutex);
        _image = image;
    }

    void show()
    {
        cv::Mat image;

        {
            std::lock_guard<std::mutex> lock(_mutex);
            image = _image;
            _image.release();
        }

        if (!image.empty())
        {
            cv::imshow("Delay search", image);
        }

        cv::waitKey(1);
    }

private:
    struct Curve
    {
        std::string name;
        std::vector<double> x;
        std::vector<double> y;
    };

    inline static const std::array<cv::Scalar, 2> _colors = {cv::Scalar(180, 100, 30), cv::Scalar(30, 150, 30)};

    cv::Size _size;
    cv::Point2d _min{0.0, 0.0};
    cv::Point2d _max{1.0, 1.0};
    std::vector<Curve> _curves;
    std::mutex _mutex;
    cv::Mat _image;

    bool draw(cv::Mat& image, const Curve& curve, const cv::Scalar& color, int row, bool grid) const
    {
        if (curve.y.size() < 2)
        {
            return false;
        }

        const cv::Scalar background = cv::Scalar::all(255);
        cv::Mat plot;
        cv::Ptr<cv::plot::Plot2d> plot2d = cv::plot::Plot2d::create(cv::Mat(curve.x), cv::Mat(curve.y));
        plot2d->setPlotSize(_size.width, _size.height);
        plot2d->setMinX(_min.x);
        plot2d->setMaxX(_max.x);
        plot2d->setMinY(_min.y);
        plot2d->setMaxY(_max.y);
        plot2d->setInvertOrientation(true);
        plot2d->setShowText(false);
        plot2d->setShowGrid(grid);
        plot2d->setGridLinesNumber(10);
        plot2d->setPlotGridColor(cv::Scalar::all(220));
        plot2d->setPlotBackgroundColor(background);
        plot2d->setPlotAxisColor(background);
        plot2d->setPlotLineColor(color);
        plot2d->setPlotLineWidth(2);
        plot2d->render(plot);
        cv::resize(plot, plot, _size, 0.0, 0.0, cv::INTER_AREA);

        cv::Mat line;
        cv::inRange(plot, background, background, line);
        plot.copyTo(image, ~line);

        const std::size_t best = static_cast<std::size_t>(std::min_element(curve.y.begin(), curve.y.end()) - curve.y.begin());
        const cv::Point p(
            static_cast<int>((curve.x[best] - _min.x) / (_max.x - _min.x) * (_size.width - 1)),
            _size.height - 1 - static_cast<int>((curve.y[best] - _min.y) / (_max.y - _min.y) * (_size.height - 1)));
        cv::circle(image, p, 5, color, cv::FILLED, cv::LINE_AA);
        cv::putText(image, cv::format("%s %.0f", curve.name.c_str(), curve.x[best]),
            cv::Point(10, 25 + 25 * row), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv::LINE_AA);
        return true;
    }
};

}
