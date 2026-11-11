#pragma once

#include <estimation/grid_search.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace delay_estimator
{

struct Estimate
{
    bool valid = false;
    int64_t delay = 0;
    double sd = 0.0;
};

inline Estimate estimate(const SearchResult<int64_t, double>& result, int64_t width, std::size_t degrees_of_freedom)
{
    Estimate estimate;

    if (!std::isfinite(result.cost) || width <= 0 || degrees_of_freedom == 0)
    {
        return estimate;
    }

    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d moment = Eigen::Vector3d::Zero();
    int points = 0;

    for (const auto& [delay, cost] : result.evaluations)
    {
        if (std::isfinite(cost) && std::llabs(delay - result.value) <= width)
        {
            const double x = static_cast<double>(delay - result.value) / static_cast<double>(width);
            const Eigen::Vector3d row(1.0, x, x * x);
            normal += row * row.transpose();
            moment += row * cost;
            ++points;
        }
    }

    if (points < 3)
    {
        return estimate;
    }

    const Eigen::Vector3d coefficients = normal.ldlt().solve(moment);
    const double a = coefficients[0];
    const double b = coefficients[1];
    const double c = coefficients[2];

    if (!(c > 0.0))
    {
        return estimate;
    }

    const double vertex = -b / (2.0 * c);
    const double minimum = a - b * b / (4.0 * c);

    if (std::abs(vertex) > 1.0 || !(minimum > 0.0))
    {
        return estimate;
    }

    estimate.valid = true;
    estimate.delay = result.value + std::llround(vertex * static_cast<double>(width));
    estimate.sd = static_cast<double>(width) * std::sqrt(minimum / (static_cast<double>(degrees_of_freedom) * c));
    return estimate;
}

inline Estimate fuse(const std::vector<Estimate>& estimates)
{
    double weights = 0.0;
    double weighted = 0.0;

    for (const Estimate& e : estimates)
    {
        if (e.valid)
        {
            const double sd = std::max(e.sd, 1.0);
            const double weight = 1.0 / (sd * sd);
            weights += weight;
            weighted += weight * static_cast<double>(e.delay);
        }
    }

    Estimate fused;

    if (weights > 0.0)
    {
        fused.valid = true;
        fused.delay = std::llround(weighted / weights);
        fused.sd = 1.0 / std::sqrt(weights);
    }

    return fused;
}

}
