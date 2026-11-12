#pragma once

#include <estimation/grid_search.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <vector>

namespace delay_estimator
{

struct Estimate
{
    bool valid = false;
    int64_t delay = 0;
    double sd = 0.0;
};

inline Estimate estimate(const SearchResult<int64_t, double>& result)
{
    Estimate estimate;
    const auto& evaluations = result.evaluations;
    const auto best = std::find_if(evaluations.begin(), evaluations.end(), [&](const auto& e) { return e.first == result.value; });

    if (!std::isfinite(result.cost) || result.degrees_of_freedom == 0 || best == evaluations.end())
    {
        return estimate;
    }

    int64_t step = 0;

    if (best != evaluations.begin())
    {
        step = best->first - std::prev(best)->first;
    }

    if (std::next(best) != evaluations.end())
    {
        const int64_t next = std::next(best)->first - best->first;
        step = step == 0 ? next : std::min(step, next);
    }

    if (step <= 0)
    {
        return estimate;
    }

    auto first = best;
    auto last = std::next(best);

    while (first != evaluations.begin() && first->first - std::prev(first)->first == step)
    {
        --first;
    }

    while (last != evaluations.end() && last->first - std::prev(last)->first == step)
    {
        ++last;
    }

    const int64_t width = std::max(result.value - first->first, std::prev(last)->first - result.value);
    const double low = static_cast<double>(first->first - result.value) / static_cast<double>(width);
    const double high = static_cast<double>(std::prev(last)->first - result.value) / static_cast<double>(width);
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d moment = Eigen::Vector3d::Zero();
    int points = 0;

    for (auto it = first; it != last; ++it)
    {
        if (std::isfinite(it->second))
        {
            const double x = static_cast<double>(it->first - result.value) / static_cast<double>(width);
            const Eigen::Vector3d row(1.0, x, x * x);
            normal += row * row.transpose();
            moment += row * it->second;
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

    if (vertex < low || vertex > high || !(minimum > 0.0))
    {
        return estimate;
    }

    estimate.valid = true;
    estimate.delay = result.value + std::llround(vertex * static_cast<double>(width));
    estimate.sd = static_cast<double>(width) * std::sqrt(minimum / (static_cast<double>(result.degrees_of_freedom) * c));
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
