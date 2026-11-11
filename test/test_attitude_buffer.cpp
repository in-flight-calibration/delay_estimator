#include <input/attitude.hpp>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

namespace
{

bool check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAILED: %s\n", message);
    }

    return condition;
}

Eigen::Quaterniond rotation(double t)
{
    return Eigen::Quaterniond(Eigen::AngleAxisd(0.8 * t, Eigen::Vector3d(1.0, 2.0, -0.5).normalized()));
}

}

int main()
{
    using delay_estimator::Attitude;
    using delay_estimator::AttitudeUtils;
    const AttitudeUtils slerp;
    bool ok = true;
    const int64_t step = 5'000'000;

    std::vector<Attitude> samples;

    for (int i = 0; i <= 400; ++i)
    {
        const Eigen::Quaterniond q = rotation(static_cast<double>(i * step) * 1e-9);
        samples.push_back({i * step, i % 2 == 0 ? q : Eigen::Quaterniond(-q.coeffs())});
    }

    std::mt19937_64 rng(3);
    std::uniform_int_distribution<int64_t> time(0, 400 * step);
    std::vector<int64_t> queries(500);

    for (int64_t& t : queries)
    {
        t = time(rng);
    }

    std::sort(queries.begin(), queries.end());
    const ifc_tools::Interpolator interpolator(samples);
    const auto result = interpolator.interpolate(queries, slerp);
    double worst = 0.0;

    for (std::size_t k = 0; k < queries.size(); ++k)
    {
        worst = std::max(worst, result[k]->q.angularDistance(rotation(static_cast<double>(queries[k]) * 1e-9)));
    }

    ok &= check(worst < 1e-9, "constant-rate rotation is interpolated exactly despite q/-q flips");

    const auto outside = interpolator.interpolate({-1, 400 * step + 1}, slerp);
    ok &= check(!outside[0] && !outside[1], "queries outside the buffer return nothing");

    return ok ? 0 : 1;
}
