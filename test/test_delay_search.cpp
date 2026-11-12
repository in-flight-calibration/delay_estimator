#include <alignment/sample_synchronizer.hpp>
#include <estimation/fusion.hpp>
#include <estimation/grid_search.hpp>
#include <method/flow_method.hpp>
#include <method/horizon_method.hpp>

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>

namespace
{

constexpr double width = 640.0;
constexpr double height = 480.0;

bool check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAILED: %s\n", message);
    }

    return condition;
}

double sweep(double t, double f0, double f1, double duration)
{
    const double k = std::log(f1 / f0) / duration;
    return 2.0 * M_PI * f0 * (std::exp(k * t) - 1.0) / k;
}

Eigen::Quaterniond excited(double t)
{
    const double roll = 0.25 * std::sin(sweep(t, 0.3, 2.5, 8.0));
    const double pitch = 0.15 * std::sin(sweep(t, 2.0, 0.4, 8.0) + 0.7);
    const double yaw = 0.3 * std::sin(2.0 * M_PI * 0.15 * t);
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
}

Eigen::Quaterniond still(double)
{
    return Eigen::Quaterniond::Identity();
}

Eigen::Vector3d steady(double t)
{
    return {20.0 * t, 0.0, -120.0};
}

Eigen::Vector3d manoeuvring(double t)
{
    return {20.0 * t + 6.0 * std::sin(0.8 * t), 4.0 * std::sin(0.5 * t), -120.0 + 3.0 * std::sin(0.9 * t)};
}

template <typename AttitudeFunction, typename PositionFunction>
delay_estimator::SearchResult<int64_t, double> simulate(AttitudeFunction attitude, PositionFunction position, double true_delay_s, double min_depth = 40.0, double max_depth = 200.0)
{
    using namespace delay_estimator;

    std::vector<Attitude> attitude_samples;
    std::vector<LocalPosition> position_samples;

    for (int i = 0; i <= 1600; ++i)
    {
        const double t = i * 0.005;
        attitude_samples.push_back({std::llround(t * 1e9), attitude(t)});
        position_samples.push_back({std::llround(t * 1e9), position(t)});
    }

    Eigen::Matrix3d r_bc;
    r_bc.col(0) = Eigen::Vector3d::UnitY();
    r_bc.col(1) = Eigen::Vector3d::UnitZ();
    r_bc.col(2) = Eigen::Vector3d::UnitX();
    r_bc = Eigen::AngleAxisd(-0.35, Eigen::Vector3d::UnitY()).toRotationMatrix() * r_bc;
    Eigen::Matrix3d k;
    k << 600.0, 0.0, 330.0, 0.0, 610.0, 235.0, 0.0, 0.0, 1.0;

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0.0, width);
    std::uniform_real_distribution<double> v(0.55 * height, height);
    std::uniform_real_distribution<double> depth(min_depth, max_depth);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<Features> pairs;
    const double frame = 1.0 / 30.0;

    for (int n = 0; n < 150; ++n)
    {
        const double t1 = 2.0 + n * frame;
        const double t2 = t1 + frame;
        const Eigen::Quaterniond q1 = attitude(t1 + true_delay_s);
        const Eigen::Matrix3d r_rel = (q1.conjugate() * attitude(t2 + true_delay_s)).toRotationMatrix();
        const Eigen::Matrix3d r_c1c2 = r_bc.transpose() * r_rel * r_bc;
        const Eigen::Vector3d translation = r_bc.transpose() * (q1.conjugate() * (position(t2 + true_delay_s) - position(t1 + true_delay_s)));
        std::vector<cv::Point2f> p1;
        std::vector<cv::Point2f> p2;

        for (int i = 0; i < 150; ++i)
        {
            const Eigen::Vector3d pixel(u(rng), v(rng), 1.0);
            const Eigen::Vector3d point = depth(rng) * (k.inverse() * pixel);
            const Eigen::Vector3d moved = k * (r_c1c2.transpose() * (point - translation));

            if (moved.z() <= 0.0)
            {
                continue;
            }

            const double u2 = moved.x() / moved.z() + noise(rng);
            const double v2 = moved.y() / moved.z() + noise(rng);
            const double u1 = pixel.x() + noise(rng);
            const double v1 = pixel.y() + noise(rng);

            if (u2 < 0.0 || u2 >= width || v2 < 0.0 || v2 >= height)
            {
                continue;
            }

            p1.emplace_back((u1 - width / 2) / (width / 2), (v1 - height / 2) / (height / 2));
            p2.emplace_back((u2 - width / 2) / (width / 2), (v2 - height / 2) / (height / 2));
        }

        std::vector<uchar> inliers;
        const cv::Mat affine = cv::estimateAffine2D(p1, p2, inliers, cv::RANSAC, 0.005);

        if (!affine.empty())
        {
            Features pair;
            pair.timestamp = {std::llround(t1 * 1e9), std::llround(t2 * 1e9)};
            pair.A << affine.at<double>(0, 0), affine.at<double>(0, 1), affine.at<double>(1, 0), affine.at<double>(1, 1);
            pair.b << affine.at<double>(0, 2), affine.at<double>(1, 2);
            pairs.push_back(pair);
        }
    }

    const FlowMethod method("delay_estimator_flow");
    GridSearch<int64_t, double> search;
    search.setParameters({-500'000'000, 0, {50'000'000, 10'000'000, 2'000'000}});
    const SampleSynchronizer synchronizer(std::move(pairs), {}, std::move(attitude_samples), std::move(position_samples));

    return search.run(
        [&](int64_t delay_ns) -> Evaluation<double>
        {
            const std::vector<FlowSample> samples = synchronizer.get_flow(delay_ns);
            return {samples.size(), method.fit(samples)};
        },
        [&](std::size_t samples) { return method.degrees_of_freedom(samples); });
}

template <typename AttitudeFunction>
delay_estimator::SearchResult<int64_t, double> simulate_horizon(AttitudeFunction attitude, double true_delay_s)
{
    using namespace delay_estimator;

    std::vector<Attitude> attitude_samples;

    for (int i = 0; i <= 2600; ++i)
    {
        const double t = i * 0.005;
        attitude_samples.push_back({std::llround(t * 1e9), attitude(t)});
    }

    Eigen::Matrix3d r_bc;
    r_bc.col(0) = Eigen::Vector3d::UnitY();
    r_bc.col(1) = Eigen::Vector3d::UnitZ();
    r_bc.col(2) = Eigen::Vector3d::UnitX();
    r_bc = Eigen::AngleAxisd(-0.35, Eigen::Vector3d::UnitY()).toRotationMatrix() * r_bc;
    const double focal = 600.0;
    const double half_diagonal = std::hypot(width, height) / 2.0;

    std::mt19937 rng(5);
    std::normal_distribution<double> theta_noise(0.0, 0.2 * M_PI / 180.0);
    std::normal_distribution<double> rho_noise(0.0, 0.002);
    std::vector<Horizon> horizons;

    for (int n = 0; n < 300; ++n)
    {
        const double t = 1.0 + n / 30.0;
        const Eigen::Quaterniond q = attitude(t + true_delay_s);
        const Eigen::Matrix3d r_wc = q.toRotationMatrix() * r_bc;
        Eigen::Vector3d forward = r_wc.col(2);
        forward.z() = 0.0;
        forward.normalize();
        std::array<Eigen::Vector2d, 2> points;

        for (int i = 0; i < 2; ++i)
        {
            const Eigen::Vector3d direction = Eigen::AngleAxisd(i == 0 ? -0.2 : 0.2, Eigen::Vector3d::UnitZ()) * forward;
            const Eigen::Vector3d camera = r_wc.transpose() * direction;
            const double u = focal * camera.x() / camera.z();
            const double v = focal * camera.y() / camera.z();
            points[i] = Eigen::Vector2d(u, -v) / half_diagonal;
        }

        const Eigen::Vector2d d = (points[1] - points[0]).normalized();
        Eigen::Vector2d normal(-d.y(), d.x());

        if (normal.y() < 0.0)
        {
            normal = -normal;
        }

        const double theta = std::atan2(-normal.x(), normal.y()) + theta_noise(rng);
        const double rho = normal.dot(points[0]) + rho_noise(rng);
        horizons.push_back({std::llround(t * 1e9), static_cast<float>(theta), static_cast<float>(rho)});
    }

    const HorizonMethod method("delay_estimator_horizon_method");
    GridSearch<int64_t, double> search;
    search.setParameters({-500'000'000, 0, {50'000'000, 10'000'000, 2'000'000}});
    const SampleSynchronizer synchronizer({}, std::move(horizons), std::move(attitude_samples), {});

    return search.run(
        [&](int64_t delay_ns) -> Evaluation<double>
        {
            const std::vector<HorizonSample> samples = synchronizer.get_horizon(delay_ns);
            return {samples.size(), method.fit(samples)};
        },
        [&](std::size_t samples) { return method.degrees_of_freedom(samples); });
}

}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    bool ok = true;

    const auto excitedResult = simulate(excited, steady, -0.137);
    std::printf("steady flight: %.1f ms\n", static_cast<double>(excitedResult.value) * 1e-6);
    ok &= check(!std::isnan(excitedResult.cost), "excited window is accepted");
    ok &= check(std::abs(static_cast<double>(excitedResult.value) * 1e-6 + 137.0) <= 4.0, "delay is recovered within 4 ms");

    const auto manoeuvringResult = simulate(excited, manoeuvring, -0.137);
    std::printf("manoeuvring flight: %.1f ms\n", static_cast<double>(manoeuvringResult.value) * 1e-6);
    ok &= check(!std::isnan(manoeuvringResult.cost), "manoeuvring window is accepted");
    ok &= check(std::abs(static_cast<double>(manoeuvringResult.value) * 1e-6 + 137.0) <= 4.0, "delay is recovered within 4 ms while manoeuvring");

    const auto lowResult = simulate(excited, manoeuvring, -0.137, 15.0, 60.0);
    std::printf("low manoeuvring flight: %.1f ms\n", static_cast<double>(lowResult.value) * 1e-6);
    ok &= check(!std::isnan(lowResult.cost), "low manoeuvring window is accepted");
    ok &= check(std::abs(static_cast<double>(lowResult.value) * 1e-6 + 137.0) <= 4.0, "delay is recovered within 4 ms in low manoeuvring flight");

    const auto stillResult = simulate(still, steady, -0.137);
    ok &= check(std::isnan(stillResult.cost), "window without rotation is rejected");

    const cv::Matx23d affine(1.01, -0.02, 0.03, 0.04, 0.98, -0.05);
    ifc_msgs::msg::Features message;
    message.image_timestamp[0].nanosec = 1;
    message.image_timestamp[1].nanosec = 2;

    for (int i = 0; i < 6; ++i)
    {
        message.affine[i] = static_cast<float>(affine.val[i]);
    }

    const auto decoded = delay_estimator::FeaturesUtils().convert(message);
    const cv::Vec2d expected = affine * cv::Vec3d(0.3, -0.7, 1.0);
    const Eigen::Vector2d mapped = decoded->A * Eigen::Vector2d(0.3, -0.7) + decoded->b;
    ok &= check(decoded && std::abs(mapped.x() - expected[0]) < 1e-6 && std::abs(mapped.y() - expected[1]) < 1e-6,
        "affine message decodes to x2 = A * x1 + b");

    {
        using delay_estimator::Attitude;
        std::vector<Attitude> attitude;

        for (int i = 0; i <= 100; ++i)
        {
            attitude.push_back({i * 10'000'000LL, excited(i * 0.01)});
        }

        const delay_estimator::SampleSynchronizer synchronizer(
            {}, {{100'000'000, 0.1f, 0.0f}, {500'000'000, 0.2f, 0.0f}, {950'000'000, 0.3f, 0.0f}}, std::move(attitude), {});
        const auto horizon = synchronizer.get_horizon(-200'000'000);
        ok &= check(horizon.size() == 2 && horizon[0].horizon.timestamp == 500'000'000 && horizon[1].horizon.timestamp == 950'000'000,
            "horizon samples without shifted attitude are dropped");
        ok &= check(horizon.size() == 2 && horizon[0].attitude.timestamp == 500'000'000
            && horizon[0].attitude.q.angularDistance(excited(0.3)) < 1e-9,
            "horizon attitude is interpolated at the shifted time and stamped with the horizon time");
    }

    const auto horizonResult = simulate_horizon(excited, -0.137);
    std::fprintf(stderr, "horizon: %.1f ms\n", static_cast<double>(horizonResult.value) * 1e-6);
    ok &= check(!std::isnan(horizonResult.cost), "horizon window is accepted");
    ok &= check(std::abs(static_cast<double>(horizonResult.value) * 1e-6 + 137.0) <= 2.0, "delay is recovered from the horizon within 2 ms");

    const auto stillHorizon = simulate_horizon(still, -0.137);
    ok &= check(std::isnan(stillHorizon.cost), "horizon window without rotation is rejected");

    {
        delay_estimator::SearchResult<int64_t, double> parabola;
        parabola.value = -50;
        parabola.cost = 0.5 + 0.04 * 4.0;

        for (int64_t d = -55; d <= -45; ++d)
        {
            const double x = static_cast<double>(d + 52);
            parabola.evaluations.emplace_back(d, 0.5 + 0.04 * x * x);
        }

        parabola.evaluations.insert(parabola.evaluations.begin(), {-75, 10.0});
        parabola.evaluations.emplace_back(-25, 10.0);
        parabola.degrees_of_freedom = 2;
        const auto curvature = delay_estimator::estimate(parabola);
        ok &= check(curvature.valid && curvature.delay == -52 && std::abs(curvature.sd - 2.5) < 1e-9,
            "delay is the parabola vertex over the finest-step window and sd = sqrt(minimum / (dof * curvature))");

        parabola.cost = std::numeric_limits<double>::quiet_NaN();
        ok &= check(!delay_estimator::estimate(parabola).valid, "invalid minimum gives no estimate");

        const auto fused = delay_estimator::fuse({{true, -40, 4.0}, {true, -60, 3.0}, {false, 0, 0.0}});
        ok &= check(fused.valid && fused.delay == std::llround((-40.0 / 16.0 - 60.0 / 9.0) / (1.0 / 16.0 + 1.0 / 9.0)) && std::abs(fused.sd - 2.4) < 1e-9,
            "estimates are fused by inverse variance");
        ok &= check(!delay_estimator::fuse({{}, {}}).valid, "fusing invalid estimates is invalid");
    }

    {
        delay_estimator::GridSearch<int64_t, int> search;
        search.setParameters({-500, 0, {20, 5, 1}, 8});
        const auto result = search.run(
            [](int64_t value) { return delay_estimator::Evaluation<int> {static_cast<std::size_t>(-value), static_cast<int>(std::abs(value + 137))}; },
            [](std::size_t samples) { return samples; });
        bool neighbours = true;

        for (int64_t value = -145; value <= -129; ++value)
        {
            neighbours &= std::any_of(result.evaluations.begin(), result.evaluations.end(),
                [value](const std::pair<int64_t, int>& evaluation) { return evaluation.first == value; });
        }

        ok &= check(result.value == -137 && result.cost == 0 && neighbours && std::is_sorted(result.evaluations.begin(), result.evaluations.end()),
            "neighbours of the best value are evaluated on the finest step");
        ok &= check(result.degrees_of_freedom == 137, "degrees of freedom are computed from the data of the best value");
    }

    const auto throwsFor = [](const std::vector<int64_t>& steps)
    {
        try
        {
            delay_estimator::GridSearch<int64_t, double> search;
            search.setParameters({-500, 0, steps});
        }
        catch (const std::invalid_argument&)
        {
            return true;
        }

        return false;
    };

    ok &= check(throwsFor({50, 20}), "steps that do not divide the previous step are rejected");
    ok &= check(throwsFor({30}), "range not divisible by the first step is rejected");

    rclcpp::shutdown();
    return ok ? 0 : 1;
}
