#pragma once

#include <alignment/sample_synchronizer.hpp>

#include <rclcpp/rclcpp.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace delay_estimator
{

struct HorizonResidual
{
    HorizonResidual(double theta, double rho, const Eigen::Vector3d& gravity, double scale)
        : _theta(theta), _rho(rho), _gravity(gravity), _scale(scale)
    {
    }

    template <typename T>
    bool operator()(const T* mount, const T* log_focal, T* residual) const
    {
        const T gravity[3] = {T(_gravity.x()), T(_gravity.y()), T(_gravity.z())};
        T camera[3];
        ceres::AngleAxisRotatePoint(mount, gravity, camera);
        const T line[3] = {-camera[0], camera[1], -ceres::exp(log_focal[0]) * camera[2]};
        const T normal = ceres::sqrt(line[0] * line[0] + line[1] * line[1]);
        const double m0 = -std::sin(_theta);
        const double m1 = std::cos(_theta);
        T dot = T(m0) * line[0] + T(m1) * line[1];
        T cross = T(m0) * line[1] - T(m1) * line[0];
        T rho = -line[2] / normal;

        if (dot < T(0))
        {
            dot = -dot;
            cross = -cross;
            rho = -rho;
        }

        residual[0] = ceres::atan2(cross, dot) / T(_scale);
        residual[1] = (rho - T(_rho)) / T(_scale);
        return true;
    }

private:
    double _theta;
    double _rho;
    Eigen::Vector3d _gravity;
    double _scale;
};

class HorizonMethod final : public rclcpp::Node
{
public:
    explicit HorizonMethod(const std::string& name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node(name, options)
    {
        _huber_k = declare_parameter<double>("delay.horizon.huber_k", 1.345);
        _max_iterations = declare_parameter<int>("delay.horizon.max_iterations", 50);
        _min_samples = static_cast<std::size_t>(declare_parameter<int>("delay.horizon.min_samples", 60));
        _min_excitation = declare_parameter<double>("delay.horizon.min_excitation_rad", 0.02);
        _max_cost = declare_parameter<double>("delay.horizon.max_cost", 0.5);

        if (_huber_k <= 0.0 || _max_iterations <= 0)
        {
            throw std::invalid_argument("invalid horizon method parameter");
        }
    }

    double fit(const std::vector<HorizonSample>& samples) const
    {
        constexpr double invalid = std::numeric_limits<double>::quiet_NaN();

        if (samples.size() < _min_samples)
        {
            return invalid;
        }

        std::vector<double> theta;
        std::vector<double> rho;
        theta.reserve(samples.size());
        rho.reserve(samples.size());

        for (const HorizonSample& sample : samples)
        {
            theta.push_back(sample.horizon.theta);
            rho.push_back(sample.horizon.rho);
        }

        const double theta_centre = median(theta);
        const double rho_centre = median(rho);
        std::vector<double> deviation;
        deviation.reserve(samples.size());

        for (const double value : theta)
        {
            deviation.push_back(std::abs(wrap(value - theta_centre)));
        }

        const double scale = 1.4826 * median(deviation);

        if (!(scale >= _min_excitation))
        {
            return invalid;
        }

        std::vector<double> spread;
        spread.reserve(samples.size());

        for (std::size_t k = 0; k < samples.size(); ++k)
        {
            spread.push_back(std::hypot(wrap(theta[k] - theta_centre), rho[k] - rho_centre) / scale);
        }

        const double delta = _huber_k * 1.4826 * median(spread);

        if (!(delta > 0.0))
        {
            return invalid;
        }

        double baseline = 0.0;

        for (const double e : spread)
        {
            baseline += huber(delta, e * e);
        }

        const double value = solve(samples, scale, delta, rho_centre) / baseline;
        return value <= _max_cost ? value : invalid;
    }

    double max_cost() const
    {
        return _max_cost;
    }

    std::size_t degrees_of_freedom(std::size_t samples) const
    {
        return samples > 2 ? 2 * samples - 4 : 0;
    }

private:
    double _huber_k = 0.0;
    int _max_iterations = 0;
    std::size_t _min_samples = 0;
    double _min_excitation = 0.0;
    double _max_cost = 0.0;

    static double median(std::vector<double> values)
    {
        const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    }

    static double wrap(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    static double huber(double delta, double squared_norm)
    {
        double rho[3];
        ceres::HuberLoss(delta).Evaluate(squared_norm, rho);
        return 0.5 * rho[0];
    }

    static Eigen::Vector3d gravity(const Attitude& attitude)
    {
        static const Eigen::Matrix3d nominal = (Eigen::Matrix3d() << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0).finished();
        return nominal * (attitude.q.conjugate() * Eigen::Vector3d::UnitZ());
    }

    double solve(const std::vector<HorizonSample>& samples, double scale, double delta, double rho_centre) const
    {
        constexpr double initial_focal = 1.0;
        std::array<double, 3> mount = {std::atan(rho_centre / initial_focal), 0.0, 0.0};
        double log_focal = std::log(initial_focal);
        ceres::Problem::Options problem_options;
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem problem(problem_options);
        ceres::HuberLoss loss(delta);

        for (const HorizonSample& sample : samples)
        {
            auto* cost = new ceres::AutoDiffCostFunction<HorizonResidual, 2, 3, 1>(
                new HorizonResidual(sample.horizon.theta, sample.horizon.rho, gravity(sample.attitude), scale));
            problem.AddResidualBlock(cost, &loss, mount.data(), &log_focal);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = _max_iterations;
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        return summary.final_cost;
    }
};

}
