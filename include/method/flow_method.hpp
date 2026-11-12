#pragma once

#include <alignment/sample_synchronizer.hpp>

#include <rclcpp/rclcpp.hpp>

#include <ceres/ceres.h>
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

constexpr int flow_regressor_size = 6;

template <int S>
using FlowSummary = Eigen::Matrix<double, S, 1>;
using FlowRegressor = Eigen::Matrix<double, flow_regressor_size, 1>;

template <int S>
struct FlowResidual
{
    FlowResidual(const FlowSummary<S>& s, const FlowRegressor& phi)
        : _s(s), _phi(phi)
    {
    }

    template <typename T>
    bool operator()(const T* theta, T* residual) const
    {
        for (int i = 0; i < S; ++i)
        {
            T prediction = T(0);

            for (int j = 0; j < flow_regressor_size; ++j)
            {
                prediction += theta[flow_regressor_size * i + j] * T(_phi[j]);
            }

            residual[i] = T(_s[i]) - prediction;
        }

        return true;
    }

private:
    FlowSummary<S> _s;
    FlowRegressor _phi;
};

template <int N>
struct RidgeResidual
{
    explicit RidgeResidual(const std::array<double, N>& weights)
        : _weights(weights)
    {
    }

    template <typename T>
    bool operator()(const T* theta, T* residual) const
    {
        for (int i = 0; i < N; ++i)
        {
            residual[i] = T(_weights[i]) * theta[i];
        }

        return true;
    }

private:
    std::array<double, N> _weights;
};

class FlowMethod final : public rclcpp::Node
{
public:
    explicit FlowMethod(const std::string& name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node(name, options)
    {
        _huber_k = declare_parameter<double>("delay_estimator.flow.huber_k", 1.345);
        _max_iterations = declare_parameter<int>("delay_estimator.flow.max_iterations", 20);
        _ridge = declare_parameter<double>("delay_estimator.flow.ridge", 1e-3);
        _min_samples = static_cast<std::size_t>(declare_parameter<int>("delay_estimator.flow.min_samples", 60));
        _min_excitation = declare_parameter<double>("delay_estimator.flow.min_excitation_rad_s", 0.05);
        _max_cost = declare_parameter<double>("delay_estimator.flow.max_cost", 0.5);
        _expansion = declare_parameter<bool>("delay_estimator.flow.expansion", false);

        if (_huber_k <= 0.0 || _max_iterations <= 0 || _ridge < 0.0)
        {
            throw std::invalid_argument("invalid flow method parameter");
        }
    }

    double fit(const std::vector<FlowSample>& samples) const
    {
        return _expansion ? fit<6>(samples) : fit<4>(samples);
    }

    double max_cost() const
    {
        return _max_cost;
    }

    std::size_t degrees_of_freedom(std::size_t samples) const
    {
        const std::size_t outputs = _expansion ? 6 : 4;
        return samples > flow_regressor_size ? outputs * (samples - flow_regressor_size) : 0;
    }

private:
    template <int S>
    struct Prepared
    {
        bool ready = false;
        double delta = 0.0;
        std::vector<FlowSummary<S>> white;
        std::vector<double> baseline;
    };

    double _huber_k = 0.0;
    int _max_iterations = 0;
    double _ridge = 0.0;
    std::size_t _min_samples = 0;
    double _min_excitation = 0.0;
    double _max_cost = 0.0;
    bool _expansion = false;

    static double median(std::vector<double> values)
    {
        const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    }

    static double huber(double delta, double squared_norm)
    {
        double rho[3];
        ceres::HuberLoss(delta).Evaluate(squared_norm, rho);
        return 0.5 * rho[0];
    }

    static double duration(const FlowSample& sample)
    {
        return 1e-9 * static_cast<double>(sample.features.timestamp[1] - sample.features.timestamp[0]);
    }

    template <int S>
    static FlowSummary<S> summary(const FlowSample& sample)
    {
        const Eigen::Matrix2d& A = sample.features.A;
        const Eigen::Vector2d& b = sample.features.b;
        FlowSummary<S> s;

        if constexpr (S == 6)
        {
            s << b.x(), b.y(), A(0, 1), A(1, 0), A(0, 0) - 1.0, A(1, 1) - 1.0;
        }
        else
        {
            s << b.x(), b.y(), A(0, 1), A(1, 0);
        }

        return s;
    }

    template <int S>
    double fit(const std::vector<FlowSample>& samples) const
    {
        constexpr double invalid = std::numeric_limits<double>::quiet_NaN();

        if (samples.size() < _min_samples || samples.size() <= flow_regressor_size)
        {
            return invalid;
        }

        const Prepared<S> prepared = prepare<S>(samples);

        if (!prepared.ready)
        {
            return invalid;
        }

        std::vector<FlowRegressor> regressors;
        regressors.reserve(samples.size());

        for (const FlowSample& sample : samples)
        {
            regressors.push_back(regressor(sample));
        }

        if (!excited(samples, regressors))
        {
            return invalid;
        }

        const double value = solve<S>(regressors, prepared);
        return value <= _max_cost ? value : invalid;
    }

    static FlowRegressor regressor(const FlowSample& sample)
    {
        const Eigen::Quaterniond& q1 = sample.attitude[0].q;
        const Eigen::AngleAxisd rotation(q1.conjugate() * sample.attitude[1].q);
        FlowRegressor phi;
        phi << rotation.angle() * rotation.axis(), q1.conjugate() * (sample.position[1].p - sample.position[0].p);
        return phi;
    }

    bool excited(const std::vector<FlowSample>& samples, const std::vector<FlowRegressor>& regressors) const
    {
        Eigen::Matrix3d excitation = Eigen::Matrix3d::Zero();

        for (std::size_t k = 0; k < samples.size(); ++k)
        {
            const Eigen::Vector3d rate = regressors[k].head<3>() / duration(samples[k]);
            excitation += rate * rate.transpose();
        }

        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(excitation / static_cast<double>(samples.size()));
        return std::sqrt(svd.singularValues()[1]) >= _min_excitation;
    }

    template <int S>
    double solve(const std::vector<FlowRegressor>& regressors, const Prepared<S>& prepared) const
    {
        constexpr int parameters = S * flow_regressor_size;
        std::array<double, parameters> theta{};
        std::array<double, flow_regressor_size> column_energy{};
        ceres::Problem::Options problem_options;
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem problem(problem_options);
        ceres::HuberLoss loss(prepared.delta);
        std::vector<ceres::ResidualBlockId> data_blocks;
        double baseline = 0.0;

        for (std::size_t k = 0; k < regressors.size(); ++k)
        {
            auto* cost = new ceres::AutoDiffCostFunction<FlowResidual<S>, S, parameters>(new FlowResidual<S>(prepared.white[k], regressors[k]));
            data_blocks.push_back(problem.AddResidualBlock(cost, &loss, theta.data()));
            baseline += prepared.baseline[k];

            for (int j = 0; j < flow_regressor_size; ++j)
            {
                column_energy[j] += regressors[k][j] * regressors[k][j];
            }
        }

        if (baseline <= 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::array<double, parameters> ridge_weights{};

        for (int i = 0; i < S; ++i)
        {
            for (int j = 0; j < flow_regressor_size; ++j)
            {
                ridge_weights[flow_regressor_size * i + j] = std::sqrt(_ridge * column_energy[j]);
            }
        }

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RidgeResidual<parameters>, parameters, parameters>(new RidgeResidual<parameters>(ridge_weights)),
            nullptr,
            theta.data());

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = _max_iterations;
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        ceres::Problem::EvaluateOptions evaluate_options;
        evaluate_options.residual_blocks = data_blocks;
        double cost = 0.0;
        problem.Evaluate(evaluate_options, &cost, nullptr, nullptr, nullptr);
        return cost / baseline;
    }

    template <int S>
    Prepared<S> prepare(const std::vector<FlowSample>& samples) const
    {
        Prepared<S> prepared;
        std::vector<FlowSummary<S>> summaries;
        summaries.reserve(samples.size());

        for (const FlowSample& sample : samples)
        {
            summaries.push_back(summary<S>(sample));
        }

        FlowSummary<S> centre;
        FlowSummary<S> scale;

        for (int c = 0; c < S; ++c)
        {
            std::vector<double> values;
            values.reserve(summaries.size());

            for (const FlowSummary<S>& s : summaries)
            {
                values.push_back(s[c]);
            }

            centre[c] = median(values);

            for (double& value : values)
            {
                value = std::abs(value - centre[c]);
            }

            scale[c] = 1.4826 * median(values);
        }

        if (!(scale.minCoeff() > 0.0))
        {
            return prepared;
        }

        std::vector<double> spread;
        spread.reserve(summaries.size());

        for (const FlowSummary<S>& s : summaries)
        {
            prepared.white.push_back(s.cwiseQuotient(scale));
            spread.push_back((s - centre).cwiseQuotient(scale).norm());
        }

        prepared.delta = _huber_k * 1.4826 * median(spread);

        if (!(prepared.delta > 0.0))
        {
            return prepared;
        }

        for (const double e : spread)
        {
            prepared.baseline.push_back(huber(prepared.delta, e * e));
        }

        prepared.ready = true;
        return prepared;
    }
};

}
