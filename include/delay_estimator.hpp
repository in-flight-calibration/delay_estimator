#pragma once

#include <alignment/sample_synchronizer.hpp>
#include <debug.hpp>
#include <estimation/fusion.hpp>
#include <estimation/grid_search.hpp>
#include <ifc_tools/buffered_receiver.hpp>
#include <ifc_tools/performance_monitor.hpp>
#include <input/horizon.hpp>
#include <method/flow_method.hpp>
#include <method/horizon_method.hpp>

#include <ifc_msgs/msg/delay.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace delay_estimator
{

class DelayEstimator : public rclcpp::Node
{
public:
    DelayEstimator()
        : Node("delay_estimator"),
          _flow_method(std::make_shared<FlowMethod>("delay_estimator_flow")),
          _horizon_method(std::make_shared<HorizonMethod>("delay_estimator_horizon_method")),
          _performance_monitor(std::make_shared<ifc_tools::PerformanceMonitor>("delay_estimator"))
    {
        SearchParameters<int64_t> search;
        search.min = -declare_parameter<int>("delay.search.max_delay_ms", 500) * 1'000'000LL;
        search.max = 0;
        search.neighbours = static_cast<std::size_t>(declare_parameter<int>("delay.search.neighbours", 5));

        for (const int64_t step : declare_parameter<std::vector<int64_t>>("delay.search.steps_ms", {20, 5, 1}))
        {
            search.steps.push_back(step * 1'000'000LL);
        }

        _search.setParameters(search);

        _features_size = static_cast<std::size_t>(declare_parameter<int>("delay.feature_buffer_size", 300));
        _horizon_size = static_cast<std::size_t>(declare_parameter<int>("delay.horizon_buffer_size", 300));

        _attitude = std::make_shared<AttitudeReceiver>(
            "delay_estimator_attitude",
            std::make_shared<AttitudeUtils>(),
            static_cast<std::size_t>(declare_parameter<int>("delay.attitude_buffer_size", 4000)),
            declare_parameter<std::string>("delay.attitude_topic", "/fmu/out/vehicle_attitude"));

        _position = std::make_shared<LocalPositionReceiver>(
            "delay_estimator_local_position",
            std::make_shared<LocalPositionUtils>(),
            static_cast<std::size_t>(declare_parameter<int>("delay.local_position_buffer_size", 2000)),
            declare_parameter<std::string>("delay.local_position_topic", "/fmu/out/vehicle_local_position"));

        _features = std::make_shared<FeaturesReceiver>(
            "delay_estimator_features",
            std::make_shared<FeaturesUtils>(),
            _features_size,
            declare_parameter<std::string>("delay.features_topic", "/feature_tracker/features"),
            rclcpp::ServicesQoS());

        _horizon = std::make_shared<HorizonReceiver>(
            "delay_estimator_horizon",
            std::make_shared<HorizonUtils>(),
            _horizon_size,
            declare_parameter<std::string>("delay.horizon_topic", "/horizon_detection/horizon"),
            rclcpp::ServicesQoS());

        _publisher = create_publisher<ifc_msgs::msg::Delay>(
            declare_parameter<std::string>("delay.topic", "/delay_estimator/delay"), 
            rclcpp::ServicesQoS());

        _debug = declare_parameter<bool>("delay.debug", false);

        if (declare_parameter<bool>("delay.debug_view", false))
        {
            _plot.emplace(cv::Size(640, 320));
            _plot->setAxis(
                static_cast<double>(search.min) * 1e-6,
                static_cast<double>(search.max) * 1e-6,
                0.0,
                std::max(_flow_method->max_cost(), _horizon_method->max_cost()));
            _plot_thread = std::jthread([this](std::stop_token stop)
            {
                while (!stop.stop_requested())
                {
                    _plot->show();
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
            });
        }

        const std::string performance_topic = declare_parameter<std::string>("delay.performance_topic", "/performance/delay_estimator");

        if (!performance_topic.empty())
        {
            _performance_monitor->setTopic(performance_topic);
        }

        _timer = create_wall_timer(
            std::chrono::milliseconds(declare_parameter<int>("delay.update_ms", 1000)),
            [this]() { update(); });
    }

    std::vector<rclcpp::Node::SharedPtr> children() const
    {
        return {_attitude, _position, _features, _horizon, _flow_method, _horizon_method, _performance_monitor};
    }

private:
    std::size_t _features_size = 0;
    std::size_t _horizon_size = 0;
    bool _debug = false;
    
    std::shared_ptr<AttitudeReceiver> _attitude;
    std::shared_ptr<LocalPositionReceiver> _position;
    std::shared_ptr<FeaturesReceiver> _features;
    std::shared_ptr<HorizonReceiver> _horizon;

    rclcpp::TimerBase::SharedPtr _timer;
    GridSearch<int64_t, double> _search;
    std::shared_ptr<FlowMethod> _flow_method;
    std::shared_ptr<HorizonMethod> _horizon_method;

    rclcpp::Publisher<ifc_msgs::msg::Delay>::SharedPtr _publisher;

    std::optional<SearchPlot> _plot;
    std::jthread _plot_thread;

    std::shared_ptr<ifc_tools::PerformanceMonitor> _performance_monitor;

    void update()
    {
        _performance_monitor->start();
        std::vector<Attitude> attitude = _attitude->snapshot();
        std::vector<LocalPosition> position = _position->snapshot();
        std::vector<Features> features = _features->snapshot();
        std::vector<Horizon> horizon = _horizon->snapshot();
        const bool flow_ready = features.size() >= _features_size && !attitude.empty() && !position.empty();
        const bool horizon_ready = horizon.size() >= _horizon_size && !attitude.empty();
        const std::string flow_waiting = std::format("waiting for data (pairs {}/{}, attitude {}, position {})", features.size(), _features_size, attitude.size(), position.size());
        const std::string horizon_waiting = std::format("waiting for data (horizons {}/{}, attitude {})", horizon.size(), _horizon_size, attitude.size());

        if (!flow_ready && !horizon_ready)
        {
            publish(Estimate{});
            _performance_monitor->stop();

            if (_debug)
            {
                RCLCPP_WARN(get_logger(), "no delay: flow %s; horizon %s", flow_waiting.c_str(), horizon_waiting.c_str());
            }

            return;
        }

        const SampleSynchronizer synchronizer(std::move(features), std::move(horizon), std::move(attitude), std::move(position));
        const int64_t width = static_cast<int64_t>(_search.parameters().neighbours) * _search.parameters().steps.back();
        SearchResult<int64_t, double> flow_result;
        SearchResult<int64_t, double> horizon_result;
        Estimate flow;
        Estimate horizon_estimate;
        if (flow_ready)
        {
            flow_result = _search.run([&](int64_t delay_ns) { return _flow_method->fit(synchronizer.get_flow(delay_ns)); });
            flow = estimate(flow_result, width, _flow_method->degrees_of_freedom(synchronizer.get_flow(flow_result.value).size()));
        }

        if (horizon_ready)
        {
            horizon_result = _search.run([&](int64_t delay_ns) { return _horizon_method->fit(synchronizer.get_horizon(delay_ns)); });
            horizon_estimate = estimate(horizon_result, width, _horizon_method->degrees_of_freedom(synchronizer.get_horizon(horizon_result.value).size()));
        }

        const Estimate fused = fuse({flow, horizon_estimate});
        _performance_monitor->stop();
        publish(fused);

        if (_debug)
        {
            const std::string details = std::format(
                "flow {}; horizon {}",
                flow_ready ? describe(flow_result, flow) : flow_waiting,
                horizon_ready ? describe(horizon_result, horizon_estimate) : horizon_waiting);

            if (fused.valid)
            {
                RCLCPP_INFO(get_logger(), "delay %.1f ± %.1f ms: %s", static_cast<double>(fused.delay) * 1e-6, fused.sd * 1e-6, details.c_str());
            }
            else
            {
                RCLCPP_WARN(get_logger(), "no delay: %s", details.c_str());
            }
        }

        if (_plot)
        {
            plot("flow", flow_result);
            plot("horizon", horizon_result);
            _plot->draw();
        }
    }

    static std::string describe(const SearchResult<int64_t, double>& result, const Estimate& estimate)
    {
        const std::size_t valid = static_cast<std::size_t>(std::count_if(
            result.evaluations.begin(),
            result.evaluations.end(),
            [](const auto& evaluation) { return std::isfinite(evaluation.second); }));
        const std::string search = std::format(
            "best {:.0f} ms, J {:.3g}, {}/{} delays fitted",
            static_cast<double>(result.value) * 1e-6,
            result.cost,
            valid,
            result.evaluations.size());

        if (!estimate.valid)
        {
            return std::format("invalid ({})", search);
        }

        return std::format("{:.1f} ± {:.1f} ms ({})", static_cast<double>(estimate.delay) * 1e-6, estimate.sd * 1e-6, search);
    }

    void plot(const std::string& name, const SearchResult<int64_t, double>& result)
    {
        std::vector<double> x;
        std::vector<double> y;

        for (const auto& [delay, fitness] : result.evaluations)
        {
            x.push_back(static_cast<double>(delay) * 1e-6);
            y.push_back(fitness);
        }

        _plot->add(name, x, y);
    }

    void publish(const Estimate& estimate)
    {
        ifc_msgs::msg::Delay message;
        message.header.stamp = now();
        message.valid = estimate.valid;

        if (estimate.valid)
        {
            message.delay = estimate.delay;
            message.delay_sd = std::llround(estimate.sd);
        }

        _publisher->publish(message);
    }
};

}
