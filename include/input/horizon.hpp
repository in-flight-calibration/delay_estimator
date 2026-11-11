#pragma once

#include <ifc_tools/buffered_receiver.hpp>
#include <ifc_tools/interpolator.hpp>

#include <ifc_msgs/msg/horizon.hpp>

#include <rclcpp/time.hpp>

#include <cmath>
#include <cstdint>
#include <optional>

namespace delay_estimator
{

struct Horizon
{
    int64_t timestamp = 0;
    float theta = 0.0f;
    float rho = 0.0f;
};

using HorizonReceiver = ifc_tools::BufferedReceiver<ifc_msgs::msg::Horizon, Horizon>;

class HorizonUtils
    : public ifc_tools::MessageConverter<ifc_msgs::msg::Horizon, Horizon>
{
public:
    std::optional<Horizon> convert(const ifc_msgs::msg::Horizon& message) const override
    {
        if (message.confidence <= 0.0f || !std::isfinite(message.theta) || !std::isfinite(message.rho))
        {
            return std::nullopt;
        }

        return Horizon{
            rclcpp::Time(message.image_timestamp).nanoseconds(),
            message.theta,
            message.rho};
    }
};

}
