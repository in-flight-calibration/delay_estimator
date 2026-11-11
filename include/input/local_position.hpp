#pragma once

#include <ifc_tools/buffered_receiver.hpp>
#include <ifc_tools/interpolator.hpp>

#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <optional>

namespace delay_estimator
{

struct LocalPosition
{
    int64_t timestamp = 0;
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

using LocalPositionReceiver = ifc_tools::BufferedReceiver<px4_msgs::msg::VehicleLocalPosition, LocalPosition>;

class LocalPositionUtils
    : public ifc_tools::MessageConverter<px4_msgs::msg::VehicleLocalPosition, LocalPosition>,
      public ifc_tools::InterpolationMethod<LocalPosition>
{
public:
    std::optional<LocalPosition> convert(const px4_msgs::msg::VehicleLocalPosition& message) const override
    {
        const Eigen::Vector3d p(message.x, message.y, message.z);

        if (!message.xy_valid || !message.z_valid || !p.allFinite())
        {
            return std::nullopt;
        }

        return LocalPosition{static_cast<int64_t>(message.timestamp_sample) * 1000, p};
    }

    LocalPosition interpolate(const LocalPosition& a, const LocalPosition& b, float alpha) const override
    {
        return {a.timestamp, a.p + static_cast<double>(alpha) * (b.p - a.p)};
    }
};

}
