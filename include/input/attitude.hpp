#pragma once

#include <ifc_tools/buffered_receiver.hpp>
#include <ifc_tools/interpolator.hpp>

#include <px4_msgs/msg/vehicle_attitude.hpp>

#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace delay_estimator
{

struct Attitude
{
    int64_t timestamp = 0;
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

using AttitudeReceiver = ifc_tools::BufferedReceiver<px4_msgs::msg::VehicleAttitude, Attitude>;

class AttitudeUtils
    : public ifc_tools::MessageConverter<px4_msgs::msg::VehicleAttitude, Attitude>,
      public ifc_tools::InterpolationMethod<Attitude>
{
public:
    std::optional<Attitude> convert(const px4_msgs::msg::VehicleAttitude& message) const override
    {
        const Eigen::Quaterniond q(message.q[0], message.q[1], message.q[2], message.q[3]);
        const double norm = q.norm();

        if (!std::isfinite(norm) || norm <= 0.0)
        {
            return std::nullopt;
        }

        return Attitude{static_cast<int64_t>(message.timestamp_sample) * 1000, q.normalized()};
    }

    Attitude interpolate(const Attitude& a, const Attitude& b, float alpha) const override
    {
        Eigen::Quaterniond delta = a.q.conjugate() * b.q;

        if (delta.w() < 0.0)
        {
            delta.coeffs() = -delta.coeffs();
        }

        const Eigen::AngleAxisd rotation(delta);
        return {a.timestamp, (a.q * Eigen::Quaterniond(Eigen::AngleAxisd(alpha * rotation.angle(), rotation.axis()))).normalized()};
    }
};

}
