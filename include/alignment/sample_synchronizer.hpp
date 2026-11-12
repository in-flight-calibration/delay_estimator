#pragma once

#include <alignment/data_shifter.hpp>
#include <ifc_tools/msg/features.hpp>
#include <ifc_tools/msg/horizon.hpp>
#include <ifc_tools/msg/vehicle_attitude.hpp>
#include <ifc_tools/msg/vehicle_local_position.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace delay_estimator
{

struct FlowSample
{
    ifc_tools::Features features;
    std::array<ifc_tools::VehicleAttitude, 2> attitude;
    std::array<ifc_tools::VehicleLocalPosition, 2> local_position;
};

struct HorizonSample
{
    ifc_tools::Horizon horizon;
    ifc_tools::VehicleAttitude attitude;
};

class SampleSynchronizer
{
public:
    SampleSynchronizer(
        std::vector<ifc_tools::Features> features,
        std::vector<ifc_tools::Horizon> horizon,
        std::vector<ifc_tools::VehicleAttitude> attitude,
        std::vector<ifc_tools::VehicleLocalPosition> position)
        : _features(std::move(features)),
          _horizon(std::move(horizon)),
          _attitude_shifter(std::move(attitude), std::make_shared<ifc_tools::VehicleAttitudeUtils>()),
          _position_shifter(std::move(position), std::make_shared<ifc_tools::VehicleLocalPositionUtils>())
    {
    }

    std::vector<FlowSample> get_flow(int64_t delay_ns) const
    {
        std::vector<int64_t> timestamps;
        timestamps.reserve(2 * _features.size());

        for (const ifc_tools::Features& features : _features)
        {
            timestamps.insert(timestamps.end(), features.timestamp.begin(), features.timestamp.end());
        }

        timestamps.erase(std::unique(timestamps.begin(), timestamps.end()), timestamps.end());

        const std::vector<std::optional<ifc_tools::VehicleAttitude>> attitude = _attitude_shifter.get(timestamps, delay_ns);
        const std::vector<std::optional<ifc_tools::VehicleLocalPosition>> position = _position_shifter.get(timestamps, delay_ns);
        std::vector<FlowSample> samples;
        samples.reserve(_features.size());

        for (const ifc_tools::Features& features : _features)
        {
            const auto first = std::lower_bound(timestamps.begin(), timestamps.end(), features.timestamp[0]);
            const auto i = static_cast<std::size_t>(std::distance(timestamps.begin(), first));
            const auto j = static_cast<std::size_t>(std::distance(timestamps.begin(), std::lower_bound(first, timestamps.end(), features.timestamp[1])));

            if (!attitude[i] || !attitude[j] || !position[i] || !position[j]
                || !position[i]->position.allFinite() || !position[j]->position.allFinite())
            {
                continue;
            }

            samples.push_back({features, {*attitude[i], *attitude[j]}, {*position[i], *position[j]}});
        }

        return samples;
    }

    std::vector<HorizonSample> get_horizon(int64_t delay_ns) const
    {
        std::vector<int64_t> timestamps;
        timestamps.reserve(_horizon.size());

        for (const ifc_tools::Horizon& horizon : _horizon)
        {
            timestamps.push_back(horizon.timestamp);
        }

        const std::vector<std::optional<ifc_tools::VehicleAttitude>> attitude = _attitude_shifter.get(timestamps, delay_ns);
        std::vector<HorizonSample> samples;
        samples.reserve(_horizon.size());

        for (std::size_t k = 0; k < _horizon.size(); ++k)
        {
            if (attitude[k])
            {
                samples.push_back({_horizon[k], *attitude[k]});
            }
        }

        return samples;
    }

private:
    std::vector<ifc_tools::Features> _features;
    std::vector<ifc_tools::Horizon> _horizon;
    DataShifter<ifc_tools::VehicleAttitude> _attitude_shifter;
    DataShifter<ifc_tools::VehicleLocalPosition> _position_shifter;
};

}
