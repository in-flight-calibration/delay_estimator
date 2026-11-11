#pragma once

#include <alignment/data_shifter.hpp>
#include <input/attitude.hpp>
#include <input/features.hpp>
#include <input/horizon.hpp>
#include <input/local_position.hpp>

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
    Features features;
    std::array<Attitude, 2> attitude;
    std::array<LocalPosition, 2> position;
};

struct HorizonSample
{
    Horizon horizon;
    Attitude attitude;
};

class SampleSynchronizer
{
public:
    SampleSynchronizer(
        std::vector<Features> features,
        std::vector<Horizon> horizon,
        std::vector<Attitude> attitude,
        std::vector<LocalPosition> position)
        : _features(std::move(features)),
          _horizon(std::move(horizon)),
          _attitude_shifter(std::move(attitude), std::make_shared<AttitudeUtils>()),
          _position_shifter(std::move(position), std::make_shared<LocalPositionUtils>())
    {
    }

    std::vector<FlowSample> get_flow(int64_t delay_ns) const
    {
        std::vector<int64_t> timestamps;
        timestamps.reserve(2 * _features.size());

        for (const Features& features : _features)
        {
            timestamps.insert(timestamps.end(), features.timestamp.begin(), features.timestamp.end());
        }

        timestamps.erase(std::unique(timestamps.begin(), timestamps.end()), timestamps.end());

        const std::vector<std::optional<Attitude>> attitude = _attitude_shifter.get(timestamps, delay_ns);
        const std::vector<std::optional<LocalPosition>> position = _position_shifter.get(timestamps, delay_ns);
        std::vector<FlowSample> samples;
        samples.reserve(_features.size());

        for (const Features& features : _features)
        {
            const auto first = std::lower_bound(timestamps.begin(), timestamps.end(), features.timestamp[0]);
            const auto i = static_cast<std::size_t>(std::distance(timestamps.begin(), first));
            const auto j = static_cast<std::size_t>(std::distance(timestamps.begin(), std::lower_bound(first, timestamps.end(), features.timestamp[1])));

            if (!attitude[i] || !attitude[j] || !position[i] || !position[j])
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

        for (const Horizon& horizon : _horizon)
        {
            timestamps.push_back(horizon.timestamp);
        }

        const std::vector<std::optional<Attitude>> attitude = _attitude_shifter.get(timestamps, delay_ns);
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
    std::vector<Features> _features;
    std::vector<Horizon> _horizon;
    DataShifter<Attitude> _attitude_shifter;
    DataShifter<LocalPosition> _position_shifter;
};

}
