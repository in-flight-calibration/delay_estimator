#pragma once

#include <ifc_tools/processing/interpolator.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace delay_estimator
{

template <typename T>
class DataShifter
{
public:
    explicit DataShifter(
        std::vector<T> samples,
        std::shared_ptr<const ifc_tools::InterpolationMethod<T>> method = std::make_shared<ifc_tools::Nearest<T>>())
        : _samples(std::move(samples)), _method(std::move(method))
    {
    }

    std::vector<std::optional<T>> get(const std::vector<int64_t>& timestamps, int64_t delay_ns) const
    {
        std::vector<int64_t> shifted(timestamps);

        for (int64_t& t : shifted)
        {
            t += delay_ns;
        }

        std::vector<std::optional<T>> values = ifc_tools::Interpolator(_samples).interpolate(shifted, *_method);

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (values[i])
            {
                values[i]->timestamp = timestamps[i];
            }
        }

        return values;
    }

private:
    std::vector<T> _samples;
    std::shared_ptr<const ifc_tools::InterpolationMethod<T>> _method;
};

}
