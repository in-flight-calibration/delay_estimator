#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace delay_estimator
{

template <typename T>
struct SearchParameters
{
    T min = 0;
    T max = 0;
    std::vector<T> steps;
    std::size_t neighbours = 0;
};

template <typename T, typename Cost>
struct SearchResult
{
    T value = 0;
    Cost cost{};
    std::vector<std::pair<T, Cost>> evaluations;
};

template <typename T, typename Cost>
class GridSearch
{
    static_assert(std::is_integral_v<T>, "GridSearch requires an integer type");
    static_assert(std::is_arithmetic_v<Cost>, "GridSearch requires a numeric cost type");

public:
    using Evaluate = std::function<Cost(T)>;

    void setParameters(SearchParameters<T> parameters)
    {
        const std::vector<T>& steps = parameters.steps;

        if (steps.empty() || parameters.max <= parameters.min || steps.front() <= 0 || (parameters.max - parameters.min) % steps.front() != 0)
        {
            throw std::invalid_argument("search range must be divisible by the first step");
        }

        for (std::size_t l = 0; l + 1 < steps.size(); ++l)
        {
            if (steps[l + 1] <= 0 || steps[l + 1] >= steps[l] || steps[l] % steps[l + 1] != 0)
            {
                throw std::invalid_argument("each search step must be a smaller divisor of the previous one");
            }
        }

        _parameters = std::move(parameters);
    }

    const SearchParameters<T>& parameters() const
    {
        return _parameters;
    }

    SearchResult<T, Cost> run(const Evaluate& evaluate) const
    {
        if (_parameters.steps.empty())
        {
            throw std::logic_error("search parameters are not set");
        }

        std::map<T, Cost> evaluations;

        const auto cost = [&](T value) -> const Cost&
        {
            auto it = evaluations.find(value);

            if (it == evaluations.end())
            {
                it = evaluations.emplace(value, evaluate(value)).first;
            }

            return it->second;
        };

        T from = _parameters.min;
        T to = _parameters.max;
        T best = from;

        for (const T step : _parameters.steps)
        {
            for (T value = from; value <= to; value += step)
            {
                if (better(cost(value), cost(best)))
                {
                    best = value;
                }
            }

            from = std::max(_parameters.min, static_cast<T>(best - step));
            to = std::min(_parameters.max, static_cast<T>(best + step));
        }

        const T step = _parameters.steps.back();
        const T width = static_cast<T>(_parameters.neighbours) * step;
        T centre;

        do
        {
            centre = best;

            for (T value = std::max(_parameters.min, static_cast<T>(centre - width)); value <= std::min(_parameters.max, static_cast<T>(centre + width)); value += step)
            {
                if (better(cost(value), cost(best)))
                {
                    best = value;
                }
            }
        }
        while (best != centre);

        return {best, evaluations.at(best), {evaluations.begin(), evaluations.end()}};
    }

private:
    SearchParameters<T> _parameters;

    static bool valid(const Cost& cost)
    {
        if constexpr (std::is_floating_point_v<Cost>)
        {
            return !std::isnan(cost);
        }
        else
        {
            return true;
        }
    }

    static bool better(const Cost& candidate, const Cost& best)
    {
        return valid(candidate) && (!valid(best) || candidate < best);
    }
};

}
