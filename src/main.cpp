#include <delay_estimator.hpp>

#include <memory>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<delay_estimator::DelayEstimator>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    for (const auto& child : node->children())
    {
        executor.add_node(child);
    }

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
