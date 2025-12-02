#include <chrono>
#include <iostream>

#include "RLEnvironment.hpp"

int main()
{
    using clock  = std::chrono::steady_clock;
    using second = std::chrono::duration<double>;

    RLEnvironment env{1024, 4, 4, 123};

    env.reset();

    auto t0 = clock::now();

    for (int i = 0; i < 1; i++)
    {
        env.step();
    }

    auto t1 = clock::now();
    second dt = t1 - t0;
    std::cout << "elapsed: " << dt.count() << " s\n";

    return 0;
}