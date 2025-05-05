#include "GameStateDevice.cuh"
#include <iostream>

int main() {
    std::cout << "Starting test..." << std::endl;
    
    try {
        GameStateDevice state(1024, 2, 2);
        std::cout << "GameStateDevice created successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
    
    std::cout << "Test completed successfully" << std::endl;
    return 0;
}