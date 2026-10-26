#include "test.hpp"

#include <exception>
#include <iostream>

int main() {
    std::size_t passed = 0;
    std::size_t failed = 0;

    for (const dnp::test::TestCase& test : dnp::test::registry()) {
        try {
            test.function();
            passed++;
        } catch (const std::exception& exception) {
            failed++;
            std::cerr << "[FAIL] " << test.name << ": " << exception.what()
                      << '\n';
        } catch (...) {
            failed++;
            std::cerr << "[FAIL] " << test.name
                      << ": unknown non-standard exception\n";
        }
    }

    std::cout << "DNP tests: " << passed << " passed, " << failed
              << " failed\n";

    return failed == 0 ? 0 : 1;
}
