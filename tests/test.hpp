#ifndef DNP_TEST_HPP
#define DNP_TEST_HPP

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "dnp/tensor.hpp"

namespace dnp::test {

struct TestCase {
    std::string name;
    std::function<void()> function;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

class Registrar {
  public:
    Registrar(
        std::string p_name,
        std::function<void()> p_function
    ) {
        registry().push_back({std::move(p_name), std::move(p_function)});
    }
};

class Failure : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(
    const std::string& p_message
) {
    throw Failure(p_message);
}

inline void expectTrue(
    bool p_condition,
    const std::string& p_context
) {
    if (!p_condition) fail(p_context);
}

template <typename Actual, typename Expected>
void expectEqual(
    const Actual& p_actual,
    const Expected& p_expected,
    const std::string& p_context
) {
    if (p_actual == p_expected) return;

    std::ostringstream message;
    message << p_context << ": expected " << p_expected << ", got " << p_actual;
    fail(message.str());
}

template <typename Actual, typename Expected, typename Tolerance>
void expectNear(
    const Actual& p_actual,
    const Expected& p_expected,
    const Tolerance& p_tolerance,
    const std::string& p_context
) {
    const long double actual = static_cast<long double>(p_actual);
    const long double expected = static_cast<long double>(p_expected);
    const long double tolerance = static_cast<long double>(p_tolerance);

    if (std::abs(actual - expected) <= tolerance) return;

    std::ostringstream message;
    message << p_context << ": expected " << expected << " +/- " << tolerance
            << ", got " << actual;
    fail(message.str());
}

template <typename Exception, typename Function>
void expectThrows(
    Function&& p_function,
    const std::string& p_context,
    const std::string& p_expected_message = {}
) {
    try {
        std::forward<Function>(p_function)();
    } catch (const Exception& exception) {
        if (!p_expected_message.empty() &&
            std::string(exception.what()).find(p_expected_message) ==
                std::string::npos) {
            fail(p_context + ": exception message did not contain '" +
                 p_expected_message + "': " + exception.what());
        }

        return;
    } catch (const std::exception& exception) {
        fail(p_context + ": unexpected exception type: " + exception.what());
    } catch (...) {
        fail(p_context + ": unexpected non-standard exception");
    }

    fail(p_context + ": expected an exception");
}

template <typename T>
void expectShape(
    const Tensor<T>& p_tensor,
    std::size_t p_rows,
    std::size_t p_cols,
    const std::string& p_context
) {
    expectEqual(p_tensor.shape().rows, p_rows, p_context + " rows");
    expectEqual(p_tensor.shape().cols, p_cols, p_context + " columns");
    expectEqual(p_tensor.size(), p_rows * p_cols, p_context + " size");
}

template <typename T, typename Expected, typename Tolerance>
void expectTensorNear(
    const Tensor<T>& p_tensor,
    std::size_t p_rows,
    std::size_t p_cols,
    std::initializer_list<Expected> p_expected,
    Tolerance p_tolerance,
    const std::string& p_context
) {
    expectShape(p_tensor, p_rows, p_cols, p_context);
    expectEqual(p_expected.size(),
        p_rows * p_cols,
        p_context + " expected value count");

    auto expected = p_expected.begin();

    for (std::size_t row = 0; row < p_rows; row++) {
        for (std::size_t col = 0; col < p_cols; col++) {
            std::ostringstream element_context;
            element_context << p_context << '[' << row << ", " << col << ']';
            expectNear(p_tensor.get(row, col),
                *expected,
                p_tolerance,
                element_context.str());
            ++expected;
        }
    }
}

class TempDirectory {
  public:
    explicit TempDirectory(
        const std::string& p_name
    ) {
        const auto timestamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        _path = std::filesystem::temp_directory_path() /
                ("dnp-tests-" + p_name + '-' + std::to_string(timestamp));
        std::filesystem::create_directories(_path);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(_path, error);
    }

    const std::filesystem::path& path() const {
        return _path;
    }

  private:
    std::filesystem::path _path;
};

} // namespace dnp::test

#define DNP_TEST(name)                                                         \
    static void name();                                                        \
    static const ::dnp::test::Registrar name##_registrar(#name, name);         \
    static void name()

#endif
