#include <exception>
#include <fstream>
#include <string>
#include <cstdarg>

void f(const std::string &filename) {
    std::fstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
        return;
    }
    file << "HelloWorld!";
    std::terminate();
}

int add(int first, int second, ...) {
    int r = first + second;
    va_list va;
    va_start(va, second);
    while (int v = va_arg(va, int)) {
        r += v;
    }
}

auto g() {
    int i = 12;
    return [&] {
        i = 100;
        return i;
    };
}

void ff() {
    int j = g()();
}