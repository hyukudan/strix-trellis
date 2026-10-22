#pragma once
// Portable parser shared by CPU negative tests and the native TP qualification.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rank_qa {
inline void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
inline uint64_t little(const unsigned char* p, unsigned count) {
    uint64_t result = 0;
    for (unsigned k = 0; k < count; ++k) result |= uint64_t(p[k]) << (8 * k);
    return result;
}
struct Fixture {
    std::map<std::string, std::vector<unsigned char>> sections;
    uint32_t input = 0, output = 0, bits = 0, rows = 0;
    bool moe = false;
    explicit Fixture(const std::string& path) {
        const uint16_t endian = 1;
        require(*reinterpret_cast<const unsigned char*>(&endian) == 1, "Little endian host required");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "Cannot open rank fixture");
        const auto size = file.tellg();
        require(size >= 16 && size <= 256 * 1024 * 1024, "Invalid rank fixture size");
        std::vector<unsigned char> bytes(static_cast<size_t>(size));
        file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        require(bool(file) && std::memcmp(bytes.data(), "STRANK01", 8) == 0 &&
                little(bytes.data() + 8, 4) == 1, "Invalid rank fixture header");
        const uint64_t count = little(bytes.data() + 12, 4);
        require(count <= 32, "Too many rank fixture sections");
        size_t position = 16;
        for (uint64_t k = 0; k < count; ++k) {
            require(bytes.size() - position >= 16, "Truncated rank section header");
            std::string tag(reinterpret_cast<const char*>(bytes.data() + position), 8);
            const auto zero = tag.find('\0');
            if (zero != std::string::npos) {
                for (size_t j = zero; j < tag.size(); ++j) require(tag[j] == 0, "Invalid rank section name padding");
                tag.resize(zero);
            }
            require(!tag.empty(), "Empty rank section name");
            const uint64_t length = little(bytes.data() + position + 8, 8);
            position += 16;
            require(length <= bytes.size() - position && !sections.count(tag), "Invalid rank section extent/duplicate");
            sections.emplace(tag, std::vector<unsigned char>(bytes.begin() + position, bytes.begin() + position + size_t(length)));
            position += size_t(length);
        }
        require(position == bytes.size(), "Trailing rank fixture bytes");
        exact("meta", 32);
        const auto* m = sections.at("meta").data();
        input = uint32_t(little(m, 4)); output = uint32_t(little(m + 4, 4));
        bits = uint32_t(little(m + 8, 4)); rows = uint32_t(little(m + 20, 4));
        require(input && output && input % 128 == 0 && output % 128 == 0 &&
                input <= 8192 && output <= 16384 && rows == 8 &&
                little(m + 12, 4) == 2 && little(m + 16, 4) == 0x83DCD12Du,
                "Invalid rank metadata");
        moe = bits == 0;
        if (moe) {
            require(input == 5120 && output == 5120 && little(m + 24, 4) == 0 && sections.size() == 7,
                    "Invalid MoE fixture contract");
            exact("routes", 8 * 6 * 4); exact("scores", 8 * 6 * 4);
            exact("hidden", 8 * 6 * 1152 * 4); exact("each", 8 * 6 * 5120 * 4);
            exact("ref", 8 * 5120 * 4); exact("old", 8 * 5120 * 4);
            const auto ids = values<uint32_t>("routes");
            const auto scores = values<float>("scores");
            for (unsigned row = 0; row < 8; ++row) for (unsigned slot = 0; slot < 6; ++slot) {
                const auto id = ids[row * 6 + slot];
                require(id >= 17 && id <= 158 && (id - 17) % 3 == 0, "Invalid fixture expert ID");
                require(scores[row * 6 + slot] >= 0 && std::isfinite(scores[row * 6 + slot]), "Invalid fixture route score");
                for (unsigned previous = 0; previous < slot; ++previous)
                    require(id != ids[row * 6 + previous], "Duplicate fixture expert ID in a row");
            }
            for (const char* key : {"hidden", "each", "ref", "old"}) finite<float>(key);
        } else {
            require(bits >= 2 && bits <= 5 && little(m + 24, 4) == 1 && sections.size() == 10,
                    "Invalid projection fixture contract");
            exact("trellis", size_t(input) * output * bits / 8);
            exact("suh", size_t(input) * 2); exact("svh", size_t(output) * 2);
            exact("bias", size_t(output) * 4); exact("x", size_t(rows) * input * 4);
            exact("old", size_t(rows) * output * 4); exact("ref", size_t(rows) * output * 4);
            exact("base", size_t(rows) * output * 4); exact("bound", size_t(rows) * output * 8);
            for (const char* key : {"bias", "x", "old", "ref", "base"}) finite<float>(key);
            for (double value : values<double>("bound")) require(std::isfinite(value) && value >= 0, "Invalid rank error bound");
            for (const char* key : {"suh", "svh"}) for (uint16_t value : values<uint16_t>(key))
                require((value & 0x7c00) != 0x7c00, "Nonfinite rank half scale");
        }
    }
    void exact(const std::string& name, size_t size) const {
        const auto found = sections.find(name);
        require(found != sections.end() && found->second.size() == size, "Bad rank fixture section length");
    }
    template<class T> std::vector<T> values(const std::string& name) const {
        const auto& data = sections.at(name);
        require(data.size() % sizeof(T) == 0, "Mis-sized rank fixture scalar payload");
        std::vector<T> result(data.size() / sizeof(T));
        if (!data.empty()) std::memcpy(result.data(), data.data(), data.size());
        return result;
    }
    template<class T> void finite(const std::string& name) const {
        for (T value : values<T>(name)) require(std::isfinite(value), "Nonfinite rank fixture value");
    }
};
} // namespace rank_qa
