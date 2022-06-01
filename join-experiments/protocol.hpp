#pragma once

#include <string>
#include "highfive/H5Easy.hpp"

std::string expect(std::string what) {
    std::cerr << "[c++] Expecting to receive `sppv1 " << what << "`" << std::endl;
    std::string head = "sppv1 " + what;
    std::string protocol_line;
    std::getline(std::cin, protocol_line);
    std::cerr << "[c++] received`" << protocol_line << "`" << std::endl;
    if (protocol_line.find(head) != 0) {
        std::cout << "sppv1 err" << std::endl;
        throw std::invalid_argument("invalid message received");
    }
    std::string toret = protocol_line.substr(head.size());
    while (toret.size() > 0 && toret.at(0) == ' ') {
        toret.erase(0, 1); // strip the first character
    }
    return toret;
}

std::string protocol_read() {
    // std::cerr << "[c++] Expecting to receive `sppv1 " << what << "`" << std::endl;
    std::string protocol_line;
    std::getline(std::cin, protocol_line);
    // std::cerr << "[c++] received`" << protocol_line << "`" << std::endl;
    // Remove the prefix
    return protocol_line.substr(6);
}

void send(std::string what) {
    std::cout << "sppv1 " << what << std::endl;
}

float norm(std::vector<float> & v) {
    float n = 0.0;
    for (auto x : v) {
        n += x * x;
    }
    return n;
}

std::vector<std::vector<float>> read_float_vectors_hdf5(bool normalize) {
    std::string path = expect("path");
    std::cerr << "[c++] path" << path << std::endl;
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    std::vector<std::vector<float>> data = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");
    if (normalize) {
        for (size_t i=0; i<data.size(); i++) {
            float n = norm(data[i]);
            for (size_t j=0; j<data[i].size(); j++) {
                data[i][j] /= n;
            }
        }
    }
    return data;
}

std::vector<std::vector<uint32_t>> read_int_vectors_hdf5() {
    std::string path = expect("path");
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    std::vector<uint32_t> data = H5Easy::load<std::vector<uint32_t>>(file, "/train");
    std::vector<size_t> sizes = H5Easy::load<std::vector<size_t>>(file, "/size_train");
    size_t offset = 0;
    std::vector<std::vector<uint32_t>> res;
    for (size_t s : sizes) {
        std::vector<uint32_t> elem(data.begin() + offset, data.begin() + offset + s);
        res.push_back(elem);
        offset += s;
    }
    return res;
}

std::vector<std::vector<float>> read_vectors_stdin() {
    std::vector<std::vector<float>> vectors;
    // Read vectors from standard input
    std::string full_line;
    while(std::getline(std::cin, full_line)) {
        if (full_line.size() == 0) {
            break;
        }
        
        std::vector<float> row;
        std::istringstream line(full_line);
        float val;
        while (line >> val) {
            row.push_back(val);
        }
        vectors.push_back(row);
    }
    return vectors;
}

std::vector<std::vector<uint32_t>> read_int_vectors_stdin() {
    std::vector<std::vector<uint32_t>> vectors;
    // Read vectors from standard input
    std::string full_line;
    while(std::getline(std::cin, full_line)) {
        if (full_line.size() == 0) {
            break;
        }
        
        // std::cerr << "[c++] " << full_line <<std::endl;
        std::vector<uint32_t> row;
        std::istringstream line(full_line);
        uint32_t val;
        while (line >> val) {
            row.push_back(val);
        }
        vectors.push_back(row);
    }
    return vectors;
}


