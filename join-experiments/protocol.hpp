#pragma once

#include <string>

std::string expect(std::string what) {
    // std::cerr << "[c++] Expecting to receive `sppv1 " << what << "`" << std::endl;
    std::string protocol_line;
    std::getline(std::cin, protocol_line);
    // std::cerr << "[c++] received`" << protocol_line << "`" << std::endl;
    if (protocol_line != "sppv1 " + what) {
        std::cout << "sppv1 err" << std::endl;
        throw "invalid message received";
    }
    return protocol_line.substr(6);
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


