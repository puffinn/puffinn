#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include "puffinn/format/real_vector.hpp"

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>


using namespace puffinn;
namespace kmeans {

    
    std::unordered_map<std::string, std::vector<float>> get_data(std::string path)
    {
        using namespace std;
        unordered_map<string, vector<float>> data;

        fstream datafile;
        datafile.open(path, ios::in);
        string line, value;
        vector<float> data_entry;
        vector<string> splitted_string;
        if (datafile.is_open()) {
            while(getline(datafile, line)) {
                stringstream ss(line);
                string word;
                ss >> word;

                while(ss >> value) {
                    data[word].push_back(stof(value));
                }
            }
        }
        return data;

    }

    std::unordered_map<std::string, std::vector<float>> get_test_data()
    {
        return get_data("../data/test");
    }

    void general_test() 
    {
        // TODO:
        // test add_assign and multiply assign in math
        std::cout << "Start general_test" << std::endl;
        std::unordered_map<std::string, std::vector<float>> data = get_test_data();
        std::cout << "data loaded" << std::endl;

        Dataset<RealVectorFormat> dataset(2, data.size());
        std::cout << "dataset is initialized" << std::endl;
        for (auto entry : data) {
            dataset.insert(entry.second);
        }
        std::cout << "data inserted" << std::endl;
        KMeans<RealVectorFormat> kmeans(dataset, 8);
        kmeans.fit();

        std::cout << "test complete" << std::endl;
    }

}