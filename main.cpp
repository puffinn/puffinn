#include "puffinn.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>


std::unordered_map<std::string, std::vector<float>> get_data()
{
    using namespace std;
    unordered_map<string, vector<float>> data;

    string path = "data/glove/sample100d.txt";
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


int main() {
    std::unordered_map<std::string, std::vector<float>> data = get_data();

    std::printf("Loaded data has %zu entries\n", data.size());

    puffinn::Index<puffinn::CosineSimilarity> index(
        100,
        4LL*1024*1024*1024
    );
    for (auto p : data) {
        index.insert(p.second);
    }
    index.fit();



}
