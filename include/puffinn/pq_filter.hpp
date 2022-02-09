#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include <vector>
#include <cfloat>
#include <iostream>
namespace puffinn{
    template<typename TFormat>
    class PQFilter{
        //hardcoded n,m for now 
        const unsigned int M;
        const unsigned char K;
        const unsigned int subspaceSize;
        //codebook that contains m*k centroids
        std::vector<Dataset<TFormat>> codebook;
        Dataset<TFormat> &dataset;
        public:
        PQFilter(Dataset<TFormat> &dataset, unsigned int m = 16, unsigned int k = 256)
        :dataset(dataset),
        M(m),
        K(k),
        subspaceSize(dataset.get_description().storage_len/M)
        {
            initCodebook();
        }
        private:
        //Runs kmeans for all m subspaces and stores the centroids in codebooks
        void initCodebook(){
            KMeans<TFormat> kmeans(dataset, K, subspaceSize);
            for (size_t m = 0; m < M; m++)
            {
                //change KMeans to work in subspace of dataset;
                kmeans.fit(m);
                //std::cout << "hue" << std::endl;
                codebook.push_back(kmeans.getCentroids());
                //std::cout << "euh" << std::endl << std::endl;
            }
            showCodebook();
            
        }

        void showCodebook(){
            for(int m = 0; m < M; m++){
                std::cout << "subspace: " << m << std::endl;
                for(int k = 0; k < K; k++){
                    std::cout << "cluster: "<< k << std::endl;
                    for(int l = 0; l < subspaceSize; l++){
                        std::cout << "\t" <<codebook[m][k][l] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            } 
        }            
    };
}