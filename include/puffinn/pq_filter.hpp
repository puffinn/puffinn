#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include <vector>
#include <cfloat>
#include <iostream>
namespace puffinn{
    template<typename TFormat>
    class PQFilter{
        const unsigned int M;
        const unsigned char K;
        //The size of each subspace, has issues if n % M != 0;
        const unsigned int subspaceSize;
        //codebook that contains m*k centroids
        std::vector<Dataset<TFormat>> codebook;
        Dataset<TFormat> &dataset;
        std::vector<std::vector<uint8_t>> pqCodes;
        public:
        PQFilter(Dataset<TFormat> &dataset, unsigned int m = 16, unsigned int k = 256)
        :dataset(dataset),
        M(m),
        K(k),
        pqCodes(dataset.get_size()),
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
                //fit to each subspace
                kmeans.fit(m);
                //get the resulting centroids for each subspace
                codebook.push_back(kmeans.getCentroids());
                //get gb_labels from fitting, the m'th fit will be the i'th index of the PQCodes
                uint8_t * labels = kmeans.getLabels();
                //std::cout << "gb_labels "<< std::endl;
                for(int i = 0; i < dataset.get_size(); i++){
                    //std::cout << labels[i] << std::endl;
                    pqCodes[i].push_back(labels[i]);
                }
            }
            showCodebook();
            showPQCodes();
            std::cout << "Calculating quantization error for index 1: " << quantizationError(1) << std::endl;
        }

        void showPQCodes(){
            std::cout << "PQCODE: ";
            for(std::vector<uint8_t> pqCode: pqCodes){
                for(uint8_t val: pqCode){
                    std::cout << (unsigned int) val << " ";
                }
                std::cout << std::endl;
            }

        }

        vector<uint8_t> getPQCode(int index){
            return pqCodes[index];
        }

        float quantizationError(int index){
            float sum = 0;
            typename TFormat::Type* vec = dataset[index];
            int centroidID;
            for(int m = 0; m < M; m++){
                centroidID = pqCodes[index][m];
                sum += TFormat::distance(vec + (m*subspaceSize), codebook[m][centroidID], subspaceSize);
            }
            std::cout <<" quantization error for: ";
            for(int k = 0; k < M*subspaceSize; k++){
                std::cout << vec[k] << " ";  
            }
            std::cout << std::endl;
            return sum;
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