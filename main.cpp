#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>
#include <cassert>
#include "sparse.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    try {
        int m = 120;
        int n = (m - 1) * (m - 1);

        double L1 = -1.0, L2 = 1.0;
        double h = (L2 - L1) / m;
        double G = 1.0, sigma = 0.1;

        std::vector<double> xx(n), yy(n), b(n), x(n, 0.0);

        int k = 0;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < m; j++) {
                xx[k] = L1 + j * h;
                yy[k] = L1 + i * h;
                k++;
            }
        }

        SparseCOO Acoo = createCOO(n, 5 * n);

        {
            auto addA = [&](int r, int c, double v) {
                if (v != 0) addEntryCOO(Acoo, r, c, v);
            };

            for (int i = 0; i < n; i++) {
                addA(i, i, -4.0);
                
                // Right neighbor (if not on right boundary)
                if ((i % (m - 1)) != (m - 2)) 
                    addA(i, i + 1, 1.0);
                
                // Left neighbor (if not on left boundary)
                if ((i % (m - 1)) != 0) 
                    addA(i, i - 1, 1.0);
                
                // Bottom neighbor
                if (i + (m - 1) < n) 
                    addA(i, i + (m - 1), 1.0);
                
                // Top neighbor
                if (i - (m - 1) >= 0) 
                    addA(i, i - (m - 1), 1.0);
            }

            SparseMatrixCSR A = convertToCSR(Acoo);
            std::cout << "Matrix has " << A.nnz << " non-zero elements\n";

            // Free COO memory immediately after conversion
            freeCOO(Acoo);

            // Set up right-hand side
            double prefactor = 4 * M_PI * G / std::sqrt(M_PI * sigma);
            for (int i = 0; i < n; i++) {
                double r2 = xx[i] * xx[i] + yy[i] * yy[i];
                b[i] = h * h * prefactor * std::exp(-r2 / (2 * sigma * sigma));
            }
            
               //gaussSeidel(A, b.data(), x.data(), 30000, 1e-5);
               gaussSeidel_forward(A,b.data(),x.data(),50000,1e-5);
            // Free CSR memory
            freeCSR(A);
        }

        // Write solution to file
        std::ofstream out("solution.dat");
        if (!out) {
            throw std::runtime_error("Failed to open solution.dat for writing");
        }
        
        out.precision(10);
        for (int i = 0; i < n; i++) {
            out << xx[i] << " " << yy[i] << " " << x[i] << "\n";
        }
        out.close();

        std::cout << "Solution written to solution.dat\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time needed for the calculation: " << elapsed.count() << " seconds\n";
    
    return EXIT_SUCCESS;
}