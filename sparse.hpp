#ifndef SPARSE_HPP
#define SPARSE_HPP

struct SparseCOO {
    double* val = nullptr;
    int* row = nullptr;
    int* col = nullptr;
    int n = 0;
    int nnz = 0;
    int capacity = 0;
};

struct SparseMatrixCSR {
    double* val = nullptr;
    int* col = nullptr;
    int* row_ptr = nullptr;
    int n = 0;
    int nnz = 0;
};

SparseCOO createCOO(int n, int capacity);
void addEntryCOO(SparseCOO& A, int r, int c, double value);
void freeCOO(SparseCOO& A);
SparseMatrixCSR convertToCSR(const SparseCOO& Acoo);
void freeCSR(SparseMatrixCSR& A);
void matVec(const SparseMatrixCSR& A, const double* x, double* y);
double vectorNorm(const double* v, int n);
double residualNorm(const SparseMatrixCSR& A, const double* b, const double * x);
void forwardSubstitution(const SparseMatrixCSR& A,const double*r ,double* y);
void gaussSeidel(const SparseMatrixCSR& A, const double* b, double* x, int maxIter, double tol) ;
void gaussSeidel_forward(const SparseMatrixCSR& A, const double* b, double* x, int maxIter, double tol);

#endif