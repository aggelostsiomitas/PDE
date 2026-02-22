#include <cmath>
#include <iostream>
#include <stdexcept>
#include "sparse.hpp"

SparseCOO createCOO(int n, int capacity) {
    if (n <= 0 || capacity <= 0)
        throw std::invalid_argument("Matrix size and capacity must be positive");

    SparseCOO A;
    A.n = n;
    A.capacity = capacity;
    A.nnz = 0;

    try {
        A.val = new double[capacity];
        A.row = new int[capacity];
        A.col = new int[capacity];
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("Memory allocation failed in createCOO");
    }

    return A;
}

void addEntryCOO(SparseCOO& A, int r, int c, double value) {
    if (A.nnz >= A.capacity)
        throw std::overflow_error("Exceeded COO capacity");

    if (r < 0 || r >= A.n || c < 0 || c >= A.n)
        throw std::out_of_range("Invalid row or column index");

    A.val[A.nnz] = value;
    A.row[A.nnz] = r;
    A.col[A.nnz] = c;
    A.nnz++;
}

void freeCOO(SparseCOO& A) {
    delete[] A.val;
    delete[] A.row;
    delete[] A.col;
    A.val = nullptr;
    A.row = nullptr;
    A.col = nullptr;
    A.nnz = A.n = A.capacity = 0;
}

SparseMatrixCSR convertToCSR(const SparseCOO& Acoo) {
    SparseMatrixCSR A;
    A.n = Acoo.n;
    A.nnz = Acoo.nnz;

    try {
        A.val = new double[A.nnz];
        A.col = new int[A.nnz];
        A.row_ptr = new int[A.n + 1];
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("Memory allocation failed in convertToCSR");
    }

    for (int i = 0; i <= A.n; i++)
        A.row_ptr[i] = 0; //in the beginning all zeros 

    for (int k = 0; k < A.nnz; k++) {
        int r = Acoo.row[k]; //non zero elemnts in each row 
        if (r < 0 || r >= A.n)
            throw std::out_of_range("Invalid row index in COO");
        A.row_ptr[r + 1]++;  //increase by one 
    }

    for (int i = 1; i <= A.n; i++)
        A.row_ptr[i] += A.row_ptr[i - 1]; 

    int* counter = nullptr;
    try {
        counter = new int[A.n];
    } catch (...) {
        delete[] A.val;
        delete[] A.col;
        delete[] A.row_ptr;
        throw std::runtime_error("Memory allocation failed for counter");
    }

    for (int i = 0; i < A.n; i++)
        counter[i] = A.row_ptr[i];

    for (int k = 0; k < A.nnz; k++) {
        int r = Acoo.row[k];
        int dest = counter[r]++;
        A.val[dest] = Acoo.val[k];
        A.col[dest] = Acoo.col[k];
    }

    delete[] counter;
    return A;
}

void freeCSR(SparseMatrixCSR& A) {
    delete[] A.val;
    delete[] A.col;
    delete[] A.row_ptr;
    A.val = nullptr;
    A.col = nullptr;
    A.row_ptr = nullptr;
    A.nnz = A.n = 0;
}

void matVec(const SparseMatrixCSR& A, const double* x, double* y) {
    if (!x || !y)
        throw std::invalid_argument("Null vector passed to matVec");

    for (int i = 0; i < A.n; i++) {
        y[i] = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            int j = A.col[k];
            y[i] += A.val[k] * x[j];
        }
    }
}


double vectorNorm(const double* v, int n)
{
    double sum=0.0;
    for(int i=0;i<n;i++)
    {
        sum+=v[i]*v[i];
    }
    return std::sqrt(sum);
}

double residualNorm(const SparseMatrixCSR& A, const double* b ,const double* x )
{
    double norm2=0.0;
    for(int i=0;i<A.n;i++)
    {
        double Ax_i=0.0;
        for(int k=A.row_ptr[i];k<A.row_ptr[i+1];k++)
        {
            Ax_i+=A.val[k]*x[A.col[k]];
        }
        double r_i=b[i]-Ax_i;
        norm2+=r_i*r_i;
    }
    return std::sqrt(norm2);
}

// Solve (D+L) y = r using forward substitution
void forwardSubstitution(const SparseMatrixCSR& A, const double* r, double* y) {
    for (int i = 0; i < A.n; ++i) {
        double sum = 0.0;
        double diag = 0.0;

        // Traverse row i
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            int j = A.col[k];
            double aij = A.val[k];
            if (j < i) {
                sum += aij * y[j];   // already solved
            } else if (j == i) {
                diag = aij;          // diagonal element
            }
        }

        if (diag == 0.0)
            throw std::runtime_error("Zero diagonal in forward substitution");

        y[i] = (r[i] - sum) / diag;
    }
}


void gaussSeidel(const SparseMatrixCSR& A, const double* b, double* x, int maxIter, double tol) {
    if (!b || !x)
        throw std::invalid_argument("Null vector passed to gaussSeidel");

   //computer the norm of b
   double normb=vectorNorm(b,A.n);

   //start guass -seidle itreation
   for(int iter=0; iter<maxIter;iter++)
   {

    for(int i=0;i<A.n;i++)
   {
    double diag=0.0;
    double sum=0.0;

    for(int k=A.row_ptr[i];k<A.row_ptr[i+1]; k++)
    {
        int j=A.col[k];
        double aij=A.val[k];
        if(j==i)
        {
            diag=aij;
        }
        else 
        {
            sum+=aij*x[j];
        }  
    }
        x[i]=(b[i]-sum)/diag;
    }
    double resNorm=residualNorm(A,b,x);
    if(resNorm<tol*normb)
    {
        std::cout<<"converged after :"<< iter<< " iterations\n";
        return ;
    }
   }
}


void gaussSeidel_forward(const SparseMatrixCSR& A, const double* b, double* x, int maxIter, double tol) {
    if (!b || !x)
        throw std::invalid_argument("Null vector passed to gaussSeidel_forward");

    double normb = vectorNorm(b, A.n);
    if (normb == 0.0) normb = 1.0;

    double* Ax = new double[A.n];
    double* r  = new double[A.n];
    double* y  = new double[A.n];

    for (int iter = 0; iter < maxIter; ++iter) {

       
        // r = b - A*x
        for (int i = 0; i < A.n; ++i) {
            double s = 0.0;
            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                s += A.val[k] * x[A.col[k]];
            Ax[i] = s;
            r[i]  = b[i] - s;
        }

        // (D+L) y = r
        forwardSubstitution(A, r, y);

        // x = x + y
        for (int i = 0; i < A.n; ++i)
            x[i] += y[i];

        // residual check
        double resNorm = 0.0;
        for (int i = 0; i < A.n; ++i) {
            double ri = r[i];
            resNorm += ri * ri;
        }
        resNorm = std::sqrt(resNorm);


        if (resNorm < tol * normb) {
            std::cout << "Converged after " << (iter + 1)
                      << " iterations, residual norm: " << resNorm << "\n";
            delete[] Ax; delete[] r; delete[] y;
            return;
        }

        if(iter%1000==0)
        {
            std::cout<<"iteration :"<<iter<< " norm : "<<resNorm<<'\n';
        }
    }

    delete[] Ax; delete[] r; delete[] y;
    std::cout << "Stopped at maxIter with residual norm >= tol*normb\n";
}


