#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int
main(void)
{
    mat data;
    data.load("data.txt", raw_ascii);
    cout << data << endl;
    mat valdo_mat(size(30, 30), fill::zeros);
    for(int i = 0; i < 30; ++i)
    {
        valdo_mat.col(i) = pow(data.col(0), i);
    }
    // A.print("A: ");

    double rank_ = arma::rank(valdo_mat);
    cout << "Rank of valdermod matrix is: " << rank_ << endl;
    cout << "condition number of valdermod matrix is: " << cond(valdo_mat) << endl;

    mat U, V;
    vec sigma;
    svd(U, sigma, V, valdo_mat);

    arma::vec x_hat = arma::solve(valdo_mat, data.col(1));
    mat data_output(size(30, 3), fill::zeros);
    data_output.col(0) = data.col(0);
    data_output.col(1) = data.col(1);
    vec output = valdo_mat*x_hat;
    char name[17];
    data_output.col(2) = output;
    data_output.save("data_26.csv", csv_ascii);
    mat A(size(30, 30), fill::zeros);
    for(int i = 0; i < 13; i++)
    {
        A = A + sigma[i] * U.cols(i,i) * V.cols(i,i).t();
        if(i==8 || i==10 || i==12)
        {
            x_hat = arma::solve(A, data.col(1));
            vec output = A*x_hat;
            data_output.col(2) = output;
            snprintf(name, 17, "data_rank_%d.csv", i);
            data_output.save(name, csv_ascii);
        }
    }
    return 0;
}
