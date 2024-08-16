#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int
main(void)
{
    arma_rng::set_seed_random();
    mat matrix = mat(4, 4, fill::randn);
    matrix.print("martix A saved is: ");
    mat U, V;
    vec s;
    svd(U, s, V, matrix);

    // rank 1 approximation
    mat app_mat = s[0] * U.cols(0,0) * V.cols(0,0).t();
    cout << "norm error: " << norm(matrix - app_mat) << "\n";
    cout << "rank of matrix is "<< arma::rank(app_mat) << endl;

    // rank 2 approximation
    mat A = s[1] * U.cols(1,1) * V.cols(1,1).t();
    cout << "orthogonality is " << app_mat.t()*A << endl;
    app_mat = app_mat + A;

    cout << "norm error: " << norm(matrix - app_mat) << "\n";
    cout << "rank of matrix is "<< arma::rank(A) << endl;

    // rank 3 approximation
    A = s[2] * U.cols(2,2) * V.cols(2,2).t();
    app_mat = app_mat + A;
    cout << "norm error: " << norm(matrix - app_mat) << "\n";
    cout << "rank of matrix is "<< arma::rank(A) << endl;


    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            if (i!=j)
            {
                app_mat = (U.cols(i, i)*V.cols(i, i).t())*(U.cols(j, j)*V.cols(j, j).t()).t();
                cout << "matrix is " << app_mat << endl;
            }
        }
    }
	return 0;
}
