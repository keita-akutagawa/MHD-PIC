#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>

using namespace std;


void get_current_density(vector<vector<vector<double>>>& current_tmp, 
                         vector<int>& r_index, 
                         vector<double>& cr, 
                         vector<double>& gamma,
                         const vector<double> r, 
                         const vector<double> v, 
                         int n_start, int n_last, 
                         double q, int n_x, int n_y, double dx, double dy, double c)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        r_index[i] = floor(r[i] / dx);
        r_index[i+1] = floor(r[i+1] / dy);
        cr[i] = r[i] / dx - r_index[i];
        cr[i+1] = r[i+1] / dy - r_index[i+1];
    }

    double tmp, tmp0, tmp1, tmp2;
    double cx1cy1, cx1cy2, cx2cy1, cx2cy2;
    int x_index_1, x_index_2, y_index_1, y_index_2;
    
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {

        tmp = q / gamma[i/3];
        tmp0 = tmp * v[i];
        tmp1 = tmp * v[i+1];
        tmp2 = tmp * v[i+2];

        cx1cy1 = cr[i] * cr[i+1];
        cx1cy2 = cr[i] * (1.0 - cr[i+1]);
        cx2cy1 = (1.0 - cr[i]) * cr[i+1];
        cx2cy2 = (1.0 - cr[i]) * (1.0 - cr[i+1]);

        x_index_1 = r_index[i];
        x_index_2 = x_index_1 + 1;
        y_index_1 = r_index[i+1];
        y_index_2 = y_index_1 + 1;

        if ((x_index_2 != n_x) && (y_index_2 != n_y)) {

            current_tmp[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            current_tmp[0][x_index_1][y_index_2] += tmp0 * cx2cy1;
            current_tmp[0][x_index_2][y_index_1] += tmp0 * cx1cy2;
            current_tmp[0][x_index_2][y_index_2] += tmp0 * cx1cy1;

            current_tmp[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            current_tmp[1][x_index_1][y_index_2] += tmp1 * cx2cy1;
            current_tmp[1][x_index_2][y_index_1] += tmp1 * cx1cy2;
            current_tmp[1][x_index_2][y_index_2] += tmp1 * cx1cy1;

            current_tmp[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            current_tmp[2][x_index_1][y_index_2] += tmp2 * cx2cy1;
            current_tmp[2][x_index_2][y_index_1] += tmp2 * cx1cy2;
            current_tmp[2][x_index_2][y_index_2] += tmp2 * cx1cy1;

        } else if (x_index_2 == n_x) {
            
            current_tmp[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            current_tmp[0][x_index_1][y_index_2] += tmp0 * cx2cy1;
            //current_tmp[0][0][y_index_1] += tmp0 * cx1cy2;
            //current_tmp[0][0][y_index_2] += tmp0 * cx1cy1;

            current_tmp[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            current_tmp[1][x_index_1][y_index_2] += tmp1 * cx2cy1;
            //current_tmp[1][0][y_index_1] += tmp1 * cx1cy2;
            //current_tmp[1][0][y_index_2] += tmp1 * cx1cy1;

            current_tmp[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            current_tmp[2][x_index_1][y_index_2] += tmp2 * cx2cy1;
            //current_tmp[2][0][y_index_1] += tmp2 * cx1cy2;
            //current_tmp[2][0][y_index_2] += tmp2 * cx1cy1;

        } else if (y_index_2 == n_y) {

            current_tmp[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            //current_tmp[0][x_index_1][0] += tmp0 * cx2cy1;
            current_tmp[0][x_index_2][y_index_1] += tmp0 * cx1cy2;
            //current_tmp[0][x_index_2][0] += tmp0 * cx1cy1;

            current_tmp[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            //current_tmp[1][x_index_1][0] += tmp1 * cx2cy1;
            current_tmp[1][x_index_2][y_index_1] += tmp1 * cx1cy2;
            //current_tmp[1][x_index_2][0] += tmp1 * cx1cy1;

            current_tmp[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            //current_tmp[2][x_index_1][0] += tmp2 * cx2cy1;
            current_tmp[2][x_index_2][y_index_1] += tmp2 * cx1cy2;
            //current_tmp[2][x_index_2][0] += tmp2 * cx1cy1;

        } else {

            current_tmp[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            //current_tmp[0][x_index_1][0] += tmp0 * cx2cy1;
            //current_tmp[0][0][y_index_1] += tmp0 * cx1cy2;
            //current_tmp[0][0][0] += tmp0 * cx1cy1;

            current_tmp[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            //current_tmp[1][x_index_1][0] += tmp1 * cx2cy1;
            //current_tmp[1][0][y_index_1] += tmp1 * cx1cy2;
            //current_tmp[1][0][0] += tmp1 * cx1cy1;

            current_tmp[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            //current_tmp[2][x_index_1][0] += tmp2 * cx2cy1;
            //current_tmp[2][0][y_index_1] += tmp2 * cx1cy2;
            //current_tmp[2][0][0] += tmp2 * cx1cy1;
        }
    }
}


void get_current_density_open(vector<vector<vector<double>>>& current_tmp, 
                              vector<int>& cross_r_index, 
                              vector<double>& cross_cr, 
                              vector<double>& cross_gamma,
                              const vector<double> cross_r, 
                              const vector<double> cross_v, 
                              int n_start, int n_last, 
                              double q, int n_x, int n_y, double dx, double dy, double c)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        cross_r_index[i] = floor(cross_r[i] / dx);
        cross_r_index[i+1] = floor(cross_r[i+1] / dy);
        cross_cr[i] = cross_r[i] / dx - cross_r_index[i];
        cross_cr[i+1] = cross_r[i+1] / dy - cross_r_index[i+1];
    }

    double tmp, tmp0, tmp1, tmp2;
    double cx1cy1, cx1cy2, cx2cy1, cx2cy2;
    int x_index_1, x_index_2, y_index_1, y_index_2;
    
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {

        tmp = q / cross_gamma[i/3];
        tmp0 = tmp * cross_v[i];
        tmp1 = tmp * cross_v[i+1];
        tmp2 = tmp * cross_v[i+2];

        cx1cy1 = cross_cr[i] * cross_cr[i+1];
        cx1cy2 = cross_cr[i] * (1.0 - cross_cr[i+1]);
        cx2cy1 = (1.0 - cross_cr[i]) * cross_cr[i+1];
        cx2cy2 = (1.0 - cross_cr[i]) * (1.0 - cross_cr[i+1]);

        y_index_1 = cross_r_index[i+1];
        y_index_2 = y_index_1 + 1;

        if (y_index_2 != n_y) {

            current_tmp[0][n_x-1][y_index_1] += tmp0 * cx2cy2;
            current_tmp[0][n_x-1][y_index_2] += tmp0 * cx2cy1;

            current_tmp[1][n_x-1][y_index_1] += tmp1 * cx2cy2;
            current_tmp[1][n_x-1][y_index_2] += tmp1 * cx2cy1;

            current_tmp[2][n_x-1][y_index_1] += tmp2 * cx2cy2;
            current_tmp[2][n_x-1][y_index_2] += tmp2 * cx2cy1;

        } else {

            current_tmp[0][n_x-1][y_index_1] += tmp0 * cx2cy2;
            current_tmp[0][n_x-1][0] += tmp0 * cx2cy1;

            current_tmp[1][n_x-1][y_index_1] += tmp1 * cx2cy2;
            current_tmp[1][n_x-1][0] += tmp1 * cx2cy1;
            
            current_tmp[2][n_x-1][y_index_1] += tmp2 * cx2cy2;
            current_tmp[2][n_x-1][0] += tmp2 * cx2cy1;

        }
    }
}




