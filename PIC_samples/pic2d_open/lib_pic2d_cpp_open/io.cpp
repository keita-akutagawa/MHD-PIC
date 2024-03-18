#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;


void io_xvKE(const vector<double> r, const vector<double> v, 
             const vector<double> cross_r_ion, const vector<double> cross_v_ion, 
             const vector<double> cross_r_electron, const vector<double> cross_v_electron, 
             string dirname, string filename, int k, 
             int total_ion, int total_electron, int n_particle,
             int in_ion, int in_electron, 
             int out_ion, int out_electron,  
             int n_x, int n_y, double m_ion, double m_electron)
{
    //ios::sync_with_stdio(false);
    //cin.tie(nullptr);

    string path_x, path_v, path_ion_number, path_electron_number, path_KE;

    path_x = "./" + dirname + "/" + filename + "_x_" + to_string(k) + ".csv";
    path_v = "./" + dirname + "/" + filename + "_v_" + to_string(k) + ".csv";
    path_ion_number = "./" + dirname + "/" + filename + "_ion_number_" + to_string(k) + ".csv";
    path_electron_number = "./" + dirname + "/" + filename + "_electron_number_" + to_string(k) + ".csv";
    path_KE = "./" + dirname + "/" + filename + "_KE_" + to_string(k) + ".csv";
    ofstream file_x(path_x);
    ofstream file_v(path_v);
    ofstream file_ion_number(path_ion_number);
    ofstream file_electron_number(path_electron_number);
    ofstream file_KE(path_KE);
    double KE = 0.0;

    for (int i = 0; i < 3 * total_ion; i+=3) {
        file_x << setprecision(16) << r[i] << ',' << r[i+1] << ',' << r[i+2] << "\n";
        file_v << setprecision(16) << v[i] << ',' << v[i+1] << ',' << v[i+2] << "\n";
        KE += 1.0 / 2.0 * m_ion * (v[i] * v[i] + v[i+1] * v[i+1] + v[i+2] * v[i+2]);
    }
    for (int i = 3 * n_particle / 2; i < 3 * (n_particle / 2 + total_electron); i+=3) {
        file_x << setprecision(16) << r[i] << ',' << r[i+1] << ',' << r[i+2] << "\n";
        file_v << setprecision(16) << v[i] << ',' << v[i+1] << ',' << v[i+2] << "\n";
        KE += 1.0 / 2.0 * m_electron * (v[i] * v[i] + v[i+1] * v[i+1] + v[i+2] * v[i+2]);
    }
    file_KE << setprecision(16) << KE;
    file_ion_number << total_ion << "\n";
    file_ion_number << in_ion << "\n";
    file_ion_number << out_ion << "\n";
    file_electron_number << total_electron << "\n";
    file_electron_number << in_electron << "\n";
    file_electron_number << out_electron << "\n";
}


void io_EBmoment(const vector<vector<vector<double>>> E, 
                 const vector<vector<vector<double>>> B, 
                 const vector<vector<double>> rho, 
                 vector<vector<double>>& zeroth_moment_ion, 
                 vector<vector<double>>& zeroth_moment_electron,  
                 vector<vector<vector<double>>>& first_moment_ion, 
                 vector<vector<vector<double>>>& first_moment_electron, 
                 vector<vector<vector<double>>>& second_moment_ion, 
                 vector<vector<vector<double>>>& second_moment_electron, 
                 string dirname, string filename, int k, 
                 int n_x, int n_y, double dx, double dy, double epsilon0, double mu_0)
{
    //ios::sync_with_stdio(false);
    //cin.tie(nullptr);

    string path_E, path_B, path_energy_E, path_energy_B, path_div_E_error, path_div_B_error; 
    string path_zeroth_momemt_ion, path_zeroth_momemt_electron; 
    string path_first_momemt_ion, path_first_momemt_electron; 
    string path_second_momemt_ion, path_second_momemt_electron; 

    path_E = "./" + dirname + "/" + filename + "_E_" + to_string(k) + ".csv";
    path_B = "./" + dirname + "/" + filename + "_B_" + to_string(k) + ".csv";
    path_energy_E = "./" + dirname + "/" + filename + "_energy_E_" + to_string(k) + ".csv";
    path_energy_B = "./" + dirname + "/" + filename + "_energy_B_" + to_string(k) + ".csv";
    path_div_E_error = "./" + dirname + "/" + filename + "_div_E_error_" + to_string(k) + ".csv";
    path_div_B_error = "./" + dirname + "/" + filename + "_div_B_error_" + to_string(k) + ".csv";
    path_zeroth_momemt_ion = "./" + dirname + "/" + filename + "_zeroth_moment_ion_" + to_string(k) + ".csv";
    path_zeroth_momemt_electron = "./" + dirname + "/" + filename + "_zeroth_moment_electron_" + to_string(k) + ".csv";
    path_first_momemt_ion = "./" + dirname + "/" + filename + "_first_moment_ion_" + to_string(k) + ".csv";
    path_first_momemt_electron = "./" + dirname + "/" + filename + "_first_moment_electron_" + to_string(k) + ".csv";
    path_second_momemt_ion = "./" + dirname + "/" + filename + "_second_moment_ion_" + to_string(k) + ".csv";
    path_second_momemt_electron = "./" + dirname + "/" + filename + "_second_moment_electron_" + to_string(k) + ".csv";
    ofstream file_E(path_E);
    ofstream file_B(path_B);
    ofstream file_energy_E(path_energy_E);
    ofstream file_energy_B(path_energy_B);
    ofstream file_div_E_error(path_div_E_error);
    ofstream file_div_B_error(path_div_B_error);
    ofstream file_zeroth_moment_ion(path_zeroth_momemt_ion);
    ofstream file_zeroth_moment_electron(path_zeroth_momemt_electron);
    ofstream file_first_moment_ion(path_first_momemt_ion);
    ofstream file_first_moment_electron(path_first_momemt_electron);
    ofstream file_second_moment_ion(path_second_momemt_ion);
    ofstream file_second_moment_electron(path_second_momemt_electron);

    double energy_E = 0.0, energy_B = 0.0;
    for (int i = 0; i < n_x; i++) {
        for (int j = 0; j < n_y; j++) {
            file_E << setprecision(16) << E[0][i][j] << ',' << E[1][i][j] << ',' << E[2][i][j] << ',' 
            << i << ',' << j << ',' << 0 << "\n";
            energy_E += 1.0/2.0 * epsilon0 * (E[0][i][j] * E[0][i][j] + E[1][i][j] * E[1][i][j] + E[2][i][j] * E[2][i][j]);
        
            file_B << setprecision(16) << B[0][i][j] << ',' << B[1][i][j] << ',' << B[2][i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";
            energy_B += 1.0/2.0 / mu_0 * (B[0][i][j] * B[0][i][j] + B[1][i][j] * B[1][i][j] + B[2][i][j] * B[2][i][j]);

            file_zeroth_moment_ion << setprecision(16) << zeroth_moment_ion[i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";

            file_zeroth_moment_electron << setprecision(16) << zeroth_moment_electron[i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";

            file_first_moment_ion << setprecision(16) << first_moment_ion[0][i][j] << ',' << first_moment_ion[1][i][j] << ',' << first_moment_ion[2][i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";

            file_first_moment_electron << setprecision(16) << first_moment_electron[0][i][j] << ',' << first_moment_electron[1][i][j] << ',' << first_moment_electron[2][i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";

            file_second_moment_ion << setprecision(16) 
            << second_moment_ion[0][i][j] << ',' << second_moment_ion[1][i][j] << ',' << second_moment_ion[2][i][j] << ','
            << second_moment_ion[3][i][j] << ',' << second_moment_ion[4][i][j] << ',' << second_moment_ion[5][i][j] << ','
            << second_moment_ion[6][i][j] << ',' << second_moment_ion[7][i][j] << ',' << second_moment_ion[8][i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";

            file_second_moment_electron << setprecision(16) 
            << second_moment_electron[0][i][j] << ',' << second_moment_electron[1][i][j] << ',' << second_moment_electron[2][i][j] << ','
            << second_moment_electron[3][i][j] << ',' << second_moment_electron[4][i][j] << ',' << second_moment_electron[5][i][j] << ','
            << second_moment_electron[6][i][j] << ',' << second_moment_electron[7][i][j] << ',' << second_moment_electron[8][i][j] << ','
            << i << ',' << j << ',' << 0 << "\n";
        }
    }

    double div_E = 0.0, div_B = 0.0, div_E_rho_error = 0.0; 
    for (int i = 0; i < n_x-1; i++) {
        for (int j = 0; j < n_y-1; j++) {
            div_B = max(div_B, abs((B[0][(i+1)%n_x][j] - B[0][i][j]) / dx + (B[1][i][(j+1)%n_y] - B[1][i][j]) / dy));
            //if (abs((B[0][(i+1)%n_x][j] - B[0][i][j]) / dx + (B[1][i][(j+1)%n_y] - B[1][i][j]) / dy) > 1e-4) cout << abs((B[0][(i+1)%n_x][j] - B[0][i][j]) / dx + (B[1][i][(j+1)%n_y] - B[1][i][j]) / dy) << " " << i << " " << j << "\n";
        }
    }
    for (int i = 1; i < n_x; i++) {
        for (int j = 1; j < n_y; j++) {
            div_E = (E[0][i][j] - E[0][(i-1+n_x)%n_x][j]) / dx + (E[1][i][j] - E[1][i][(j-1+n_y)%n_y]) / dy;
            div_E_rho_error = max(div_E_rho_error, abs(div_E - rho[i][j]));
        }
    }

    file_energy_E << setprecision(16) << energy_E;
    file_energy_B << setprecision(16) << energy_B;
    file_div_E_error << setprecision(16) << div_E_rho_error;
    file_div_B_error << setprecision(16) << div_B;
    
}


void get_moment(vector<vector<double>>& zeroth_moment, 
                vector<vector<vector<double>>>& first_moment,
                vector<vector<vector<double>>>& second_moment,  
                vector<int>& r_index, 
                vector<double>& cr, 
                vector<double>& gamma,
                const vector<double> r, 
                const vector<double> v, 
                int n_start, int n_last, 
                int n_x, int n_y, double dx, double dy, double c)
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

        tmp = 1.0 / gamma[i/3];
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

            zeroth_moment[x_index_1][y_index_1] += cx2cy2;
            zeroth_moment[x_index_1][y_index_2] += cx2cy1;
            zeroth_moment[x_index_2][y_index_1] += cx1cy2;
            zeroth_moment[x_index_2][y_index_2] += cx1cy1;
            //////////
            first_moment[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            first_moment[0][x_index_1][y_index_2] += tmp0 * cx2cy1;
            first_moment[0][x_index_2][y_index_1] += tmp0 * cx1cy2;
            first_moment[0][x_index_2][y_index_2] += tmp0 * cx1cy1;

            first_moment[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            first_moment[1][x_index_1][y_index_2] += tmp1 * cx2cy1;
            first_moment[1][x_index_2][y_index_1] += tmp1 * cx1cy2;
            first_moment[1][x_index_2][y_index_2] += tmp1 * cx1cy1;

            first_moment[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            first_moment[2][x_index_1][y_index_2] += tmp2 * cx2cy1;
            first_moment[2][x_index_2][y_index_1] += tmp2 * cx1cy2;
            first_moment[2][x_index_2][y_index_2] += tmp2 * cx1cy1;
            //////////
            second_moment[0][x_index_1][y_index_1] += tmp0 * tmp0 * cx2cy2;
            second_moment[0][x_index_1][y_index_2] += tmp0 * tmp0 * cx2cy1;
            second_moment[0][x_index_2][y_index_1] += tmp0 * tmp0 * cx1cy2;
            second_moment[0][x_index_2][y_index_2] += tmp0 * tmp0 * cx1cy1;

            second_moment[1][x_index_1][y_index_1] += tmp0 * tmp1 * cx2cy2;
            second_moment[1][x_index_1][y_index_2] += tmp0 * tmp1 * cx2cy1;
            second_moment[1][x_index_2][y_index_1] += tmp0 * tmp1 * cx1cy2;
            second_moment[1][x_index_2][y_index_2] += tmp0 * tmp1 * cx1cy1;

            second_moment[2][x_index_1][y_index_1] += tmp0 * tmp2 * cx2cy2;
            second_moment[2][x_index_1][y_index_2] += tmp0 * tmp2 * cx2cy1;
            second_moment[2][x_index_2][y_index_1] += tmp0 * tmp2 * cx1cy2;
            second_moment[2][x_index_2][y_index_2] += tmp0 * tmp2 * cx1cy1;

            second_moment[3][x_index_1][y_index_1] += tmp1 * tmp0 * cx2cy2;
            second_moment[3][x_index_1][y_index_2] += tmp1 * tmp0 * cx2cy1;
            second_moment[3][x_index_2][y_index_1] += tmp1 * tmp0 * cx1cy2;
            second_moment[3][x_index_2][y_index_2] += tmp1 * tmp0 * cx1cy1;

            second_moment[4][x_index_1][y_index_1] += tmp1 * tmp1 * cx2cy2;
            second_moment[4][x_index_1][y_index_2] += tmp1 * tmp1 * cx2cy1;
            second_moment[4][x_index_2][y_index_1] += tmp1 * tmp1 * cx1cy2;
            second_moment[4][x_index_2][y_index_2] += tmp1 * tmp1 * cx1cy1;

            second_moment[5][x_index_1][y_index_1] += tmp1 * tmp2 * cx2cy2;
            second_moment[5][x_index_1][y_index_2] += tmp1 * tmp2 * cx2cy1;
            second_moment[5][x_index_2][y_index_1] += tmp1 * tmp2 * cx1cy2;
            second_moment[5][x_index_2][y_index_2] += tmp1 * tmp2 * cx1cy1;

            second_moment[6][x_index_1][y_index_1] += tmp2 * tmp0 * cx2cy2;
            second_moment[6][x_index_1][y_index_2] += tmp2 * tmp0 * cx2cy1;
            second_moment[6][x_index_2][y_index_1] += tmp2 * tmp0 * cx1cy2;
            second_moment[6][x_index_2][y_index_2] += tmp2 * tmp0 * cx1cy1;

            second_moment[7][x_index_1][y_index_1] += tmp2 * tmp1 * cx2cy2;
            second_moment[7][x_index_1][y_index_2] += tmp2 * tmp1 * cx2cy1;
            second_moment[7][x_index_2][y_index_1] += tmp2 * tmp1 * cx1cy2;
            second_moment[7][x_index_2][y_index_2] += tmp2 * tmp1 * cx1cy1;

            second_moment[8][x_index_1][y_index_1] += tmp2 * tmp2 * cx2cy2;
            second_moment[8][x_index_1][y_index_2] += tmp2 * tmp2 * cx2cy1;
            second_moment[8][x_index_2][y_index_1] += tmp2 * tmp2 * cx1cy2;
            second_moment[8][x_index_2][y_index_2] += tmp2 * tmp2 * cx1cy1;

        } else if (x_index_2 == n_x) {

            zeroth_moment[x_index_1][y_index_1] += cx2cy2;
            zeroth_moment[x_index_1][y_index_2] += cx2cy1;
            zeroth_moment[0][y_index_1] += cx1cy2;
            zeroth_moment[0][y_index_2] += cx1cy1;
            //////////
            first_moment[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            first_moment[0][x_index_1][y_index_2] += tmp0 * cx2cy1;
            first_moment[0][0][y_index_1] += tmp0 * cx1cy2;
            first_moment[0][0][y_index_2] += tmp0 * cx1cy1;

            first_moment[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            first_moment[1][x_index_1][y_index_2] += tmp1 * cx2cy1;
            first_moment[1][0][y_index_1] += tmp1 * cx1cy2;
            first_moment[1][0][y_index_2] += tmp1 * cx1cy1;

            first_moment[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            first_moment[2][x_index_1][y_index_2] += tmp2 * cx2cy1;
            first_moment[2][0][y_index_1] += tmp2 * cx1cy2;
            first_moment[2][0][y_index_2] += tmp2 * cx1cy1;
            //////////
            second_moment[0][x_index_1][y_index_1] += tmp0 * tmp0 * cx2cy2;
            second_moment[0][x_index_1][y_index_2] += tmp0 * tmp0 * cx2cy1;
            second_moment[0][0][y_index_1] += tmp0 * tmp0 * cx1cy2;
            second_moment[0][0][y_index_2] += tmp0 * tmp0 * cx1cy1;

            second_moment[1][x_index_1][y_index_1] += tmp0 * tmp1 * cx2cy2;
            second_moment[1][x_index_1][y_index_2] += tmp0 * tmp1 * cx2cy1;
            second_moment[1][0][y_index_1] += tmp0 * tmp1 * cx1cy2;
            second_moment[1][0][y_index_2] += tmp0 * tmp1 * cx1cy1;

            second_moment[2][x_index_1][y_index_1] += tmp0 * tmp2 * cx2cy2;
            second_moment[2][x_index_1][y_index_2] += tmp0 * tmp2 * cx2cy1;
            second_moment[2][0][y_index_1] += tmp0 * tmp2 * cx1cy2;
            second_moment[2][0][y_index_2] += tmp0 * tmp2 * cx1cy1;

            second_moment[3][x_index_1][y_index_1] +=  tmp1 * tmp0 * cx2cy2;
            second_moment[3][x_index_1][y_index_2] += tmp1 * tmp0 * cx2cy1;
            second_moment[3][0][y_index_1] += tmp1 * tmp0 * cx1cy2;
            second_moment[3][0][y_index_2] += tmp1 * tmp0 * cx1cy1;

            second_moment[4][x_index_1][y_index_1] += tmp1 * tmp1 * cx2cy2;
            second_moment[4][x_index_1][y_index_2] += tmp1 * tmp1 * cx2cy1;
            second_moment[4][0][y_index_1] += tmp1 * tmp1 * cx1cy2;
            second_moment[4][0][y_index_2] += tmp1 * tmp1 * cx1cy1;

            second_moment[5][x_index_1][y_index_1] += tmp1 * tmp2 * cx2cy2;
            second_moment[5][x_index_1][y_index_2] += tmp1 * tmp2 * cx2cy1;
            second_moment[5][0][y_index_1] += tmp1 * tmp2 * cx1cy2;
            second_moment[5][0][y_index_2] += tmp1 * tmp2 * cx1cy1;

            second_moment[6][x_index_1][y_index_1] += tmp2 * tmp0 * cx2cy2;
            second_moment[6][x_index_1][y_index_2] += tmp2 * tmp0 * cx2cy1;
            second_moment[6][0][y_index_1] += tmp2 * tmp0 * cx1cy2;
            second_moment[6][0][y_index_2] += tmp2 * tmp0 * cx1cy1;

            second_moment[7][x_index_1][y_index_1] += tmp2 * tmp1 * cx2cy2;
            second_moment[7][x_index_1][y_index_2] += tmp2 * tmp1 * cx2cy1;
            second_moment[7][0][y_index_1] += tmp2 * tmp1 * cx1cy2;
            second_moment[7][0][y_index_2] += tmp2 * tmp1 * cx1cy1;

            second_moment[8][x_index_1][y_index_1] += tmp2 * tmp2 * cx2cy2;
            second_moment[8][x_index_1][y_index_2] += tmp2 * tmp2 * cx2cy1;
            second_moment[8][0][y_index_1] += tmp2 * tmp2 * cx1cy2;
            second_moment[8][0][y_index_2] += tmp2 * tmp2 * cx1cy1;

        } else if (y_index_2 == n_y) {

            zeroth_moment[x_index_1][y_index_1] += cx2cy2;
            zeroth_moment[x_index_1][0] += cx2cy1;
            zeroth_moment[x_index_2][y_index_1] += cx1cy2;
            zeroth_moment[x_index_2][0] += cx1cy1;
            //////////
            first_moment[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            first_moment[0][x_index_1][0] += tmp0 * cx2cy1;
            first_moment[0][x_index_2][y_index_1] += tmp0 * cx1cy2;
            first_moment[0][x_index_2][0] += tmp0 * cx1cy1;

            first_moment[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            first_moment[1][x_index_1][0] += tmp1 * cx2cy1;
            first_moment[1][x_index_2][y_index_1] += tmp1 * cx1cy2;
            first_moment[1][x_index_2][0] += tmp1 * cx1cy1;

            first_moment[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            first_moment[2][x_index_1][0] += tmp2 * cx2cy1;
            first_moment[2][x_index_2][y_index_1] += tmp2 * cx1cy2;
            first_moment[2][x_index_2][0] += tmp2 * cx1cy1;
            //////////
            second_moment[0][x_index_1][y_index_1] += tmp0 * tmp0 * cx2cy2;
            second_moment[0][x_index_1][0] += tmp0 * tmp0 * cx2cy1;
            second_moment[0][x_index_2][y_index_1] += tmp0 * tmp0 * cx1cy2;
            second_moment[0][x_index_2][0] += tmp0 * tmp0 * cx1cy1;

            second_moment[1][x_index_1][y_index_1] += tmp0 * tmp1 * cx2cy2;
            second_moment[1][x_index_1][0] += tmp0 * tmp1 * cx2cy1;
            second_moment[1][x_index_2][y_index_1] += tmp0 * tmp1 * cx1cy2;
            second_moment[1][x_index_2][0] += tmp0 * tmp1 * cx1cy1;

            second_moment[2][x_index_1][y_index_1] += tmp0 * tmp2 * cx2cy2;
            second_moment[2][x_index_1][0] += tmp0 * tmp2 * cx2cy1;
            second_moment[2][x_index_2][y_index_1] += tmp0 * tmp2 * cx1cy2;
            second_moment[2][x_index_2][0] += tmp0 * tmp2 * cx1cy1;

            second_moment[3][x_index_1][y_index_1] += tmp1 * tmp0 * cx2cy2;
            second_moment[3][x_index_1][0] += tmp1 * tmp0 * cx2cy1;
            second_moment[3][x_index_2][y_index_1] += tmp1 * tmp0 * cx1cy2;
            second_moment[3][x_index_2][0] += tmp1 * tmp0 * cx1cy1;

            second_moment[4][x_index_1][y_index_1] += tmp1 * tmp1 * cx2cy2;
            second_moment[4][x_index_1][0] += tmp1 * tmp1 * cx2cy1;
            second_moment[4][x_index_2][y_index_1] += tmp1 * tmp1 * cx1cy2;
            second_moment[4][x_index_2][0] += tmp1 * tmp1 * cx1cy1;

            second_moment[5][x_index_1][y_index_1] += tmp1 * tmp2 * cx2cy2;
            second_moment[5][x_index_1][0] += tmp1 * tmp2 * cx2cy1;
            second_moment[5][x_index_2][y_index_1] += tmp1 * tmp2 * cx1cy2;
            second_moment[5][x_index_2][0] += tmp1 * tmp2 * cx1cy1;

            second_moment[6][x_index_1][y_index_1] += tmp2 * tmp0 * cx2cy2;
            second_moment[6][x_index_1][0] += tmp2 * tmp0 * cx2cy1;
            second_moment[6][x_index_2][y_index_1] += tmp2 * tmp0 * cx1cy2;
            second_moment[6][x_index_2][0] += tmp2 * tmp0 * cx1cy1;

            second_moment[7][x_index_1][y_index_1] += tmp2 * tmp1 * cx2cy2;
            second_moment[7][x_index_1][0] += tmp2 * tmp1 * cx2cy1;
            second_moment[7][x_index_2][y_index_1] += tmp2 * tmp1 * cx1cy2;
            second_moment[7][x_index_2][0] += tmp2 * tmp1 * cx1cy1;

            second_moment[8][x_index_1][y_index_1] += tmp2 * tmp2 * cx2cy2;
            second_moment[8][x_index_1][0] += tmp2 * tmp2 * cx2cy1;
            second_moment[8][x_index_2][y_index_1] += tmp2 * tmp2 * cx1cy2;
            second_moment[8][x_index_2][0] += tmp2 * tmp2 * cx1cy1;

        } else {

            zeroth_moment[x_index_1][y_index_1] += cx2cy2;
            zeroth_moment[x_index_1][0] += cx2cy1;
            zeroth_moment[0][y_index_1] += cx1cy2;
            zeroth_moment[0][0] += cx1cy1;
            //////////
            first_moment[0][x_index_1][y_index_1] += tmp0 * cx2cy2;
            first_moment[0][x_index_1][0] += tmp0 * cx2cy1;
            first_moment[0][0][y_index_1] += tmp0 * cx1cy2;
            first_moment[0][0][0] += tmp0 * cx1cy1;

            first_moment[1][x_index_1][y_index_1] += tmp1 * cx2cy2;
            first_moment[1][x_index_1][0] += tmp1 * cx2cy1;
            first_moment[1][0][y_index_1] += tmp1 * cx1cy2;
            first_moment[1][0][0] += tmp1 * cx1cy1;

            first_moment[2][x_index_1][y_index_1] += tmp2 * cx2cy2;
            first_moment[2][x_index_1][0] += tmp2 * cx2cy1;
            first_moment[2][0][y_index_1] += tmp2 * cx1cy2;
            first_moment[2][0][0] += tmp2 * cx1cy1;
            //////////
            second_moment[0][x_index_1][y_index_1] += tmp0 * tmp0 * cx2cy2;
            second_moment[0][x_index_1][0] += tmp0 * tmp0 * cx2cy1;
            second_moment[0][0][y_index_1] += tmp0 * tmp0 * cx1cy2;
            second_moment[0][0][0] += tmp0 * tmp0 * cx1cy1;

            second_moment[1][x_index_1][y_index_1] += tmp0 * tmp1 * cx2cy2;
            second_moment[1][x_index_1][0] += tmp0 * tmp1 * cx2cy1;
            second_moment[1][0][y_index_1] += tmp0 * tmp1 * cx1cy2;
            second_moment[1][0][0] += tmp0 * tmp1 * cx1cy1;

            second_moment[2][x_index_1][y_index_1] += tmp0 * tmp2 * cx2cy2;
            second_moment[2][x_index_1][0] += tmp0 * tmp2 * cx2cy1;
            second_moment[2][0][y_index_1] += tmp0 * tmp2 * cx1cy2;
            second_moment[2][0][0] += tmp0 * tmp2 * cx1cy1;

            second_moment[3][x_index_1][y_index_1] += tmp1 * tmp0 * cx2cy2;
            second_moment[3][x_index_1][0] += tmp1 * tmp0 * cx2cy1;
            second_moment[3][0][y_index_1] += tmp1 * tmp0 * cx1cy2;
            second_moment[3][0][0] += tmp1 * tmp0 * cx1cy1;

            second_moment[4][x_index_1][y_index_1] += tmp1 * tmp1 * cx2cy2;
            second_moment[4][x_index_1][0] += tmp1 * tmp1 * cx2cy1;
            second_moment[4][0][y_index_1] += tmp1 * tmp1 * cx1cy2;
            second_moment[4][0][0] += tmp1 * tmp1 * cx1cy1;

            second_moment[5][x_index_1][y_index_1] += tmp1 * tmp2 * cx2cy2;
            second_moment[5][x_index_1][0] += tmp1 * tmp2 * cx2cy1;
            second_moment[5][0][y_index_1] += tmp1 * tmp2 * cx1cy2;
            second_moment[5][0][0] += tmp1 * tmp2 * cx1cy1;

            second_moment[6][x_index_1][y_index_1] += tmp2 * tmp0 * cx2cy2;
            second_moment[6][x_index_1][0] += tmp2 * tmp0 * cx2cy1;
            second_moment[6][0][y_index_1] += tmp2 * tmp0 * cx1cy2;
            second_moment[6][0][0] += tmp2 * tmp0 * cx1cy1;

            second_moment[7][x_index_1][y_index_1] += tmp2 * tmp1 * cx2cy2;
            second_moment[7][x_index_1][0] += tmp2 * tmp1 * cx2cy1;
            second_moment[7][0][y_index_1] += tmp2 * tmp1 * cx1cy2;
            second_moment[7][0][0] += tmp2 * tmp1 * cx1cy1;

            second_moment[8][x_index_1][y_index_1] += tmp2 * tmp2 * cx2cy2;
            second_moment[8][x_index_1][0] += tmp2 * tmp2 * cx2cy1;
            second_moment[8][0][y_index_1] += tmp2 * tmp2 * cx1cy2;
            second_moment[8][0][0] += tmp2 * tmp2 * cx1cy1;
        }
    }
}



