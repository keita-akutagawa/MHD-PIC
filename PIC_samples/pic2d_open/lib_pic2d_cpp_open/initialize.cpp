#include <vector>
#include <random>
#include <cmath>
#include <omp.h>

using namespace std;

//openmpを使わないこと。STLと喧嘩しているみたい。

void set_initial_position_x(vector<double>& r, 
                          int n_start, int n_last,  
                          double x_min, double x_max, int seed)
{
    mt19937_64 mt64(seed);
    uniform_real_distribution<double> set_x(0.0, 1.0);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        double random_value = set_x(mt64);
        if (random_value == 1.0) random_value -= 1e-10;
        r[i] = random_value * (x_max - x_min) + x_min;

    }
}

void set_initial_position_y(vector<double>& r, 
                          int n_start, int n_last,  
                          double y_min, double y_max, int seed)
{
    mt19937_64 mt64(seed);
    uniform_real_distribution<double> set_x(0.0, 1.0);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        double random_value = set_x(mt64);
        if (random_value == 1.0) random_value -= 1e-10;
        r[i+1] = random_value * (y_max - y_min) + y_min;

    }
}

void set_initial_position_y_harris(vector<double>& r, 
                                   int n_start, int n_last,  
                                   double y_min, double y_max, double sheat_thickness, 
                                   int seed)
{
    mt19937_64 mt64(seed);
    uniform_real_distribution<double> set_x(0.0, 1.0);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        double random_value = set_x(mt64);
        if (random_value == 1.0) random_value -= 1e-10;
        double y_position = y_max / 2.0 + sheat_thickness * atanh(2.0 * random_value - 1.0);
        if (y_position > y_max) y_position = y_max / 2.0; 
        if (y_position < y_min) y_position = y_max / 2.0; 
        r[i+1] = y_position;
    }
}

void set_initial_position_y_background(vector<double>& r, 
                                       int n_start, int n_last,  
                                       double y_min, double y_max, double sheat_thickness, 
                                       int seed)
{
    mt19937_64 mt64(seed);
    uniform_real_distribution<double> set_x(0.0, 1.0);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        while (true)
        {
            double random_value = set_x(mt64);
            if (random_value == 1.0) random_value -= 1e-10;
            double y_position = random_value * (y_max - y_min);
            double rand_pn = set_x(mt64);
            if (rand_pn < (1.0 - 1.0 / cosh((y_position - y_max/2.0)/sheat_thickness))) {
                r[i+1] = y_position;
                break;
            }
        }
        
    }
}

void set_initial_position_z(vector<double>& r, 
                          int n_start, int n_last)
{
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        r[i+2] = 0.0;
    }
}


void set_initial_velocity_x(vector<double>& v,
                          int n_start, int n_last,  
                          double v_species, double v_th, int seed) 
{
    mt19937_64 mt64(seed);
    normal_distribution<double> set_v_comp(v_species, v_th);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        double random_value = set_v_comp(mt64);
        v[i] = random_value;
    }
}


void set_initial_velocity_y(vector<double>& v,
                          int n_start, int n_last,  
                          double v_species, double v_th, int seed) 
{
    mt19937_64 mt64(seed);
    normal_distribution<double> set_v_comp(v_species, v_th);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        double random_value = set_v_comp(mt64);
        v[i+1] = random_value;
    }
}


void set_initial_velocity_z(vector<double>& v,
                          int n_start, int n_last,  
                          double v_species, double v_th, int seed) 
{
    mt19937_64 mt64(seed);
    normal_distribution<double> set_v_comp(v_species, v_th);
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        double random_value = set_v_comp(mt64);
        v[i+2] = random_value;
    }
}


