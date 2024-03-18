#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <omp.h>
//#include <filesystem>
#include <numeric>
#include <chrono>


using namespace std;

void set_initial_position_x(vector<double>&, int, int, double, double, int);
void set_initial_position_y_harris(vector<double>&, int, int, double, double, double, int);
void set_initial_position_y_background(vector<double>&, int, int, double, double, double, int);
void set_initial_position_z(vector<double>&, int, int);
void set_initial_velocity_x(vector<double>&, int, int, double, double, int); 
void set_initial_velocity_y(vector<double>&, int, int, double, double, int); 
void set_initial_velocity_z(vector<double>&, int, int, double, double, int); 

void get_rho(vector<vector<double>>&, 
             vector<int>&, 
             vector<double>&, 
             const vector<double>, 
             int, int, double, 
             int, int, double, double);
void get_rho_open(vector<vector<double>>&, 
                  vector<int>&, 
                  vector<double>&, 
                  const vector<double>, 
                  int, int, double, 
                  int, int, double, double);
void poisson_solver_jacobi(vector<vector<double>>&, 
                           const vector<vector<double>>, int, 
                           int, int, double, double, double);
void get_E(vector<vector<vector<double>>>&, vector<vector<double>>, 
           int, int, double, double);
void E_modification(vector<vector<vector<double>>>&, vector<vector<double>>&, 
                    vector<vector<double>>, vector<vector<double>>, int, 
                    int, int, double, double, double);
void sort_particles(vector<int>&, 
                    vector<double>&, vector<double>&, 
                    vector<double>&, 
                    int, int, int, int);

void time_evolution_B(vector<vector<vector<double>>>&, 
                      const vector<vector<vector<double>>>,  
                      int, int, double, double, double);
void time_evolution_E(vector<vector<vector<double>>>&, 
                      const vector<vector<vector<double>>>, 
                      const vector<vector<vector<double>>>,
                      int, int, double, double, double, 
                      double, double); 
void get_current_density(vector<vector<vector<double>>>&,  
                         vector<int>&, 
                         vector<double>&, 
                         vector<double>&, 
                         const vector<double>, 
                         const vector<double>,
                         int, int, 
                         double, int, int, double, double, double);
void get_current_density_open(vector<vector<vector<double>>>&, 
                              vector<int>&, vector<double>&, vector<double>&,
                              const vector<double>, const vector<double>, 
                              int, int, 
                              double, int, int, double, double, double);
void get_particle_field(vector<double>&, vector<double>&, 
                        vector<int>&, vector<double>&, 
                        const vector<vector<vector<double>>>, const vector<vector<vector<double>>>, 
                        const vector<double>,  
                        int, int, int, int, double, double);
void time_evolution_v(vector<double>&, vector<double>&, 
                      vector<double>&, vector<double>&, 
                      vector<double>&, vector<double>&, vector<double>&, 
                      const vector<double>, const vector<double>, 
                      int, int, double, double, double, double);
void time_evolution_x(vector<double>&, vector<double>&, 
                      const vector<double>, 
                      int, int, double, double);
void refrective_boudary_condition_x_left(vector<double>&, vector<double>&, int, int, double);
void refrective_boudary_condition_x_right(vector<double>&, vector<double>&, int, int, double);
void open_boudary_condition_x_right(vector<double>&, vector<double>&, vector<double>&, 
                                    vector<double>&, vector<double>&, 
                                    vector<double>&, vector<double>&, 
                                    int&, int&, 
                                    int, int, int, double, double);
void sort_buffer(vector<double>&, vector<double>&, vector<double>&, vector<double>&, 
                 int&, int&, int);
void refrective_boudary_condition_y(vector<double>&, vector<double>&, int, int, double, double);
void boundary_B(vector<vector<vector<double>>>&, 
                int, int, double, double, double);
void boundary_E(vector<vector<vector<double>>>&, vector<vector<double>>, 
                int, int, double, double, double);
void filter_E(vector<vector<vector<double>>>&, 
              vector<vector<double>>, vector<vector<double>>&, 
              int, int, double, double, double, 
              double, double, double);


void io_xvKE(const vector<double>, const vector<double>, 
             const vector<double>, const vector<double>, 
             const vector<double>, const vector<double>, 
             string, string, int, 
             int, int, int, 
             int, int, int, int, 
             int, int, double, double);
void io_EBmoment(const vector<vector<vector<double>>>, 
                 const vector<vector<vector<double>>>, 
                 const vector<vector<double>>, 
                 vector<vector<double>>&, vector<vector<double>>&, 
                 vector<vector<vector<double>>>&, vector<vector<vector<double>>>&, 
                 vector<vector<vector<double>>>&, vector<vector<vector<double>>>&, 
                 string, string, int, int, int, double, double, double, double);
void get_moment(vector<vector<double>>&, 
                vector<vector<vector<double>>>&,
                vector<vector<vector<double>>>&,  
                vector<int>&, 
                vector<double>&, 
                vector<double>&,
                const vector<double>, 
                const vector<double>, 
                int, int, 
                int, int, double, double, double);