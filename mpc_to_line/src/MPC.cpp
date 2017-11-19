#include "MPC.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/QR"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using CppAD::AD;

// define N and dt
const size_t N = 25;
const double dt = 0.05;

// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// NOTE: feel free to play around with this
// or do something completely different
const double ref_v = 40;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lives easier.
// define array positions of state variables, errors, actuators
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
public:
  // property
  Eigen::VectorXd coeffs;

  // Coefficients of the fitted polynomial.
  // constructor
  FG_eval(const Eigen::VectorXd& coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  // () operator overloaded
  void operator()(ADvector &fg, const ADvector &vars) {
    /*
     * fg is a vector where cost function and vehicle model/constraints are defined
     * vars is a vector containing all variables used by cost function i.e:
     *    x, y, v, psi --> state
     *    cte, epsi    --> errors
     *    delta,a      --> control inputs
     * fg[0] contains cost
     */

    // set initial cost
    fg[0] = 0;
    // cost based on reference state
    for (int t = 0; t < N; t++) {
      // cte error
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      // orientation (heading) error
      fg[0] += CppAD::pow(vars[epsi_start + t], 2);
      // velocity error
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }
    // cost based on control inputs
    // no abrupt control input change
    for (int t = 0; t < N - 1; t++) {
      fg[0] += 200 * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t], 2);
    }
    // minimize value gap between sequential actuations
    for (int t = 0; t < N - 2; t++) {
      fg[0] += 500 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // set addresses of state, and errors variables
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // starting from 1
    // 0 is initial state
    // initial states are not part of optimizer solver
    for (int t = 1; t < N; t++) {
      AD<double> x0 = vars[x_start + t - 1], x1 = vars[x_start + t];
      AD<double> y0 = vars[y_start + t - 1], y1 = vars[y_start + t];
      AD<double> psi0 = vars[psi_start + t - 1], psi1 = vars[psi_start + t];
      AD<double> v0 = vars[v_start + t - 1], v1 = vars[v_start + t];
      AD<double> cte0 = vars[cte_start + t - 1], cte1 = vars[cte_start + t];
      AD<double> epsi0 = vars[epsi_start + t - 1], epsi1 = vars[epsi_start + t];
      AD<double> delta0 = vars[delta_start + t - 1], delta1 = vars[delta_start + t];
      AD<double> a0 = vars[a_start + t - 1];
      AD<double> f0 = coeffs[0] + coeffs[1] * x0;
      AD<double> psi_des0 = CppAD::atan(coeffs[1]);

      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * (delta0 / Lf) * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] = cte1 - ((f0 - y0) + v0 * CppAD::sin(epsi0) * dt);
      fg[1 + epsi_start + t] = epsi1 - (psi0 - psi_des0 + v0 * (delta0 / Lf) * dt);
    }
  }
};

//
// MPC class definition
//

MPC::MPC() {}

MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd x0, Eigen::VectorXd coeffs) {
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = x0[0];
  double y = x0[1];
  double psi = x0[2];
  double v = x0[3];
  double cte = x0[4];
  double epsi = x0[5];

  // N actuations -> N - 1
  size_t n_vars = N * 6 + (N - 1) * 2;
  // number of constraints
  size_t n_constraints = N * 6;

  Dvector vars(n_vars);
  for (size_t t = 0; t < n_vars; t++) {
    vars[t] = 0.0;
  }

  // set initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // lower and upper boundaries for variables
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // boundaries for x, y, psi, v, cte, epsi
  for (int t = 0; t < delta_start; t++) {
    vars_lowerbound[t] = -1.0e19;
    vars_upperbound[t] = +1.0e19;
  }
  // boundaries for steering angle
  for (size_t t = delta_start; t < a_start; t++) {
    vars_lowerbound[t] = -25 * M_PI / 180;
    vars_upperbound[t] = +25 * M_PI / 180;
  }
  // boundaries for acceleration
  for (size_t t = a_start; t < n_vars; t++) {
    vars_lowerbound[t] = -1.0;
    vars_upperbound[t] = +1.0;
  }

  // lower and upper boundaries for constraints
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);

  // all are zero except the initial state indices

  for (size_t t = 0; t < n_constraints; t++) {
    constraints_lowerbound[t] = 0.0;
    constraints_upperbound[t] = 0.0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_upperbound[x_start] = x;

  constraints_lowerbound[y_start] = y;
  constraints_upperbound[y_start] = y;

  constraints_lowerbound[psi_start] = psi;
  constraints_upperbound[psi_start] = psi;

  constraints_lowerbound[v_start] = v;
  constraints_upperbound[v_start] = v;

  constraints_lowerbound[cte_start] = cte;
  constraints_upperbound[cte_start] = cte;

  constraints_lowerbound[epsi_start] = epsi;
  constraints_upperbound[epsi_start] = epsi;

  // instance which computes constraints
  FG_eval fg_eval(coeffs);

  // options
  std::string options;
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";

  // using instance of fg_eval to find the lowest cost trajectory
  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  //
  // Check some of the solution values
  //
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;
  return {solution.x[x_start + 1], solution.x[y_start + 1],
          solution.x[psi_start + 1], solution.x[v_start + 1],
          solution.x[cte_start + 1], solution.x[epsi_start + 1],
          solution.x[delta_start], solution.x[a_start]};
}

//
// Helper functions to fit and evaluate polynomials.
//

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(const Eigen::VectorXd& xvals, const Eigen::VectorXd& yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  MPC mpc;
  int iters = 50;

  Eigen::VectorXd ptsx(2);
  Eigen::VectorXd ptsy(2);
  ptsx << -100, 100;
  ptsy << -1, -1;

  // TODO: fit a polynomial to the above x and y coordinates
  auto coeffs = polyfit(ptsx, ptsy, 1);

  // NOTE: free feel to play around with these
  double x = -1;
  double y = 10;
  double psi = 0;
  double v = 10;
  // TODO: calculate the cross track error
  double cte = polyeval(coeffs, x) - y; // current_y - total_y
  // TODO: calculate the orientation error
  // psi_desired = atan(coeff of f'(x))
  double epsi = psi - atan(coeffs[1]); // psi - psi_reference

  Eigen::VectorXd state(6);
  state << x, y, psi, v, cte, epsi;

  std::vector<double> x_vals = {state[0]};
  std::vector<double> y_vals = {state[1]};
  std::vector<double> psi_vals = {state[2]};
  std::vector<double> v_vals = {state[3]};
  std::vector<double> cte_vals = {state[4]};
  std::vector<double> epsi_vals = {state[5]};
  std::vector<double> delta_vals = {};
  std::vector<double> a_vals = {};

  for (size_t i = 0; i < iters; i++) {
    std::cout << "Iteration " << i << std::endl;

    auto vars = mpc.Solve(state, coeffs);

    x_vals.push_back(vars[0]);
    y_vals.push_back(vars[1]);
    psi_vals.push_back(vars[2]);
    v_vals.push_back(vars[3]);
    cte_vals.push_back(vars[4]);
    epsi_vals.push_back(vars[5]);

    delta_vals.push_back(vars[6]);
    a_vals.push_back(vars[7]);

    state << vars[0], vars[1], vars[2], vars[3], vars[4], vars[5];
    std::cout << "x = " << vars[0] << std::endl;
    std::cout << "y = " << vars[1] << std::endl;
    std::cout << "psi = " << vars[2] << std::endl;
    std::cout << "v = " << vars[3] << std::endl;
    std::cout << "cte = " << vars[4] << std::endl;
    std::cout << "epsi = " << vars[5] << std::endl;
    std::cout << "delta = " << vars[6] << std::endl;
    std::cout << "a = " << vars[7] << std::endl;
    std::cout << std::endl;
  }

  // Plot values
  // NOTE: feel free to play around with this.
  // It's useful for debugging!
  plt::subplot(3, 1, 1);
  plt::title("CTE");
  plt::plot(cte_vals);
  plt::subplot(3, 1, 2);
  plt::title("Delta (Radians)");
  plt::plot(delta_vals);
  plt::subplot(3, 1, 3);
  plt::title("Velocity");
  plt::plot(v_vals);

  plt::show();
}
