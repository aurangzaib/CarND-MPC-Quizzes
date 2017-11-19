// In this quiz you'll implement the global kinematic model.
#include <math.h>
#include <iostream>
#include "Eigen-3.3/Eigen/Core"

//
// Helper functions
//
constexpr double pi() { return M_PI; }

double deg2rad(double x) { return x * pi() / 180; }

double rad2deg(double x) { return x * 180 / pi(); }

const double Lf = 2;

// TODO: Implement the global kinematic model.
// Return the next state.
//
// NOTE: state is [x, y, psi, v]
// NOTE: actuators is [delta, a]
Eigen::VectorXd globalKinematic(Eigen::VectorXd state,
                                Eigen::VectorXd actuators,
                                double dt) {
  // states & control inputs
  auto x = state[0], y = state[1], wheel_turn = state[2], velocity = state[3];
  // control inputs
  auto steering_angle = actuators[0], acceleration = actuators[1];

  // new state
  Eigen::VectorXd next_state(state.size());

  // state update using global kinematics equations
  next_state[0] = x + velocity * cos(wheel_turn) * dt; // x
  next_state[1] = y + velocity * sin(wheel_turn) * dt; // y
  next_state[2] = wheel_turn + velocity * ( steering_angle / Lf) * dt; // psi
  next_state[3] = velocity + acceleration * dt;

  return next_state;
}

int main() {
  // [x, y, psi, v]
  Eigen::VectorXd state(4);
  // [delta, v]
  Eigen::VectorXd actuators(2);
  // x, y, wheel turn, v
  state << 0, 0, deg2rad(45), 1;
  // steering angle, acceleration
  actuators << deg2rad(5), 1;
  // should be [0.212132, 0.212132, 0.798488, 1.3]
  auto next_state = globalKinematic(state, actuators, 0.3);
  std::cout << next_state << std::endl;
}