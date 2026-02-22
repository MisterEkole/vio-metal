#pragma once

#include <Eigen/Dense>
#include <vector>
#include<ceres/ceres.h>


namespace vio {

// Simplified marginalization prior for Phase 1.
// Stores the linearized Hessian and gradient from marginalized states
// as a prior factor on the remaining states.
class MarginalizationInfo {
public:
    MarginalizationInfo() = default;

    // Set the prior from a Hessian block and gradient
    void setPrior(const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
                  const Eigen::VectorXd& x0);

    bool hasPrior() const { return has_prior_; }

    // Evaluate the prior residual: r = sqrt_info * (x - x0)
    // Returns residual dimension
    int evaluate(const double* const* parameters, int num_params,
                 double* residuals) const;

    const Eigen::MatrixXd& sqrtInfo() const { return sqrt_info_; }
    const Eigen::VectorXd& linearizationPoint() const { return x0_; }
    int dimension() const { return dim_; }

private:
    bool has_prior_ = false;
    int dim_ = 0;
    Eigen::MatrixXd sqrt_info_;
    Eigen::VectorXd x0_;     // Linearization point
    Eigen::VectorXd b0_;     // Gradient at linearization
};

// Prior cost function for Ceres
// Applies the marginalization prior to the oldest remaining keyframe state
class MarginalizationFactor : public ceres::CostFunction {
public:
    explicit MarginalizationFactor(const MarginalizationInfo& info);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;

private:
    Eigen::MatrixXd sqrt_info_;
    Eigen::VectorXd x0_;
    int dim_;
};

} // namespace vio
