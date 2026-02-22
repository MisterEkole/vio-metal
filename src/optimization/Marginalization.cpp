#include "Marginalization.h"
#include <ceres/ceres.h>
#include <Eigen/Eigenvalues>
#include <iostream>

namespace vio {

void MarginalizationInfo::setPrior(const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
                                    const Eigen::VectorXd& x0) {
    dim_ = static_cast<int>(H.rows());
    x0_ = x0;
    b0_ = b;

    // Compute sqrt of information matrix via eigendecomposition
    // H = V * D * V^T, sqrt_info = D^{1/2} * V^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    // Threshold small eigenvalues for numerical stability
    Eigen::VectorXd sqrt_eigenvalues(dim_);
    for (int i = 0; i < dim_; i++) {
        sqrt_eigenvalues(i) = (eigenvalues(i) > 1e-8) ? std::sqrt(eigenvalues(i)) : 0.0;
    }

    sqrt_info_ = sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose();
    has_prior_ = true;
}

int MarginalizationInfo::evaluate(const double* const* parameters, int num_params,
                                   double* residuals) const {
    if (!has_prior_) return 0;

    // Collect current state into a single vector
    // For simplicity, assume parameters are contiguous state blocks
    Eigen::VectorXd x(dim_);
    int offset = 0;
    for (int i = 0; i < num_params && offset < dim_; i++) {
        // This is simplified — in practice you'd know each block size
    }

    return dim_;
}

// ============================================================
// MarginalizationFactor
// ============================================================

MarginalizationFactor::MarginalizationFactor(const MarginalizationInfo& info)
    : sqrt_info_(info.sqrtInfo()), x0_(info.linearizationPoint()), dim_(info.dimension())
{
    // For Phase 1: apply prior to the oldest remaining keyframe
    // Parameter block: [p(3), q(4), v_bg_ba(9)] = 16 dims
    // But tangent space = 15 (quaternion tangent = 3)
    set_num_residuals(dim_);
    // Parameter blocks will be set by the optimizer
}

bool MarginalizationFactor::Evaluate(
    double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
    // Simplified Phase 1: just penalize deviation from linearization point
    // Full implementation would handle the Schur complement properly
    Eigen::Map<Eigen::VectorXd> res(residuals, dim_);

    // Collect parameters into a delta vector
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(dim_);

    // For now, use position and velocity components
    if (dim_ >= 3) {
        Eigen::Map<const Eigen::Vector3d> p(parameters[0]);
        dx.head<3>() = p - x0_.head<3>();
    }

    res = sqrt_info_ * dx;

    // Let Ceres compute Jacobians numerically
    (void)jacobians;
    return true;
}

} 
