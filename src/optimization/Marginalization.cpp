#include "Marginalization.h"
#include <Eigen/Eigenvalues>

namespace vio {

void MarginalizationInfo::setPrior(const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
                                    const Eigen::VectorXd& x0) {
    dim_ = static_cast<int>(H.rows());
    x0_ = x0;

    Eigen::MatrixXd H_reg = H;
    H_reg.diagonal().array() += 1e-2;

    // H = V D V^T → sqrt_info = D^{1/2} V^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_reg);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    // Clamp eigenvalues: minimum for numerical stability, maximum to prevent
    // the prior from becoming overly stiff and dominating vision observations.
    // Without a max clamp, eigenvalues grow unboundedly through repeated
    // marginalization, making the prior fight with current observations.
    double max_eigenvalue = eigenvalues.maxCoeff();
    double min_allowed = max_eigenvalue * 1e-2;
    if (min_allowed < 1e-2) min_allowed = 1e-2;
    double max_allowed = 1e4;  // prevent prior from becoming too dominant

    Eigen::VectorXd sqrt_eigenvalues(dim_);
    Eigen::VectorXd inv_sqrt_eigenvalues(dim_);
    for (int i = 0; i < dim_; i++) {
        double ev = std::min(eigenvalues(i), max_allowed);
        if (ev > min_allowed) {
            sqrt_eigenvalues(i) = std::sqrt(ev);
            inv_sqrt_eigenvalues(i) = 1.0 / sqrt_eigenvalues(i);
        } else {
            sqrt_eigenvalues(i) = 0.0;
            inv_sqrt_eigenvalues(i) = 0.0;
        }
    }

    sqrt_info_ = sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose();
    // e0 captures the gradient at the linearization point
    e0_ = -(inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b).eval();

    has_prior_ = true;
}

} // namespace vio
