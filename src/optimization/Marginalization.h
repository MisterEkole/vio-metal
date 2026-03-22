#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace vio {

// Schur complement marginalization prior on kept variables.
// cost = 0.5 * || sqrt_info * dx + e0 ||^2
class MarginalizationInfo {
public:
    MarginalizationInfo() = default;

    // H: information matrix, b: gradient, x0: linearization point [p(3), q_xyzw(4), vba(9)]
    void setPrior(const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
                  const Eigen::VectorXd& x0);

    bool hasPrior() const { return has_prior_; }
    void resetPrior() { has_prior_ = false; }

    const Eigen::MatrixXd& sqrtInfo() const { return sqrt_info_; }
    const Eigen::VectorXd& linearizationPoint() const { return x0_; }
    const Eigen::VectorXd& residualOffset() const { return e0_; }
    int dimension() const { return dim_; }

private:
    bool has_prior_ = false;
    int dim_ = 0;
    Eigen::MatrixXd sqrt_info_;
    Eigen::VectorXd x0_;
    Eigen::VectorXd e0_;
};

// Prior factor for one keyframe state. Residual dim = 15 (tangent space).
struct MarginalizationFactor {
    MarginalizationFactor(const Eigen::MatrixXd& sqrt_info,
                          const Eigen::VectorXd& x0,
                          const Eigen::VectorXd& e0)
        : sqrt_info_(sqrt_info), x0_(x0), e0_(e0) {}

    template <typename T>
    bool operator()(const T* const p, const T* const q,
                    const T* const vba, T* residuals) const {
        Eigen::Map<Eigen::Matrix<T, 15, 1>> res(residuals);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos(p);
        Eigen::Map<const Eigen::Quaternion<T>> quat(q);
        Eigen::Map<const Eigen::Matrix<T, 9, 1>> vel_bias(vba);

        Eigen::Matrix<T, 3, 1> p0 = x0_.head<3>().cast<T>();
        Eigen::Quaternion<T> q0(T(x0_(6)), T(x0_(3)), T(x0_(4)), T(x0_(5)));
        q0.normalize();
        Eigen::Matrix<T, 9, 1> vba0 = x0_.tail<9>().cast<T>();

        res.template segment<3>(0) = pos - p0;
        Eigen::Quaternion<T> dq = q0.inverse() * quat;
        if (dq.w() < T(0)) { dq.coeffs() *= T(-1.0); }
        res.template segment<3>(3) = T(2.0) * dq.vec();

        res.template segment<9>(6) = vel_bias - vba0;
        res = sqrt_info_.cast<T>() * res + e0_.cast<T>();
        return true;
    }

    Eigen::MatrixXd sqrt_info_;
    Eigen::VectorXd x0_;
    Eigen::VectorXd e0_;
};

} // namespace vio
