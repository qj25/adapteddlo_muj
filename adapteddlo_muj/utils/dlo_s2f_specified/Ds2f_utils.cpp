#include "Ds2f_utils.h"
#include "Ds2f_obj.h"
#include <string>
#include <iostream>
#include "Eigen/Dense"

const Eigen::VectorXd Ds2fUtils::solveUpperTriangular(const Eigen::MatrixXd &A, const Eigen::VectorXd &B)
{
    // Function to solve Ax = B when A is upper triangular
    int n = A.rows();
    Eigen::VectorXd x(n);
    
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        if (A(i, i) == 0) {
            throw std::runtime_error("Matrix is singular and cannot be solved.");
        }
        x(i) = B(i);
        for (int j = i + 1; j < n; ++j) {
            x(i) -= A(i, j) * x(j);
        }
        x(i) /= A(i, i);
    }
    return x;
}

const Eigen::VectorXd Ds2fUtils::solveUTBig(const Eigen::MatrixXd &A, const Eigen::VectorXd &C)
{
    // Function to solve Ax = B when A is upper triangular 
    // and when UTMatrix is stairs (blocks of 3x3) rather than slope (1x1)
    int n = A.rows()/3;
    Eigen::VectorXd x_sol(n);
    Eigen::Vector3d rhsTmp;
    
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        // std::cout << A.block(i*3,i*3,3,3).determinant() << std::endl;
        if (A.block(i*3,i*3,3,3).determinant() == 0) {
            throw std::runtime_error("Matrix is singular and cannot be solved.");
        }
        rhsTmp = C(Eigen::seqN(i*3,3));
        for (int j = i + 1; j < n; ++j) {
            rhsTmp -= A.block(i*3,j*3,3,3) * x_sol(Eigen::seqN(j*3,3));
        }
        // Solve 3x3 lu here
        // at each section pieces, coordinates for torque is 3x1
        // therefore a need to solve a 3x3 simult linear eqn
        x_sol(Eigen::seqN(i*3,3)) = A.block(i*3,i*3,3,3).lu().solve(rhsTmp);
        // x(i) /= A(i, i);
    }
    return x_sol;
}

const Eigen::VectorXd Ds2fUtils::solveUTXBig(const Eigen::MatrixXd &A, const Eigen::VectorXd &C)
{
    // Function to solve Ax = B when A is upper triangular 
    // and when UTMatrix is stairs (blocks of 3x3) rather than slope (1x1)
    int n = A.rows()/(3*2);
    Eigen::VectorXd x_sol(n*3);
    Eigen::Matrix3d A1;
    Eigen::Matrix3d A2;
    Eigen::Vector3d C1;
    Eigen::Vector3d C2;
    int idx;
    x_sol.setZero();
    bool raiseErrsUtils = false;
    // std::cout << A << std::endl;
    // std::cout << C << std::endl;
    
    // Back substitution
    for (int i = n - 1; i > 0; --i) {
        /*
        condition 1 already fulfilled (earlier):
            - linear independence of A1 and A2
        condition 2 already fulfilled (quasistatic assumption):
            - geometric condition A perpendicular to C 
            if (std::abs(A1.dot(C1)) > 1e-5) {
                return false
            }
        */
        // forming As and Cs
        idx = i*2*3;
        A1 = A.block(idx,i*3,3,3);
        A2 = A.block(idx+3,i*3,3,3);
        C1 = C(Eigen::seqN(idx,3));
        C2 = C(Eigen::seqN(idx+3,3));
        for (int j = i + 1; j < n; ++j) {
            // std::cout << 'j' << std::endl;
            // std::cout << j << std::endl;
            C1 -= A.block(idx,j*3,3,3) * x_sol(Eigen::seqN(j*3,3));
            C2 -= A.block(idx+3,j*3,3,3) * x_sol(Eigen::seqN(j*3,3));
        }

        // check condition 3: consistency condition
        double consist_val = std::abs((getVecfromSkewSym(A1).dot(C2) + getVecfromSkewSym(A2).dot(C1)));

        std::cout << 'c' << std::endl;
        std::cout << consist_val << std::endl;
        // std::cout << getVecfromSkewSym(A1) << std::endl;
        // std::cout << getVecfromSkewSym(A2) << std::endl;
        // std::cout << getVecfromSkewSym(A1).dot(C2) << std::endl;
        // std::cout << getVecfromSkewSym(A2).dot(C1) << std::endl;
        if (consist_val > 1e-5) {
            // Condition D
            // for a case with 3 unknowns and 4 eqns,
            // formula is not consistent. solutions contradict each other
            std::cerr << "Error: consistency check failed!" << std::endl;
            if (raiseErrsUtils) {
                throw std::runtime_error(
                    "Consistency checks failed.. least squares might be able to solve this."
                );    
            }
        }

        // // Solve 3x3 lu here (through least squares)
        // std::cout << 'a' << std::endl;
        // std::cout << A << std::endl;
        // std::cout << 'b' << std::endl;
        // std::cout << A1 << std::endl;
        // std::cout << A2 << std::endl;
        // std::cout << C1 << std::endl;
        // std::cout << C2 << std::endl;
        Eigen::Matrix<double, 6, 3> A_stacked;
        A_stacked.topRows<3>() = A1;
        A_stacked.bottomRows<3>() = A2;

        Eigen::Matrix<double, 6, 1> C_stacked;
        C_stacked.head<3>() = C1;
        C_stacked.tail<3>() = C2;

        x_sol(Eigen::seqN(i*3,3)) = A_stacked.colPivHouseholderQr().solve(C_stacked);
    }
    // solving for sum(F) = 0
    for (int i = 0; i < 3; i++) {
        double Csingle = C(i);
        for (int j = 1; j < n; ++j) {
            Csingle -= A(i,j*3+i) * x_sol(j*3+i);
            // Csingle -= A.block(i,j*3,1,3).dot(x_sol(Eigen::seqN(j*3,3)));
        }
        x_sol(i) = Csingle;
    }
    // std::cout << 'b' << std::endl;
    // std::cout << x_sol << std::endl;
    return x_sol;
}

bool Ds2fUtils::checkLinearCorrelation(const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2)
{
    // returns true if vectors are linearly correlated (i.e., parallel to each other)
    std::cerr << "Error: linear correlation check failed!" << std::endl;
    return ((vec1.cross(vec2)).norm() < 1e-5);
}


// old utils
const Eigen::Vector3d Ds2fUtils::rotateVector3(const Eigen::Vector3d &v, const Eigen::Vector3d &u, const double a)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d v_res;
    Eigen::Vector3d u_norm = u / u.norm();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                R(i,j) = (
                    cos(a)
                    + (pow(u_norm(i),2.) * (1. - cos(a)))
                );
            }
            else {
                double ss = 1.;
                if (i < j) {ss *= -1.;}
                if (((i+1) * (j+1)) % 2 != 0) {ss *= -1.;}
                R(i,j) = (
                    u_norm(i) * u_norm(j) * (1. - cos(a))
                    + ss * u(3-(i+j)) * sin(a)
                );
            }
        }
    }
    v_res = R*v;
    return v_res;
}

const double Ds2fUtils::calculateAngleBetween(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
{
    double cos_ab, ab;
    cos_ab = v1.dot(v2) / (v1.norm() * v2.norm());
    if (cos_ab > 1) {return 0;}
    ab = acos(cos_ab);
    if (std::isnan(ab)) {
        std::string str_error = "Error in Ds2fUtils::calculateAngleBetween";
        std::cout << str_error << std::endl;
        std::cout << v1 << std::endl;
        std::cout << v2 << std::endl;
    }
    return ab;
}

const double Ds2fUtils::calculateAngleBetween2(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v_anchor)
{
    double e_tol, sab, n_div, cab, ab;
    e_tol = 1e-3;
    sab = 0.;
    n_div = 0.;

    Eigen::Vector3d cross12;
    if ((v1-v2).norm() < e_tol) {return 0;}
    cross12 = v1.cross(v2);
    for (int i = 0; i < 3; i++) {
        if (std::abs(v_anchor(i)) < e_tol) {continue;}
        sab += (
            cross12(i)
            / (v1.norm() * v2.norm())
            / v_anchor(i)
        );
        n_div += 1.;
    }
    sab /= n_div;
    cab = (
        v1.dot(v2)
        / (v1.norm() * v2.norm())
    );
    ab = atan2(sab, cab);
    return ab;
}

const Eigen::Matrix3d Ds2fUtils::createSkewSym(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d skew_sym_v;
    skew_sym_v << 0., -v[2], v[1],
        v[2], 0., -v[0],
        -v[1], v[0], 0.;
    return skew_sym_v;
}

const Eigen::Vector3d Ds2fUtils::getVecfromSkewSym(const Eigen::Matrix3d  &v_ss)
{
    Eigen::Vector3d v_out;
    v_out << v_ss(2,1), v_ss(0,2), v_ss(1,0); 
    return v_out;
}

const Eigen::Vector4d Ds2fUtils::inverseQuat(const Eigen::Vector4d &quat)
{
    Eigen::Vector4d quat_inv;
    quat_inv = - quat;
    quat_inv(0) = - quat_inv(0);
    quat_inv = quat_inv / quat_inv.dot(quat_inv);
    return quat_inv;
}

const Eigen::Vector3d Ds2fUtils::rotVecQuat(const Eigen::Vector3d &vec, const Eigen::Vector4d &quat)
{
    Eigen::Vector3d res;
    Eigen::Vector3d tmp;
    // zero vec: zero res
    if (vec[0] == 0 && vec[1] == 0 && vec[2] == 0) {
        res << 0., 0., 0.;
    }

    // null quat: copy vec
    else if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
        res = vec;
    }

    // regular processing
    else {
        // tmp = q_w * v + cross(q_xyz, v)
        tmp <<  quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1],
            quat[0]*vec[1] + quat[3]*vec[0] - quat[1]*vec[2],
            quat[0]*vec[2] + quat[1]*vec[1] - quat[2]*vec[0];

        // res = v + 2 * cross(q_xyz, t)
        res[0] = vec[0] + 2 * (quat[2]*tmp[2] - quat[3]*tmp[1]);
        res[1] = vec[1] + 2 * (quat[3]*tmp[0] - quat[1]*tmp[2]);
        res[2] = vec[2] + 2 * (quat[1]*tmp[1] - quat[2]*tmp[0]);
    }
    return res;
}

// int main()
// {

// }
