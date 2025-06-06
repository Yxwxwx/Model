#include "src/integral/integral.hpp"
#include <iostream>

int main()
{
    std::vector<int> atm {
        8,
        20,
        1,
        23,
        0,
        0,
        1,
        24,
        1,
        27,
        0,
        0,
        1,
        28,
        1,
        31,
        0,
        0,
    };

    int natm = atm.size() / ATM_SLOTS;
    std::vector<int> bas {
        0,
        0,
        3,
        1,
        0,
        38,
        41,
        0,
        0,
        0,
        3,
        1,
        0,
        44,
        47,
        0,
        0,
        1,
        3,
        1,
        0,
        50,
        53,
        0,
        1,
        0,
        3,
        1,
        0,
        32,
        35,
        0,
        2,
        0,
        3,
        1,
        0,
        32,
        35,
        0,
    };

    int nbas = bas.size() / BAS_SLOTS;
    std::vector<double> env {
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        -1.43052268,
        1.10926924,
        0.,
        0.,
        1.43052268,
        1.10926924,
        0.,
        3.42525091,
        0.62391373,
        0.1688554,
        0.98170675,
        0.94946401,
        0.29590645,
        130.70932,
        23.808861,
        6.4436083,
        15.07274649,
        14.57770167,
        4.54323359,
        5.0331513,
        1.1695961,
        0.380389,
        -0.848697,
        1.13520079,
        0.85675304,
        5.0331513,
        1.1695961,
        0.380389,
        3.42906571,
        2.15628856,
        0.34159239,
    };
    int nao = 7;
    Eigen::MatrixXd dm(nao, nao);
    dm << 2.10629383e+00, -4.46244962e-01, 6.03417177e-18, 8.00586877e-17, 1.08585498e-01, -2.82985902e-02, -2.82985902e-02, -4.46244962e-01, 1.96824054e+00, 9.99194828e-17, -3.28189511e-16, -6.17517877e-01, -3.45503140e-02, -3.45503140e-02, 6.03417177e-18, 9.99194828e-17, 2.00000000e+00, 2.67560756e-16, 1.24064249e-16, -2.78393580e-16, -2.60407226e-16, 8.00586877e-17, -3.28189511e-16, 2.67560756e-16, 7.35831324e-01, 8.00559119e-16, -5.39971365e-01, 5.39971365e-01, 1.08585498e-01, -6.17517877e-01, 1.24064249e-16, 8.00559119e-16, 1.23872996e+00, 4.72987228e-01, 4.72987228e-01, -2.82985902e-02, -3.45503140e-02, -2.78393580e-16, -5.39971365e-01, 4.72987228e-01, 6.01482721e-01, -1.91006170e-01, -2.82985902e-02, -3.45503140e-02, -2.60407226e-16, 5.39971365e-01, 4.72987228e-01, -1.91006170e-01, 6.01482721e-01;
    YXTensor::print_tensor(integral::get_bp_hso2e(dm, nao, atm.data(), natm, bas.data(), nbas, env.data()));
    return 0;
}