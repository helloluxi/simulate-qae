#include <iostream>
#include <boost/math/special_functions.hpp>
#include <boost/math/distributions/binomial.hpp>

using namespace boost::math;
using namespace std;

namespace iqae {
    
constexpr double PI = 3.14159265358979323846;

double pow2(double d) {
    return d * d;
}

void ClopperPearson(int t, int N, double alpha, double& aMin, double& aMax) {
    aMin = t == 0 ? 0.0 : ibeta_inv(t, N - t + 1, alpha / 2);
    aMax = t == N ? 1.0 : ibeta_inv(t + 1, N - t, 1 - alpha / 2);
}

double iqae_getLMax(int N, double alpha2T){
    // return asin(pow(2.0 / R * -log(alpha2T), 0.25));
    double LMax = 0;
    for (int t = 0; t <= N; t++)
    {
        double thetaMin = 0, thetaMax = 0;
        ClopperPearson(t, N, alpha2T, thetaMin, thetaMax);
        double h = 0.5 * (acos(1 - 2 * thetaMax) - acos(1 - 2 * thetaMin));
        if(h > LMax) LMax = h;
    }
    return LMax;
}

int iqae_findNextK(int k, double thetaMin, double thetaMax, double r) {
    int KMax = (int)floor(PI / (thetaMax - thetaMin));
    int K = KMax - (KMax - 2) % 4;
    while (K >= r * (4 * k + 2)) {
        if ((int)floor(K * thetaMin / PI + 1e-6) + 1 == (int)ceil(K * thetaMax / PI - 1e-6)) {
            return (K - 2) / 4;
        }
        K -= 4;
    }
    return k;
}

void simulate_iqae(double* src, double* res,
double epsilon, double alpha, int R, int nAll,
double& runCount, double& CRLB){
    int T = (int)ceil(log2(PI / (8 * epsilon)));
    double alpha2T = alpha / (2 * T);
    //double LMax = asin(pow(2.0 / R * -log(alpha2T), 0.25));
    double LMax = iqae_getLMax(R, alpha2T);

    double RunCount_LocalSum = 0;
    double CRLB_LocalSum = 0;
    for (int dataIdx = 0; dataIdx < nAll; dataIdx++)
    {
        double f = src[dataIdx], angle = asin(sqrt(f));
        int k = 0, lastK = 0, cumOneCount = 0, cumShotCount = 0;
        int sumRM = 0, sumRM2 = 0;
        double thetaMin = 0, thetaMax = PI * 0.5;
        while (thetaMax - thetaMin > 2 * epsilon) {
            // Find Parameter
            lastK = k;
            k = iqae_findNextK(k, thetaMin, thetaMax, 2.0);
            if (lastK != k) {
                cumOneCount = 0;
                cumShotCount = 0;
            }
            int K = 4 * k + 2;
            int nShot = K > (int)ceil(LMax / epsilon) ? (int)ceil(R * LMax / (epsilon * K * 10)) : R;
            // if (dataIdx == 0) cout << (2 * k + 1) << ',' << nShot << ';' << endl;
            sumRM += nShot * (2 * k + 1);
            sumRM2 += nShot * (2 * k + 1) * (2 * k + 1);

            // Simulation
            cumShotCount += nShot;
            double pr = 0.5 * (1 - cos(angle * (k * 2 + 1) * 2));
            for (int shot = 0; shot < nShot; shot++)
                cumOneCount += (double)rand() / RAND_MAX < pr;
            
            // Processing
            // double epsilonA = sqrt(-log(alpha2T) / (2 * cumShotCount));
            // double aMin = max(0.0, (double)cumOneCount / cumShotCount - epsilonA);
            // double aMax = min(1.0, (double)cumOneCount / cumShotCount + epsilonA);
            double aMin = 0, aMax = 0;
            ClopperPearson(cumOneCount, cumShotCount, alpha2T, aMin, aMax);

            int octave = (int)floor(thetaMin * K / PI + 1e-6);
            if (octave % 2 == 0) {
                thetaMin = (octave * PI + acos(1 - 2 * aMin)) / K;
                thetaMax = (octave * PI + acos(1 - 2 * aMax)) / K;
            }
            else {
                thetaMin = ((octave + 1) * PI - acos(1 - 2 * aMax)) / K;
                thetaMax = ((octave + 1) * PI - acos(1 - 2 * aMin)) / K;
            }
        }
        res[dataIdx] = (pow2(sin(thetaMin)) + pow2(sin(thetaMax))) * 0.5;
        RunCount_LocalSum += sumRM / (double)nAll;
        CRLB_LocalSum += pow(sumRM2, -0.5) / (double)nAll;
    }
    runCount = RunCount_LocalSum;
    CRLB = CRLB_LocalSum;
}

double chebae_getMaxErrorCP(int R, double alphaT){
    double errorMax = 0;
    for(int t = 0; t <= R; ++t){
        double aMin = 0, aMax = 0;
        ClopperPearson(t, R, alphaT, aMin, aMax);
        double h = (aMax - aMin) * 0.5;
        if(h > errorMax) errorMax = h;
    }
    return errorMax;
}

double chebae_findNextD(int d, double aMin, double aMax, double r=2.0) {
    double thetaMin = acos(aMax), thetaMax = acos(aMin);
    int dNew = (int)floor(PI / 2 / (thetaMax - thetaMin));
    while (dNew >= d * r && (int)floor(2 * dNew * thetaMin / PI + 1e-6) != (int)floor(2 * dNew * thetaMax / PI - 1e-6))
        --dNew;
    return dNew >= d * r ? dNew : d;
}

void simulate_chebae(double* src, double* res,
double epsilon, double alpha, int R, int nAll,
double& runCount, double& CRLB){
    int T = (int)ceil(log2(0.5 / epsilon));
    double alphaT = alpha / T;
    double errorMax = chebae_getMaxErrorCP(R, alphaT);

    for (int dataIdx = 0; dataIdx < nAll; dataIdx++)
    {
        double RunCount_LocalSum = 0;
        double CRLB_LocalSum = 0;
        double f = src[dataIdx], angle = asin(sqrt(f));
        int d = 1, lastD = 0, cumOneCount = 0, cumShotCount = 0;
        int sumRM = 0, sumRM2 = 0;
        double aMin = 0, aMax = 1, aMid = 0.5;
        while (aMax - aMin > 2 * epsilon) {
            // Find Parameter
            lastD = d;
            d = chebae_findNextD(d, aMin, aMax);
            if (lastD != d) {
                cumOneCount = 0;
                cumShotCount = 0;
            }
            double gap = pow2(cos(d * acos(aMax))) - pow2(cos(d * acos(aMin)));
            int nShot = (aMax - aMin) / gap * errorMax < 8.0 * epsilon ? 1 : R; // nu = 8.0
            // if (dataIdx == 0) cout << d << ',' << nShot << ';' << endl;
            sumRM += nShot * d;
            sumRM2 += nShot * d * d;

            // Simulation
            cumShotCount += nShot;
            double pr = 0.5 * (1 - cos(angle * d * 2));
            for (int shot = 0; shot < nShot; shot++)
                cumOneCount += (double)rand() / RAND_MAX < pr;
            
            // Processing
            double pMin = 0, pMax = 0;
            ClopperPearson(cumOneCount, cumShotCount, alphaT, pMin, pMax);

            int octave = (int)floor(acos(aMid) * 2 * d / PI);
            double aMinStar, aMaxStar;
            if (octave % 2 == 0) {
                aMinStar = cos((octave >> 1) * PI / d + acos(2 * pMin - 1) / (2 * d));
                aMaxStar = cos((octave >> 1) * PI / d + acos(2 * pMax - 1) / (2 * d));
            }
            else {
                aMinStar = cos(((octave >> 1) + 1) * PI / d - acos(2 * pMin - 1) / (2 * d));
                aMaxStar = cos(((octave >> 1) + 1) * PI / d - acos(2 * pMax - 1) / (2 * d));
            }
            if (aMinStar > aMaxStar) {
                double tmp = aMinStar; aMinStar = aMaxStar; aMaxStar = tmp;
            }
            /*if (dataIdx == 0) {
                cout << "A=" << f << endl;
                cout << "D=" << d << ", " << cumOneCount << "/" << cumShotCount << endl;
                cout << "P in [" << pMin << ',' << pMax << ']' << endl;
                cout << "A* in [" << aMinStar << ',' << aMaxStar << ']' << endl;
                cout << "A in [" << aMin << ',' << aMax << ']' << endl;
                cout << endl << endl;
            }*/
            aMin = max(aMin, aMinStar - 1e-15);
            aMax = min(aMax, aMaxStar + 1e-15);
            aMid = 0.5 * (aMin + aMax);
        }
        res[dataIdx] = aMid * aMid;
        //if (dataIdx == 0) cout << "\n*************************\n" << src[dataIdx] << "->" << res[dataIdx] << "\n*************************\r\n\r\n\r\n\r\n";
        RunCount_LocalSum += (double)sumRM / (double)nAll;
        CRLB_LocalSum += pow((double)sumRM2, -0.5) / (double)nAll;

        // For later parallel impl.
        runCount += RunCount_LocalSum;
        CRLB += CRLB_LocalSum;
    }
}

}

double randf() {
    return (double)rand() / RAND_MAX;
}

void main(int argc, char** argv) {
    int nAll = 4000, readParamIdx = 0;
    double* data = new double[nAll], * estimate = new double[nAll];
    for (int i = 0; i < nAll; i++)
    {
        data[i] = randf();
    }

    string method = argv[++readParamIdx];
    double runCount = 0;
    double CRLB = 0;
    if (method == "iqae" || method == "chebae") {
        double epsilon = atof(argv[++readParamIdx]), alpha = atof(argv[++readParamIdx]);
        int R = atoi(argv[++readParamIdx]);
        if(method == "iqae"){
            iqae::simulate_iqae(data, estimate, epsilon, alpha, R, nAll, runCount, CRLB);
            // call_iqae_kernel(data, estimate, epsilon, alpha, R, nAll, runCount, CRLB);
        }
        else{
            iqae::simulate_chebae(data, estimate, epsilon, alpha, R, nAll, runCount, CRLB);
            // call_chebae_kernel(data, estimate, epsilon, alpha, R, nAll, runCount, CRLB);
        }
    }
    else {
        cerr << "Unknown method: " << method << endl;
        return;
    }

    // Calculate absolute errors
    vector<double> absErrors(nAll);
    double errorSquareSum = 0;
    for (int i = 0; i < nAll; i++)
    {
        double bias = estimate[i] - data[i];
        absErrors[i] = abs(bias);
        errorSquareSum += bias * bias;
    }
    // Sort absolute errors
    sort(absErrors.begin(), absErrors.end());
    // 5% top error
    int idx_5p = (int)(nAll * 0.05);
    double error_5p = absErrors[idx_5p];
    cout << runCount
        << ',' << sqrt(errorSquareSum / nAll)
        << ',' << error_5p
        << endl;
    
    delete[] data, estimate;
}