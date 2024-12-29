#include"data.h"
#include"Kmeans.h"
#include<Python.h>
#include"matplotlibcpp.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include "utils.cuh"

namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
    string path = argv[1];
    Data<double> df(path, true, false);
    //df.removeColumn(4);
    //df.removeColumn(0);
    MatrixXd datas = df.getMatrix();

    vector<int>clusters;
    vector<double>interias;
    for (int i = 1; i < 11; i++) {
        Kmeans km(i, 100, 0.000001);
        km.fit(datas);
        clusters.push_back(i);
        interias.push_back(km.get_inertia());
        cout << interias[i - 1] << "\n";
    }

    plt::plot(clusters, interias);
    plt::show();

    return 0;
}