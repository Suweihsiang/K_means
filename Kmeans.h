#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include<vector>
#include<algorithm>
#include<eigen3/Eigen/Dense>
#include<cmath>
#include<random>
#include<ctime>

using std::vector;
using std::pair;
using std::make_pair;
using std::sort;
using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Kmeans {
public:
	Kmeans();
	Kmeans(int n_clusters, int max_iters, double tol);
	~Kmeans();
	void fit(MatrixXd& datas);
	vector<VectorXd> get_centroids() const;
	vector<int> get_labels() const;
	double get_inertia() const;
private:
	int n_clusters = 2;
	int max_iters = 300;
	double tol = 0.0001;
	double inertia;
	vector<VectorXd>centroids;
	vector<int>labels;
	double dist(VectorXd& a, VectorXd& b);
	int nearest_centroid(VectorXd& x);
	inline void init_centroids(MatrixXd datas);
	void calc_inertia(vector<pair<VectorXd, int>>&data_labels, vector<VectorXd>&centroids);
};


#endif