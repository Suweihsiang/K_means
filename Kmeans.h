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
	Kmeans();											//constructor
	Kmeans(int n_clusters, int max_iters, double tol);	//constructor
	~Kmeans();											//destructor
	void fit(MatrixXd& datas);
	vector<VectorXd> get_centroids() const;
	vector<int> get_labels() const;
	double get_inertia() const;
private:
	int n_clusters = 2;									//number of clusters
	int max_iters = 300;
	double tol = 0.0001;								//error
	double inertia;										//used to measures how well a dataset was clustered by K-Means
	vector<VectorXd>centroids;
	vector<int>labels;
	double dist(VectorXd& a, VectorXd& b);				//calculate distance between points
	int nearest_centroid(VectorXd& x);					//find nearest centroid
	inline void init_centroids(MatrixXd datas);			//kmeans++
	void calc_inertia(vector<pair<VectorXd, int>>&data_labels, vector<VectorXd>&centroids);
};


#endif