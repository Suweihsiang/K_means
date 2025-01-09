#include "Kmeans.h"

Kmeans::Kmeans(){}

Kmeans::Kmeans(int n_clusters, int max_iters, double tol) :n_clusters(n_clusters), max_iters(max_iters), tol(tol) {}

Kmeans::~Kmeans() {}

void Kmeans::fit(MatrixXd& datas) {
	init_centroids(datas);//kmeans++
	int iters = 0;
	labels.reserve(datas.rows());
	while (iters < max_iters) {
		labels.clear();
		vector<pair<VectorXd, int>>data_labels(datas.rows());
		for (int i = 0; i < datas.rows(); i++) {
			VectorXd data = datas.row(i);
			int label = nearest_centroid(data);
			labels.push_back(label);
			data_labels[i] = make_pair(data, label);
		}
		sort(data_labels.begin(), data_labels.end(), [](pair<VectorXd, int>& a, pair<VectorXd, int>& b) {return a.second < b.second; });
		calc_inertia(data_labels, centroids);
		vector<VectorXd>prev_centroids = centroids;
		centroids.clear();
		int j = 0;
		for (int i = 0; i < n_clusters; i++) {
			int count = 0;
			VectorXd new_centroid = VectorXd::Zero(datas.cols());
			while (j < data_labels.size() && data_labels[j].second == i) {
				new_centroid += data_labels[j++].first;
				count++;
			}
			if (count != 0) { centroids.push_back(new_centroid / count); }
			if (dist(centroids[i], prev_centroids[i]) < tol) { centroids = prev_centroids; return; }//early stop
		}
		iters++;
	}
}

vector<VectorXd> Kmeans::get_centroids() const {
	return centroids;
}

vector<int> Kmeans::get_labels() const {
	return labels;
}

double Kmeans::get_inertia() const {
	return inertia;
}

double Kmeans::dist(VectorXd& a, VectorXd& b) {
	return (a - b).transpose() * (a - b);
}

int Kmeans::nearest_centroid(VectorXd& a) {
	int nearest_idx = -1;
	double nearest_dist = DBL_MAX;
	for (int idx = 0; idx < centroids.size(); idx++) {
		double d = dist(a, centroids[idx]);
		if (d < nearest_dist) { nearest_idx = idx; nearest_dist = d; }
	}
	return nearest_idx;
}

inline void Kmeans::init_centroids(MatrixXd datas) {
	centroids.reserve(n_clusters);
	srand(time(NULL));
	int sel_row = rand()%datas.rows();//select first centroid randomly
	centroids.push_back(datas.row(sel_row));
	datas.block(sel_row, 0, datas.rows() - sel_row - 1, datas.cols()) = datas.block(sel_row + 1, 0, datas.rows() - sel_row - 1, datas.cols()).eval();
	for (int i = 1; i < n_clusters; i++) {
		double sum_dist = 0;
		vector<double>datas_dist(datas.rows(), DBL_MAX);
		for (int j = 0; j < datas.rows(); j++) {
			for (int k = 0; k < i; k++) {
				VectorXd data = datas.row(j);
				double data_dist = dist(data, centroids[k]);
				if (data_dist < datas_dist[j]) { datas_dist[j] = data_dist; }
			}
			sum_dist += datas_dist[j];
		}
		sum_dist *= (double)rand() / RAND_MAX;
		for (int idx = 0; idx < datas_dist.size(); idx++) {
			sum_dist -= datas_dist[idx];
			if (sum_dist < 0) { 
				centroids.push_back(datas.row(idx)); //get next centroid
				datas.block(idx, 0, datas.rows() - idx - 1, datas.cols()) = datas.block(idx + 1, 0, datas.rows() - idx - 1, datas.cols()).eval();
				break; 
			}
		}
	}
}

void Kmeans::calc_inertia(vector<pair<VectorXd, int>>& data_labels, vector<VectorXd>& centroids) {
	int j = 0;
	inertia = 0;
	for (int i = 0; i < centroids.size(); i++) {
		while (j < data_labels.size() && data_labels[j].second == i) {
			inertia += dist(data_labels[j++].first, centroids[i]);//sum of the distance between each point to their centroid
		}
	}
}