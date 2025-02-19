Kmeans.h  Kmeans.cpp  
使用C++實現K-means Clustering  
===========================================================================  
  
Parameters
===========================================================================  
**n_clusters：int**  
由幾個質心進行聚類，預設為2  
  
**max_iters：int**  
最大迭代次數，預設為300  

**tol：double**  
新舊質心之距離平方和若小於tol，則視為已收斂並停止迭代，預設為0.0001  
  
Attributes  
===========================================================================  
**inertia：double**  
每個資料點到其聚合之質心之距離平方和  
  
**centroids：vector\<VectorXd\>**  
K個質心之集合  
  
**labels：vector\<int\>**  
每個資料點其對應之質心標籤  
  
Methods  
===========================================================================  
===========================================================================  
**Kmeans()**  
Kmeans建構式  
  
===========================================================================  
**Kmeans(int n_clusters, int max_iters, double tol)**  
**Parameters:**  
n_clusters：int  
   *由幾個質心進行聚類，預設為2*  
max_iters：int  
   *最大迭代次數，預設為300*  
tol：double  
   *新舊質心之距離平方合若小於tol，則視為已收斂並停止迭代，預設為0.0001*  
  
===========================================================================  
**~Kmeans()**  
Kmeans解構式  
    
===========================================================================  
**void fit(MatrixXd &datas)**  
進行K-means演算法之配適    
**parameters:**  
datas：MatrixXd  
   *訓練資料集*  
  
===========================================================================  
**vector\<VectorXd\> get_centroids()**  
**return:**  
centroids：vector\<VectorXd\>  
   *K個質心*  
  
===========================================================================  
**vector\<int\> get_labels()**  
**return:**  
labels：vector\<int\>  
   *每個資料點對應之質心標籤*  
  
===========================================================================  
**double get_inertia()**  
**return:**  
inertia：double  
   *每個資料點到其聚合之質心之距離平方和*  
  
===========================================================================  
**double dist(VectorXd &a, VectorXd &b)**  
**Parameters:**  
a：VectorXd  
   *資料點a*  
b：VectorXd  
   *資料點b*  
**return:**  
distance：double  
   *兩點之距離平方*  
  
===========================================================================  
**int nearest_centroid(VectorXd &x)**  
**Parameters:**  
x：VectorXd  
   *資料點*  
**return:**  
label：int  
   *距離資料點最近之質心標籤*  

===========================================================================  
**void init_centroids(MatrixXd datas)**  
使用K-means++方式初始化K個質心  
**Parameters:**  
datas：MatrixXd  
   *資料集*  
  
===========================================================================  
**void calc_inertia(vector\<pair\<VectorXd,int\>\> &data_labels, vector\<VectorXd\> &centroids)**  
計算每個資料點到其聚合之質心之距離平方和  
**Parameters:**  
data_labels：vector\<pair\<VectorXd,int\>\>  
   *各個資料點與對應之質心標籤集合*  
centroids：vector\<VectorXd\>  
   *K個質心*  

