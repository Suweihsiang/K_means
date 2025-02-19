data.h  data.cpp  
使用C++實現Pandas DataFrame常用功能  
===========================================================================  
Parameters
===========================================================================  
**data_name：string**  
資料集名稱  
  
**row：int**  
資料集列數，預設為0  
  
**columns：int**  
資料集行數，預設為0  
  
**hasIndexRows：bool**  
是否有索引行  
  
**index_name：string**  
索引行之特徵名稱  
  
**features：vector\<string\>**  
資料集之各個特徵名稱  
  
**indexes：vector\<string\>**  
資料集之各個索引名稱  
  
**nullpos：vector\<vector\<int\>\>**  
空值之位置(列,行)  
  
**mat：Matrix\<T,Dynamic,Dynamic\>**  
資料集之數值矩陣，T可為int、float、double  
  
Methods  
===========================================================================  
===========================================================================  
**Data()**  
Data建構式  
  
===========================================================================  
**Data(unordered_map\<string,vector\<string\>\>m, bool IndexFirst,bool sortIndex = false)**  
**Parameters:**   
m：unordered_map\<string,vector\<string\>\>  
   *string為特徵，vector<string>各資料該特徵之值*  
IndexFirst：bool  
   *資料首行是否為索引行*  
sortIndex：bool  
   *是否以索引對資料集進行排序*  
  
===========================================================================  
**Data(string path,bool IndexFirst,bool isThousand,bool sortIndex = false)**  
**parameters:**  
path：string  
   *資料之絕對路徑*  
IndexFirst：bool  
   *資料首列是否為索引*  
isThousand：bool  
   *數值是否有以camma(,)符號分隔千單位*  
sortIndex：bool  
   *是否以索引對資料集進行排序*  
  
===========================================================================  
**~Data()**  
Data解構式  
  
===========================================================================  
**Data\<T\> Data\<T\>::operator\[\]\(vector\<string\> fts\)**  
從現有的資料集中選取其中幾項特徵  
**parameters:**  
features：vector\<string\>  
   *所欲選取之特徵行*  
**return：**  
data:Data\<T\>  
   *包含選取特徵之資料集，T可為int、float、double*  
  
===========================================================================  
**vector\<string\> split(string s, char dec)**  
**parameters:**  
s：string  
   *欲分割之字串*  
dec：char  
   *分割符號*  
**return：**  
strings:vector\<string\>  
   *分割後字串之集合*  
  
===========================================================================  
**void setDataname(string _data_name)**  
設定資料集之名稱    
**parameters:**  
_data_name：string  
   *資料集之名稱*  
  
===========================================================================  
**string getDataname()**  
**return:**  
data_name：string  
   *資料集之名稱*  
  
===========================================================================  
**int getRows()**  
**return:**  
row_num：int  
   *資料之列數*  
  
===========================================================================  
**int getColumns()**  
**return:**  
col_num：int  
   *資料之特徵數*  
  
===========================================================================  
**Matrix\<T,Dynamic,Dynamic\> getMatrix()**  
**return:**  
data_matric：Matrix\<T,Dynamic,Dynamic\>  
   *資料集之數值矩陣，T可為int、float、double*  
  
===========================================================================  
**vector\<string\> getFeatures()**  
**return:**  
features：vector\<string\>  
   *資料集之特徵*  
  
===========================================================================  
**vector\<string\> getIndexs()**  
**return:**  
indices：vector\<string\>  
   *資料集之索引*  
  
===========================================================================  
**vector\<string\> camma_remove(string data_string)**  
自帶有分隔符號之csv檔讀取出之資料為一字串如："1,000","2,000","3,000"，本函式欲將其分割為三個數值並移除千位camma符號如：1000 2000 3000  
**parameters:**  
data_string：string  
   *帶有千位分隔符號之資料數值*  
**return:**  
data：vector\<string\>  
   *去除千位分隔符號之資料數值*  
  
===========================================================================  
**void setIndex(string index)**  
**parameters:**  
index：string  
   *欲設為索引之特徵行*  
  
===========================================================================  
**void removeRow(int RowToRemove)**  
**parameters:**  
RowToRemove：int  
   *欲移除之某列*  
  
===========================================================================  
**void removeRow(string idx)**  
**parameters:**  
idx：string  
   *欲移除之某列索引名稱*  
  
===========================================================================  
**void removeRows(vector\<string\> idxs)**  
**parameters:**  
idxs：vector\<string\>  
   *欲移除之索引名稱*  
  
===========================================================================  
**void merge(Data\<T\> df2)**  
**parameters:**  
df2：Data\<T\>  
   *欲合併之資料集，T可為int、float、double*  
  
===========================================================================  
**void addRows(vector\<vector\<string\>\> rows)**  
**parameters:**  
rows：vector\<vector\<string\>\>  
   *欲增加之列其資料數值(首列或為索引)*  
  
===========================================================================  
**void addColumns(unordered_map\<string,vector\<string\>\>m)**  
**parameters:**  
m：unordered_map\<string,vector\<string\>\> 
   *各索引及其增加特徵行之數值*  
  
===========================================================================  
**void removeColumn(int ColToRemove)**  
**parameters:**  
ColToRemove：int 
   *欲移除之行數*  
  
===========================================================================  
**void removeColumns(vector\<string\> fts)**  
**parameters:**  
features：vector\<string\> 
   *欲移除之特徵行名稱*  
  
===========================================================================  
**void renameColumns(unordered_map<string, string>names)**  
**parameters:**  
names：unordered_map<string, string> 
   *欲改名之特徵其新舊名稱對照{舊名：新名}*  
  
===========================================================================  
**void sortbyIndex(bool ascending)**  
以索引值之順序進行排列  
**parameters:**  
ascending：bool  
   *是否從小到大排列*  
  
===========================================================================  
**void sortby(string feature,bool ascedning)**  
以選取之特徵行之數值順序進行排列  
**parameters:**  
feature：string  
   *選取之特徵行*  
ascending：bool  
   *是否從小到大排列*  
  
===========================================================================  
**Data\<T\> groupby(string feature, string operate)**  
以選取之特徵行期相同特徵者組為一群，並進行加總或平均  
**parameters:**  
feature：string  
   *選取之特徵行*  
operate：string  
   *"sum":組內數值加總*  
   *"mean":組內數值平均數*  
**return:**  
data：Data\<T\>  
   *以某特徵分群組後之資料集，同時以該特徵行為索引行，T可為int、float、double*  
  
===========================================================================  
**void dropna(int axis)**  
移除有空值之行或列  
**parameters:**  
axis：int  
   *axis = 0，有空值之列移除*  
   *axis = 1，有空值之行移除*  
  
===========================================================================  
**void print()**  
印出資料集  
  
===========================================================================  
**void to_csv(string path)**  
將資料集存為csv檔  
**parameters:**  
path：string  
   *存檔之絕對路徑*  
  
===========================================================================  
**void setMatrix(unordered_map\<string,vector\<string\>\>::iterator it)**  
依序將各特徵行下之數值寫入對應之矩陣行數中  
**parameters:**  
it：unordered_map\<string,vector\<string\>\>::iterator  
   *各項特徵及其對應之數值*  
  
===========================================================================  
**void setMatrix(int r, int c, vector\<string\> cols)**  
自矩陣之第r列第c行開始輸入數值  
**parameters:**  
r：int  
   *開始寫入數值之矩陣列數*  
c：int  
   *開始寫入數值之矩陣行數*  
cols：vector\<string\>  
   *欲寫入之數值*  