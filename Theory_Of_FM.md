# Theory of FM（Factorization Machines） Algorithm


## 1.Background


Click-through rate is a very important link in computational advertising and recommendation system. It is necessary to determine whether an
item is recommended or not according to CTR. The commonly used methods in the industry are artificial feature engineering + LR (logistic regression), GBDT(gradient boosting decision tree),[FM(factorization machine)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
and FFM(field-aware factorization machine) model. In recent years, there have been many improved methods based on FM, such as
[deepFM](https://www.ijcai.org/proceedings/2017/0239.pdf), FNN, PNN, [DCN](https://arxiv.org/abs/1708.05123),[xDeepFM](https://arxiv.org/abs/1803.05170). 


## 2.Principle

FM mainly solves the problem of feature combination under sparse data. And the complexity of its prediction is linear, which has good generality for continuous and discrete features. 

There follows an example.

![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/3.png)


The score is label. User id, movie id and scoring time are characteristics. Because the user id and the movie id are category

characteristics, they need to be converted into numeric features through one-heat coding.So that the sample data becomes sparse after

onehot encoding. 

There follows an example from data set S.

![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/1.png)

Example for sparse real valued feature vectors x that are created from the transactions of example 1. Every row represents a feature

vector x(i) with its corresponding target y(i). The first 4 columns (blue) represent indicator variables for the active user; the next 5

(red) indicator variables for the active item. The next 5 columns (yellow) hold additional implicit indicators (i.e. other movies the

user has rated). One feature (green) represents the time in months. The last 5 columns (brown) have indicators for the last movie the

user has rated before the active one. The rightmost column is the target  here the rating.



