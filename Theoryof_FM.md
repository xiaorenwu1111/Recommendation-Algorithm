# Theory of FM（Factorization Machines） Algorithm


## 1.Background


Click-through rate is a very important link in computational advertising and recommendation system. It is necessary to determine whether an item is recommended or not according to CTR. The commonly used methods in the industry are artificial feature engineering + LR (logistic regression), GBDT(gradient boosting decision tree),[FM(factorization machine)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
and FFM(field-aware factorization machine) model. In recent years, there have been many improved methods based on FM, such as
[deepFM](https://www.ijcai.org/proceedings/2017/0239.pdf), FNN, PNN, [DCN](https://arxiv.org/abs/1708.05123),[xDeepFM](https://arxiv.org/abs/1803.05170). 


## 2.Principle

FM mainly solves the problem of feature combination under sparse data. And the complexity of its prediction is linear, which has good generality for continuous and discrete features. 

There follows an example.

![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/3.png)


The score is label. User id, movie id and scoring time are characteristics. Because the user id and the movie id are category characteristics, they need to be converted into numeric features through one-heat coding.So that the sample data becomes sparse after onehot encoding. 

There follows an example from data set S.

![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/1.png)

Example for sparse real valued feature vectors x that are created from the transactions of example 1. Every row represents a feature vector x(i) with its corresponding target y(i). The first 4 columns (blue) represent indicator variables for the active user; the next 5
(red) indicator variables for the active item. The next 5 columns (yellow) hold additional implicit indicators (i.e. other movies the
user has rated). One feature (green) represents the time in months. The last 5 columns (brown) have indicators for the last movie the
user has rated before the active one. The rightmost column is the target  here the rating.

### 2.1 Feature Crosses

In general linear models, we consider each feature independently and do not take into account the relationship between features and features. In fact, however, there may be a certain correlation between features.For example, most male users watch more military news, while female users like emotional news. So we can see that there is a certain correlation between gender and news channels. If we can find out this kind of characteristics, it is very meaningful. 


Feature crosses is taken full account by FM model.

Model Equation: 

The model equation for a factorization machine of degree d = 2 is defined as:
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/2.png)


where the model parameters that have to be estimated are:
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/4.png)

And <·,·> is the dot product of two vectors of size k:
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/5.png)

 The model equation of a factorization machine (eq. (1)) can be computed in linear time O(k n).
 Proof: Due to the factorization of the pairwise interactions, there is no model parameter that directly depends on two variables (e.g. a parameter with an index (i, j)). So the pairwise interactions can be reformulated:
 ![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/6.png)
 









