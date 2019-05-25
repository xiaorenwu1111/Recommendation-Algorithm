# Theory of FM（Factorization Machines） Algorithm


## 1.Background


Click-through rate is a very important link in computational advertising and recommendation system. It is necessary to determine whether an
item is recommended or not according to CTR. The commonly used methods in the industry are artificial feature engineering + LR (logistic regression), GBDT(gradient boosting decision tree),[FM(factorization machine)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
and FFM(field-aware factorization machine) model. In recent years, there have been many improved methods based on FM, such as
[deepFM](https://www.ijcai.org/proceedings/2017/0239.pdf), FNN, PNN, [DCN](https://arxiv.org/abs/1708.05123),[xDeepFM](https://arxiv.org/abs/1803.05170). 


## 2.Principle

FM mainly solves the problem of feature combination under sparse data. And the complexity of its prediction is linear, which has good generality for continuous and discrete features. 
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/3.png)
