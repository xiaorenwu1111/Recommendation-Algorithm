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
 
## 2.2 Prediction
FM can be applied to a variety of prediction tasks. Among them are:
• Regression: yˆ(x) can be used directly as the predictor and the optimization criterion is e.g. the minimal least
square error on D.
• Binary classification: the sign of yˆ(x) is used and the parameters are optimized for hinge loss or logit loss.
• Ranking: the vectors x are ordered by the score of yˆ(x) and optimization is done over pairs of instance vectors
(x(a), x(b)) ∈ D with a pairwise classification loss.
In all these cases, regularization terms like L2 are usually added to the optimization objective to prevent overfitting.

## 2.3 Learning Factorization Machines
As we have shown, FMs have a closed model equation that can be computed in linear time. Thus, the model parameters(w0, w and V) of FMs can be learned efficiently by gradient descent methods – e.g. stochastic gradient descent (SGD) –for a variety of losses, among them are square, logit or hinge loss. The gradient of the FM model is:
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/7.png)

## 2.4 d-way Factorization Machine
The 2-way FM described so far can easily be generalized to a d-way FM:
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/8.png)
where the interaction parameters for the l-th interaction are factorized by the PARAFAC model with the model parameters:
![image](https://github.com/xiaorenwu1111/Recommendation-Algorithm/blob/master/FM/Figure/9.png)

## 2.5 Summary
FMs model all possible interactions between values in the feature vector x using factorized interactions instead of full
parametrized ones. This has two main advantages:
1) The interactions between values can be estimated even under high sparsity. Especially, it is possible to generalize to unobserved interactions.
2) The number of parameters as well as the time for prediction and learning is linear. This makes direct optimization using SGD feasible and allows optimizing against a variety of loss functions.
In the remainder of this paper, we will show the relationships between factorization machines and support vector machines as well as matrix, tensor and specialized factorization models.

# FM In Action
Movielens-100k dataset was used in this paper incliding u.item，u.user，ua.base and ua.test.


u.item


```python
movie id | movie title | release date | video release date |
IMDb URL | unknown | Action | Adventure | Animation |
Children's | Comedy | Crime | Documentary | Drama | Fantasy |
Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
Thriller | War | Western |
```
u.user
```python
user id | age | gender | occupation | zip code
```

ua.base and ua.test
```python
user id | item id | rating | timestamp
```

In this paper, the score data equal to 5 points is taken as the click data, othewise as the unclicked data, which is constructed as a two-classification problem. 

## Input
Data must be processed into a matrix in FM model. In this paper, pandas is used to process the data, generate the input matrix, and onehot coding the label. 
```python
def onehot_encoder(labels, NUM_CLASSES):
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)
    labels = labels.astype(np.int32)
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size,1), 1)
    concated = tf.concat([indices, labels] , 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0) 
    with tf.Session() as sess:
        return sess.run(onehot_labels)

def load_dataset():
    header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv('data/u.user', sep='|', names=header)
    header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
            'Thriller', 'War', 'Western']
    df_item = pd.read_csv('data/u.item', sep='|', names=header, encoding = "ISO-8859-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])
    
    df_user['age'] = pd.cut(df_user['age'], [0,10,20,30,40,50,60,70,80,90,100], labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])
    
    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
    cols = user_features + movie_features
    
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('data/ua.base', sep='\t', names=header)
    df_train['rating'] = df_train.rating.apply(lambda x: 1 if int(x) == 5 else 0)
    df_train = df_train.merge(df_user, on='user_id', how='left') 
    df_train = df_train.merge(df_item, on='item_id', how='left')
    
    df_test = pd.read_csv('data/ua.test', sep='\t', names=header)
    df_test['rating'] = df_test.rating.apply(lambda x: 1 if int(x) == 5 else 0)
    df_test = df_test.merge(df_user, on='user_id', how='left') 
    df_test = df_test.merge(df_item, on='item_id', how='left')
    train_labels = onehot_encoder(df_train['rating'].astype(np.int32), 2)
    test_labels = onehot_encoder(df_test['rating'].astype(np.int32), 2)
    return df_train[cols].values, train_labels, df_test[cols].values, test_labels
```
## Model design
Tensorflow was used to design our model. The objective function consists of two parts, linear and feature crosses.
```python
#input
def add_input(self):
    self.X = tf.placeholder('float32', [None, self.p])
    self.y = tf.placeholder('float32', [None, self.num_classes])
    self.keep_prob = tf.placeholder('float32')

#forward
def inference(self):
    with tf.variable_scope('linear_layer'):
        w0 = tf.get_variable('w0', shape=[self.num_classes],
                            initializer=tf.zeros_initializer())
        self.w = tf.get_variable('w', shape=[self.p, num_classes],
                             initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        self.linear_terms = tf.add(tf.matmul(self.X, self.w), w0) 

    with tf.variable_scope('interaction_layer'):
        self.v = tf.get_variable('v', shape=[self.p, self.k],
                            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        self.interaction_terms = tf.multiply(0.5,
                                             tf.reduce_sum(
                                                 tf.subtract(
                                                     tf.pow(tf.matmul(self.X, self.v), 2),
                                                     tf.matmul(self.X, tf.pow(self.v, 2))),
                                                 1, keep_dims=True))
    self.y_out = tf.add(self.linear_terms, self.interaction_terms)
    if self.num_classes == 2:
        self.y_out_prob = tf.nn.sigmoid(self.y_out)
    elif self.num_classes > 2:
        self.y_out_prob = tf.nn.softmax(self.y_out)

#loss
def add_loss(self):
    if self.num_classes == 2:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
    elif self.num_classes > 2:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
    mean_loss = tf.reduce_mean(cross_entropy)
    self.loss = mean_loss
    tf.summary.scalar('loss', self.loss)

#accuracy
def add_accuracy(self):
    # accuracy
    self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out,1), tf.float32), tf.cast(tf.argmax(self.y,1), tf.float32))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    # add summary to accuracy
    tf.summary.scalar('accuracy', self.accuracy)

#train
def train(self):
    self.global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                       l2_regularization_strength=self.reg_l2)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

#graph
def build_graph(self):
    self.add_input()
    self.inference()
    self.add_loss()
    self.add_accuracy()
    self.train()
  ```
  
  
Further information is available from this link: https://github.com/xiaorenwu1111/Recommendation-Algorithm/tree/master/FM
 
Welcomes your comments and corrections!
 

# Reference
[1]https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf


[2]https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html


[3]https://zhuanlan.zhihu.com/p/50426292


[4]https://cloud.tencent.com/developer/article/1099532















