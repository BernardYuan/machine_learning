#Homework4 Hidden Markov Model
#### Yuan Bowei 1001916

##1.Parameters in HMM
HMM consists of two parts, the ***transition parameters*** and ***emission parameters***.

Initially the goal of this model is to get a mapping function:

$$f(x_1,x_2,...,x_n) = argmax_{y_1,y_2,..,y_n} P(x_1,x_2,...,x_n,y_1,...,y_n)$$

Which means that we are trying to find out the tags for a sequence of words which makes the joint probobality $P(x_1,x_2,...,x_n,y_1,....y_n)$ to be maximum. This joint probability is mathematically equivalent to conditional probability (1):
$P(y_1,y_2,...,y_n)P(x_1,x_2,x_3,...,x_n|y_1,y_2,...,y_n)$
###1.1 Transition Parameters
The first term is the probability (1) to get one particular sequence of tags, it can be extended into:

$P(y_1,y_2,...,y_n)$

$=P(y_0)P(y_1|y_0)P(y_2|y_1,y_0)...P(y_{n+1}|y_0,y_1,...,y_n)$
	
Specially, $y_0$ and $y_n$ respectively represent $START$ and $STOP$ tags. As a sentence must 'start', so $P(y_0)$ is regarded as 0.

Here we assume the probability of appearance of each tag is associated with the appearance of all other tags in front of it, it is intuitively correct but a result is that there would be an extremely long dependence list. Therefore, we can make a very strong independence assumption, the incidence of each tag is only dependent on the incidence of the tag justly in front of it. Therefore, a more concise expression is:

$\approx P(y_1|y_0)P(y_2|y_1)P(y_3|y_2)...P(y_n+1|y_n)$

$ = \prod_{i=1}^{n+1} P(y_i|y_{i-1})$

This parameter, or this product of parameters, named ***transition parameters*** which describes the probability of transitioning from one tag to another.

###1.2 Emission Parameters
Now let's look at the second term in (1), the conditional probability, which could be extended to:

$ \prod_{i=1}^n P(x_i|x_1,x_2,...,x_{i-1},y_1)$

And here we can make another strong assumption that the probability of the incidence of each word, is ***only*** dependent of its corresponding tag. So finally the second term in (1) can be presented as:

$\approx \prod_{i=1}^n P(x_i|y_i)$

This is named ***emission parameter*** which defines how one word is 'emitted' from its corresponding tag.

###1.3 How these parameters are exactly estimated
Under the two strong assumptioins, expression(1) can be presented in following form:

$ P(x_1,x_2,...,x_n,y_0,y_1,...,y_{n+1}) \approx \prod_{i=1}^n P(y_i|y_{i-1}) \prod_{i=1}^nP(x_i|y_i)$ 

Now we need to compute the estimator for the $P(y_i|y_{i-1})$'s and $P(x_i|y_i)$'s.

**First step**, the product form of this equation would cause problems when taking the derivative, instead we can take the logarithm of it to transfer them into su forms.

$logP(x_1,x_2,...,x_n,y_0,y_1,..,y_{n+1}) = \sum_{i=1}^nlogP(y_i|y_{i-1}) + \sum_{i=1}^n logP(x_i|y_i)$

Another two facts that we can use to facilitate our further calculation are that:

$\forall WORD \in D_n, \sum_{i=1}^nP(y_i|WORD) = 1$

$\forall TAG \in D_n,  \sum_{i=1}^n P(x_i|TAG) = 1$

**Second step**, optimization. In order to get the maximized log likelihood, we take derivative according to $P(y_i|y_{i-1})$ and $P(x_i|y_i)$ and set them to zero for the optimal value. We get following two expressions (2) and (3):

(2): $\frac{\partial logP(x_1,x_2,...,x_n,y_0,y_1,...,y_{n+1})}{\partial P(y_i|y_{i-1})} = 0$

(3): $\frac{\partial logP(x_1,x_2,...,x_n,y_0,y_1,...,y_{n+1})}{\partial P(x_i|y_i)} = 0$

For equation (2), we can get 