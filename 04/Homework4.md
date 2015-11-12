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

$\forall TAG \in D_n, \sum_{i=1}^nP(next=y_i|current=TAG) = 1$

$\forall TAG \in D_n,  \sum_{i=1}^n P(word=x_i|tag=TAG) = 1$

**Second step**, optimization. In order to get the maximized log likelihood, we take derivative according to $P(y_i|y_{i-1})$ and $P(x_i|y_i)$ and set them to zero for the optimal value. We get following two expressions (2) and (3), where $W_i$ stands for $word_i$, and $T_i$ stands for $Tag_i$ :

(2): $\frac{\partial logP(x_1,x_2,...,x_n,y_0,y_1,...,y_{n+1})}{\partial P(next=T_i|current=T_j)} = 0$

(3): $\frac{\partial logP(x_1,x_2,...,x_n,y_0,y_1,...,y_{n+1})}{\partial P(word=W_i|tag=T_i)} = 0$

With these two equations and two conditions given above, we can reach the estimation of the ***transition probability*** and ***emission probability***:

*Transition Probability: $T(a|b)= \frac{count(current=b,next=a)}{count(b)}$

*Emission Probability: $E(x|y)= \frac{count(label=y,word=x)}{count(label=y)}$

###1.4 In this example:
With the estimators we got from 1.3, the ***transition matrix*** and ***emission matrix*** are following:

* **Transition matrix:**

Transition|current=START|current=X|current=Y|current=Z
----|-------:|------:|-----:|-----:
next=X|$ \frac{1}{2}$|$\frac{2}{7}$|$\frac{1}{4}$|$\frac{1}{7}$
next=Y|0|0|0|$\frac{4}{7}$
next=Z	|$\frac{1}{2}$|$\frac{3}{7}$|0|$\frac{1}{7}$
next=END|0|$\frac{2}{7}$|$\frac{3}{4}$|$\frac{1}{7}$
* **Emission matrix:**

Emission|X|Y|Z
---|---|---|---
a|$\frac{3}{7}$|$\frac{1}{2}$|$\frac{1}{7}$
b|$\frac{2}{7}$|0|$\frac{4}{7}$
c|$\frac{2}{7}$|$\frac{1}{4}$|$\frac{1}{7}$
d|0|$\frac{1}{4}$|$\frac{1}{7}$

##2. Viterbi Algorithm
* **Step 1.**
At the first position, we calculate the posibility for each label *y*, where $P = T(next=y|current=START)E(word=a|label=y)$

X|Y|Z
---|---|---
$\frac{1}{2}*\frac{3}{7}=\frac{3}{14}$|0|$\frac{1}{2}*\frac{1}{7}=\frac{1}{14}$
* **Step 2.**
At the second position, we compute:
	
  1.the maximum probabilities of each labels at this position, by multiplying the trasition probability from that label and the corresponding emission probability for different labels and take the maximum one among them
  
  2.by the way memorise from wich possible label at the previous position we generated this maximum probability, in other words, the parent node (previous label) of this label.

X|Y|Z
---|---|---
$\frac{3}{14}*\frac{2}{7}*\frac{2}{7}=\frac{6}{343}$ from X|0|$\frac{3}{14}*\frac{4}{7}*\frac{3}{7}=\frac{18}{343}$ from X

* **Step 3.**

Based on the result of first two steps, now we have to compute the probability of each label when the setence end. Just multiply the probability of the label at the final position with the label's probability to occur at the end, and take the maximum one.

**End**: $\frac{18}{343}*\frac{1}{7}=\frac{18}{2401}$ from Z.

So we have got the longest path, and therefore the most probable label sequence is:

START -> X -> Z -> END

##3. 2-Order Hidden Markov Model
###Generating phase
To apply this 2-order HMM, at each position, we will calculate the highest score of combination of different possible labels at this position and the position in front of the current one. As we know at each position $i$, there are $T$ different possible labels at this position $p_i$. And for each one of these $T$ possibilities, there are also $T$ possibilities for $p_{i-1}$, the position in front of it. To compute the highest score for each pair of words in sequence ($p_{i-1}$,$p_i$), we have to enumerate all $T$ possibilities in $p_{i-2}$, and take the maximum one of them. 

Thus, at each $p_i$, our algorithm use $O(T^3)$ time to maintain a $O(T^2)$ matrix to store the highest score for labels in a pair of positions $(p_{i-1},p_i)$, and store from which label of $p_{i-2}$ each highest score was generated, with the function:

$P(x_1,x_2,...,x_n,y_1,y_2,...,y_n)=\prod_{i=1}^{n+1}P(y_i|y_{i-2},y_{i-1})\prod_{i=1}^{n}P(x_i|y_i)$

As we have *n* positions in total, in this phase, the time complexity is O(nT^3) and space complexity is O(nT^2).

###Decoding phase
Finally after our generating phase, at the $n+1$ position where the sentence's END. And we can get a pair of labels for position pair $(p_{n-1},p_n)$ from where we reached the end of sentence.

Then with this pair of label in $(p_{n-1},p_n)$ and the $O(T^2)$ matrix we stored in $p_{n}$ we can generate the label pair for position pair $(p_{n-2},p_{n-1})$. So on so forth, till back to the beginning, we can decode the entire sentence. This process is O(n).

##4. 