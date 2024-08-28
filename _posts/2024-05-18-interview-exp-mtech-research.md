---
layout: post
title: Interview Experience at IISc CSA & CDS
date: 2024-05-18
---

Recently, I appeared for the interview at CSA and CDS of IISC for MTech Research. Here I will share the experience of the same. I will also add my final thoughts on overall how I felt and where did things go right or wrong (mostly wrong).

## CSA

The interview was on May 13th. Previous years they used to take test and then the interview but this time they announced that they will only take the interview.

There were two professors of I1 and I2. (I recognized one of them)

I1: Read my details from his laptop. It was about my btech, my work experience etc and asked me to confirm it ?

Once the confirmation,

I1: Asked about why I am only applying to Mtech Research and not PhD ?

M: I have been thinking about for some time, which programme I should pursue. PhD is kind of my final goal but I wanted to take one step at a time. Also I got to know that we can always convert to PhD after one of year of MTech Research. This option attracted me a lot to choose Mtech Research.

I1: If your final goal is PhD, then why not choose PhD at the start ?

M: In this 1-1.5 year I will also know how I will perform in a research setting (like working in a research environment). Based on this, I can take more informed decision

I1: The scholarship amount for PhD is higher that Mtech Research

M: Yeah, you are right, but I am okay with it, as its only for 1 - 1.5 year.

I1: Your first preference was CDS and second one is CSA. Why is that the case ?

M: I told him that I wanted to work in the area of NLP, (or in deep learning).

I1: As you are applying for a research position, we expect you to be good with fundamental topics. So let's start with linear algebra.

I2: What topics have you studied in linear algebra, also what reference you took ?

M: I mentioned that I covered most of the topics like solutions to $Ax = b$, projections, eigenvalue & vectors and singular value decomposition. I watched gilbert strang videos and also his book which he recommends

I2: okay, can you explain about eigenvalues and eigenvectors of a matrix ?

M: These values exist for a square matrix, and they are the solutions of the equation $ Ax = \lambda x$ where $x$ is the eigenvector and $\lambda$ is the eigenvalue.

I2: Can you tell us how to find eigenvalues and eigenvectors ?

M: I have mentioned that we can rewrite the equation as $(A - \lambda I)x = 0$.
Here, $x = 0$ satisfies the equation but we are not interested where eigenvector is zero. We are interested in values of $\lambda$ where eigenvector is not zero. So due to this we can write $\lvert A - \lambda I \rvert = 0$.

If this not the case, then $A - \lambda I$ will be invertible making $x = 0$. The above equation is a polynomial of degree $n$ (where A is $n \times n$ matrix).

As, $\lambda$ is a root of a polynomial, it can be a real value or complex value. Once we found the eigenvalue, we can always find the corresponding eigenvectors. Also pointed if $x$ is eigenvector, then $2x$ is also a eigenvector. Here I have clarified in more detail, that we usually define eigenvectors are the vectors which when multiplied by matrix $A$ don't change their direction but only change their length. The reason I didn't define it like this because I particularly don't like this definition as it is possible that eigenvalue can be a complex number and when we are multiplying by a complex number, it will change the direction.

I2: You said that if $x$ is eigenvector, then $2x$ is a eigenvector. Is $2\lambda$ an eigenvalue if $\lambda$ is eigenvalue.

M: No, it will be not an eigenvalue. Like it should satisfy the polynomial, but if you see $2\lambda$ will not satisfy the polynomial.

I2: Is it possible to find the eigenvalues for any matrix ?

M: I said yes, as eigenvalues are roots of polynomial and we can always find roots of the polynomial. It is possible that they are complex, but they will always exist.

I2: Can you give me a matrix, for which all eigen values are zero ?

M: After some thought, I told him that it will be zero matrix. The polynomial for this matrix will be $\lambda^{n} = 0$ which has 0 as its only root.

I2: Can you find the eigenvalue and eigenvectors for the following matrix. You should not solve the polynomial but use properties. Also can you tell me if eigen values are real or complex ?

$$
A = \begin{pmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 \\
\end{pmatrix}
$$

M: I told him that the matrix is a permutation matrix. After some time, I recognized it as symmetric matrix and said eigen values will be real.

I2: okay, can you find them ?

M: As its permutation matrix which swaps row 2 and 3, I said one of the eigenvalue will be 1 because of

$$
\begin{pmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 \\
\end{pmatrix}
\begin{pmatrix}
x \\
1 \\
1 \\
\end{pmatrix}
=
1
\begin{pmatrix}
x \\
1 \\
1 \\
\end{pmatrix}
$$

Now I took some time trying to figure out what are other eigenvalues (meanwhile professor gave some direction).

I2: what properties do you know about eigen values ?

M: I told that sum of eigenvalues will be trace of matrix and product of eigenvalues will be determinant. So here we will have

$$
\lambda_{2} + \lambda_{3} = 0
$$

$$
\lambda_{2} \lambda_{3} = -1
$$

which gives $\lambda_{2} = 1$ and $\lambda_{3} = -1$. (The reason I was taking some time before was because I calculated the determinant as 0 (which is wrong). So because of this I concluded that one eigenvalue should be 0, but the matrix has independent columns, so it can't have eigenvalue zero. this is where I got stuck.)

With this linear algebra questions were completed. the other professor took over.

I1: Can you tell me about cumulative distribution function (CDF) of a random variable ?

M: For a random variable $X$, we defined CDF as $F_{X}(x) = P(X \leq x)$.

I1: Can you draw the graph of sample cdf ?

M: I drew the sigmoid function which is $F_{X}(x) = \frac{1}{1 + e^{-x}}$.

I1: According to your graph, CDF is continous. will it always be the case ?

M: I said no, I have plotted the graph for a continous random variable. For a discrete random variable the CDF will be not be continous.

I1: Can you draw the cdf for a geometric distribution ?, Before that can you tell me what is a geometric distribution.

M: I told him that a random variable $X$ that follows the following distribution is geometric.

$$
P(X = x) = (1-p)^{x-1}p
$$

where $x = 1, 2, 3, 4,...$

The one I have written here is probability mass function, then I drew the graph of cdf with discontinuity at $x = 1, 2, 3, 4, 5...$. This took some time as I had do some calculations for plotting the graph.

I1: Can you tell the properties of CDF ?

M: As you can see from graph, CDF is always between 0 to 1, that is $ 0\leq F_{X}(x) \leq 1$. Another thing is that at $x = + \infty$, $F_{X}(x)$ will be 1 and at $x = - \infty$, $F_{X}(x)$ will be 0. Other than that it is non-decreasing function (it took some time for me get this term. I first said it is increasing function, then corrected saying that function that cannot decrease). Also it is right continous.

I1: (by pointing to the graph) For the geometric distribution, can you tell at what points these jumps happen ?

M: These jumps will be occur at $x = 1, 2, 3, 4, 5, 6, 7...$

I1: Can you tell if this graph has finite jumps or infinite jumps ?

M: As jumps will be at positive integers, there will be infinite jumps.

I1: Is this countable infinite or uncountable infinite ?

M: As its positive integers, it will be countably infinite.

I1: Is it possible to have a random variable with uncountable infinite jumps ?

M: I told them to give sometime as I don't know this. After some thought, I was not able to give the answer,

I1: Lets conclude the interview. But do think about this question.

The last one was kind of tricky. Like if the function has uncountable infinite jumps, then it is continous (not sure though but this what I feel), but then if function is continous then it will have zero jumps. This is where I got stuck, so was not able to answer at all.

### FINAL THOUGHTS

Overall I felt that interview went well. Like I1 was very friendly, he was mostly smiling. For example at the end he asked me if I have applied at other places, I told him that I applied to IIT-B, then he asked which one I will prefer. I told him that I will prefer iisc, then he with a smile asked "All your statements are true ?" (I guess he was also hinting that I wanted to PhD, but then opting for Mtech Research and all the statements I made about choosing Mtech Research over PhD.). I just smiled (felt like he was just pulling my leg). Also in terms of experience, I also felt very good (kind of positive vibe) while giving the interview. I felt that professors and I were in good sync. Just after the end, I was confident about my performance and thought I will get selected.

On 18th May morning (same day writing this blog), they published the provisonal list in the website and I was not in the list. To be honest, I was really sad, don't know what went wrong. Even now, I feel that my interview went well, I understand that it was not perfect but it was good. There were some places that professor had to give hints (like in question of eigenvalues), but I didn't see that as very bad sign (may be because of this they didn't select but not sure though). I felt that I should write an email asking for what was the reason and all that, but at the end didn't write it. (To be honest, it will not change anything.)

## CDS

The interview was on May 17th. It had two parts, one was hacker-earth test and followed by the interview.

The hackerearth test (had to give the round in one of the department's computer lab) had 5 MCQ's and 1 programming question. Each MCQ was for 5 marks and programming question was for 15 marks. Overall it was for 40 marks and 45 minutes to complete the test. I will write the questions that I remember.

### HACKER-EARTH TEST

Q1. Plot the graph of $f(x) = sin(x)log(\lvert x \rvert)$

A: It had really confusing options (two options were very close).

Q2. If A is $15 \times 15$ skew symmetric matrix, then determinant of A is always

Ans: It will be zero (skew symmetric of odd order).

Q3. Question related to indepedent vectors.

Q4. Question related to bayes theorem.

Q5. Probability of $A$ winning a game against $B$ is 0.52. They will have series of matches. The series can contain 3 matches, 5 matches, 7 matches. In which of these series, probability of winning a series for $B$ is highest ?

Ans: I couldn't solve this completely. We say that $p = 0.48$, then probability of winning a series with $n$ matches is (for $B$ and also assume that $n$ is odd)

$$
P = {n \choose n}p^{n}(1-p)^{0} + {n \choose n-1}p^{n-1}(1-p)^{1} + {n \choose n-2}p^{n-2}(1-p)^{2} + ... + {n \choose \frac{n+1}{2}}p^{\frac{n+1}{2}}(1-p)^{\frac{n-1}{2}}
$$

$$
P = \sum_{k=1}^{\frac{n+1}{2}} {n \choose k}p^{k}(1-p)^{n-k}
$$

Here I didn't know how to proceed.

Q6. Write a program to left rotate a array $K$ times. For example if array is [1, 2, 3, 4, 5], then one left rotation will give [2, 3, 4, 5, 1].

After the exam they announced the results, and I was selected for the interview.

### INTERVIEW

As per my lab preferences I have selected NLP as my first preference and VCL as second preference. My highest preference was NLP, and also I didn't prepare much for VCL. So I got assigned to panel where NLP Lab professor was present (there was another professor but he was not there in the panel when my interview was going on)

Before starting the interview, professor confirmed my details like my btech, current organization etc.

I1: You have gone through the research topics of the lab right? Which of them you are interested in ?

M: I am interested in assitive writing

I1: Do you know about the topic ?

M: I do have some idea on this topic. Like other than completing a sentence, we should also teach people of how to write etc.

It was not that smooth answer and this set the tone of interview. From this point onwards, it was totally downhill.

I1: What do you think the important challenges in NLP ?

M: I started talking about the history of NLP. Like till 2016, we were using RNN, LSTM etc then after that transformers were introduced. The pretraining and finetuning paradigm was working really good in-case of transformers.

I1: I know about the history, can you tell about the challenges.

M: After some thought, I said solving mathematical problems is something that NLP applications are not good at.

Here, he asked can you give an example. I told him that if you give a maths problem to chatgpt (like problem of integration etc), there are high chances that it will not be correct.

I1: Can you tell how do we solve this problem ?

M: I feel we are solving the problem in a wrong way. That is for example if you want to solve a integration problem, you should know what is a function, what is a limit, what is a derivative and then what is the integral. So ideally there should be state where we should update the knowledge. Like first updating with function, then limits, derivates etc. This way of updating state is not present in transformers (which the modern LLM's use) and I feel that if we do this then we can solve this.

I1: So which of models have this property ?

M: I said RNN's, LSTM's as they are more suitable and also I feel that these models are more intuitive, that is we have internal state which get's updated after input from each timestep.

I1: So do RNN's, LSTM's perform very well on this task ?

M: I said no, they don't perform (Also I was gesturing that even they don't perform, this is the direction we should take. But then don't know if the professor understood what I wanted to say)

Till here, the interview was downhill. Like there was no sync. Also the reason I started by telling the history of NLP was to tell that even though we are performing really good with transformers etc but we don't have a view like state which gets updated after each input. (in case of RNN, LSTM we have)

I1: Can you tell me about what is word2vec and how do we train them?

M: As the name suggests, this is way to train vector representations for each word present in the corpus. Usually the number of words present in a big corpus is like in range 10 lakhs etc, but here we represent the word in vectors of size 300 etc. Because of this, they are called as  dense representations (like we will need 10 lakh dimension vector if we want to represent each word uniquely by 0 and 1) and also known as word embeddings.

I1: Can you tell how do you train them ?

M: There are 2 algorithms to train this. One of them is skip-gram and other one is continous bag of words. I told him that I know about skip-gram model. In skip-gram, we take a sentence and then a window size. For a window, we predict the neighbouring words given the center word. For example if window size is 5, then we take first 5 words of the sentence, then we say given 3rd word, predict the 1st, 2nd, 4th and 5th word. The probability is calculated like this

$$
P(w_{i} = w_{t} | w_{c}) = \frac{e^{w_{t}^{T}w_{c}}}{\sum_{k = 1}^{V}e^{w_{k}^{T}w_{c}}}
$$

where $i$ is position, that is we say probability that the word at $i$'th position is $w_{t}$ is given by above formulae (note that this is done for one position, but we have to do for all positions).

One more thing to note is that I didn't write the denominator term at first and was explaining things, but the professor asked about it in between and I had to write it.

I1: The summation in the denominator is over the whole vocabulary. Don't you think computing it every time will be very expensive.

M: yes, and this is where word2vec paper introduces a algorithm called negative sampling which will kind of approximate the summation (more precisely it will introduce another objective function). As, I was about to explain this in detail, the professor asked me to stop.

I1: That's okay, now can you tell how do you know if your embeddings are good ?

M: I told him that there is a famous example. That is we take four words "man",  "woman", "king", "queen". If you see the change only here is gender. So we expect the vector $v(man) - v(woman)$ and $v(king) - v(queen)$ should be close. The closeness can be defined as cosine similarity for example.

I1: That is okay, but can I get a single metric to compare ? For example, let's suppose that I have trained another set of embeddings using my algorithm. How do you compare it ?

M: I first told we can create multiple pairs like this then check the performance on these pairs but also asked him for some time to think. After some thought, I told that we can construct tasks like predicting a next word in a sentence. Based on performance on these tasks we can take a call which embeddings are better.

I1: Can you explain in detail ?

M: I told that if we define a task like "predicting a next word in the sentence.". First, I can use my embedding matrix and train a RNN/LSTM. Then I will use your embedding matrix and train a RNN/LSTM. Then we can compare the performance and then decide which one is better.

Here, I wanted to point out that the task I chose was just an example. We can define multiple tasks, or other tasks and then can take a decision. I am not sure if professor understood this is what I was saying or was thinking that "predicting a next word in a sentence" is only the task that we have to use. (In hindsight, I should have mentioned in more detail, but okay.)

I1: Do you think these vectors have bias ?

M: I said mostly yes. There is famous example where it shown that when asked about crime, then the model usually predicts a black person doing a crime rather than white person. I feel that this is because of the bias present in the dataset.

I1: Does these models amplify the bias in the dataset, or they keep same or they reduce ?

M: They usually keep the same. Also, I am not able to find good argument of why they amplify the bias present in the dataset.

This is where I got another one wrong. In hindsight, I should have said that they amplify the bias but the problem is I don't know why they amplify (based on data they amplify but not sure why that is the case).

I1: okay. Let's move to a maths question. Let's suppose I have a chess board, and I wanted to move from (0, 0) to (8, 8). I have only two options that is right and up. So in how many ways I can reach the destination

M: I told him one approach where we could solve use recursion.

I1: That's okay if we want to write a program. Can you give mathematical formula for number of ways.

M: I asked for some time but couldn't figure it out. Then again asked for some more time but

I1: Let's conclude the interview. Also, do you want to get interviewed for VCL lab?

M: I told him no sir, and that my interview in CSA went well, so may be I will be looking for work related to NLP there if I am not selected here. (Another mess-up I did).

There was some other general questions but that also went in wrong direction.

### FINAL THOUGHTS

Overall, the interview was below average (or bad). For the whole interview, there was no sync. There was no resonance at all.

Like, when two waves are combined at a point, it is possible that they nullify their effect because one has amplitude in one direction and other in opposite direction. I felt the same way in interview, our thoughts were opposite. Because of this, the total interview had very negative vibe.

Also when asked for the interview with VCL lab, I said no. I was not all in good mood and also VCL professor mentioned in website that students who are applying should atleast go through one paper from his recent publications (which I didn't. I tried reading one but didn't understand anything). At that time, I also thought that I will be mostly selected in CSA which in hindsight was wrong call.

Also the maths question was cherry on the cake. Like it took from zero to negative very fast. Once I was outside, I was discussing with another student and he told it should be $\frac{16!}{8!8!}$. The reason is that we need 8 right and 8 up to reach the top. So number of ways of rearranging will be $\frac{16!}{8!8!}$. I don't know if I could have figured this out when given more time (mostly no, because of how the interview was going, I don't think that I could have thought this way). In my defence, I thought that professor could have asked more questions in maths (I felt that may be I could have created good impression if asked more  questions and may be I could have solved one of them), but I can't say much as I messed up.

On the same day (17th May 11:00 pm or something), they published the results in the website. I was not present in the list. So overall, things went wrong very fast :).
