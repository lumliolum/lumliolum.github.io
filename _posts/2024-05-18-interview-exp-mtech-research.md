---
layout: post
title: Interview Experience at IISc CSA & CDS
date: 2024-05-18
---

Recently, I appeared for the interview at CSA and CDS of IISC for MTech Research. Here I will share the experience of the same. I will also add my final thoughts on overall how I felt and where did things go right or wrong (mostly wrong).

## CSA

The interview was on May 13th. Previous years they used to take test and then the interview but this time they announced that they will only take the interview.

There were two professors of which I recognized one of them was Prof. Shalabh Bhatnagar (I2) (don't know about second professor).

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

I1: With a smile he said, you want to work on trendy things ?

M: I was just calm, don't know what to say.

I1: As you are applying for a research position, we expect you to be good with fundamental topics. So let's start with linear algebra.

I2: What topics have you studied in linear algebra, also what reference you took ?

M: I mentioned that I covered most of the topics like solutions to $Ax = b$, projections, eigenvalue & vectors and singular value decomposition. I watched gilbert strang videos and also his book which he recommends

I2: okay, can you explain about eigenvalues and eigenvectors of a matrix ?

M: These values exist for a square matrix, and they are the solutions of the equation $ Ax = \lambda x$ where $x$ is the eigenvector and $\lambda$ is the eigenvalue.

I2: Can you tell us how to find eigenvalues and eigenvectors ?

M: I have mentioned that we can rewrite the equation as $(A - \lambda I)x = 0$. Here, $x = 0$ satisfies the equation but we are not interested where eigenvector is zero. We are interested in values of $\lambda$ where eigenvector is not zero. So due to this we can write $|A - \lambda I | = 0$ (if this not the case, then $A - \lambda I$ will be invertible making $x = 0$). The above equation is a polynomial of degree $n$ (where A is $n \times n$ matrix). As, $\lambda$ is a root of a polynomial, it can be a real value or complex value. Once we found the eigenvalue, we can always find the corresponding eigenvectors. Also pointed if $x$ is eigenvector, then $2x$ is also a eigenvector. Here I have clarified in more detail, that we usually define eigenvectors are the vectors which when multiplied by matrix $A$ don't change their direction but only change the length. The reason I didn't define it like this because I particularly don't like this definition as it is possible that eigenvalue can be a complex number and when were multiplying a complex number, it will change the direction.

I2: You said that if $x$ is eigenvector, then $2x$ is a eigenvector. Is $2\lambda$ is an eigenvalue if $\lambda$ is eigenvalue.

M: No, it will be not an eigenvalue. Like it should satisfy the polynomial, but if you see $2\lambda$ will not satisfy the polynomial.

I2: Is it possible to find the eigenvalues for all matrix ?

M: I said yes, as eigenvalues are roots of polynomial and we can always find roots of the polynomial. It is possible that they are complex, but they will always exist.

I2: Can you find the eigenvalue and eigenvectors for the following matrix. You should not solve the polynomial but use properties. Also can you tell me if eigen values are real or complex ?

$$
A = \begin{pmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 \\
\end{pmatrix}
$$

M: I told him that the matrix is a permutation matrix. After some time, I recognized it as symmetric matrix and told that eigen values will be real.

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

Now I took some time trying to figure out what are other eigenvalues.

I2: what properties do you know about eigen values ?

M: I told that sum of eigenvalues will be trace of matrix and product of eigenvalues will be determinant. So here we will have

$$
\lambda_{2} + \lambda_{3} = 0 \\

\lambda_{2} \lambda_{3} = -1
$$

which gives $\lambda_{2} = 1$ and $\lambda_{3} = -1$. (The reason I was taking some time before was I calculated the determinant as 0 (which is wrong). So because of this I concluded that one eigenvalue should be 0, but the matrix has independent columns, so it can't have eigenvalue zero. this is where I got stuck.)

With this linear algebra questions were completed. the other professor took over.

I1: Can you tell me about cumulative distribution function (CDF) of a random variable ?

M: For a random variable $X$, we defined CDF as $F_{X}(x) = P(X \leq x)$.

I1: Can you draw the graph of sample cdf ?

M: I drew the sigmoid function which is $F_{X}(x) = \frac{1}{1 + e^{-x}}$.

I1: According to your graph, CDF is continous. will it always be the case ?

M: I told that no, I have plotted the graph for a continous random variable. For a discrete random variable the CDF will be not be continous.

I1: Can you draw the cdf for a geometric distribution ?, Before that can you tell me what is a geometric distribution.

M: I told him that a random variable $X$ that follows the following distribution is geometric.

$$
P(X = x) = (1-p)^{x-1}p
$$

where $x = 1, 2, 3, 4,...$

The one I have written here is probability mass function, then I drew the graph of cdf with discontinuity at $x = 1, 2, 3, 4, 5...$. This took some time as I had do some calculations for plotting the graph.

I1: Can you tell the properties of CDF ?

M: As you can see from graph, one is that CDF is always between 0 to 1, that is $ 0\leq F_{X}(x) \leq 1$. Another thing is that at $x = + \infty$, $F_{X}(x)$ will be 1 and at $x = - \infty$, $F_{X}(x)$ will be 0. Other than that it is non-decreasing function (it took some time for me get this term. I first told that it is increasing function, then corrected saying that function that cannot decrease). Also it is right continous.

I1: (by pointing to the graph) For the geometric distribution, can you tell at what points these jumps happen ?

M: These jumps will be occur at $x = 1, 2, 3, 4, 5, 6, 7...$

I1: Can you tell if this graph has finite jumps or infinite jumps ?

M: As jumps will be at positive integer, there will be infinite jumps.

I1: Is this countable infinite or uncountable infinite ?

M: As its positive integers, it will be countably infinite.

I1: Is it possible to have a random variable with uncountable infinite jumps ?

M: I told them to give sometime as I don't know this. After some thought, I was not able to give the answer,

I1: Lets conclude the interview. But do think about this question.

The last one was kind of tricky. Like if the function has uncountable infinite jumps, then it is continous (not sure though but this what I feel), but then if function is continous then it will have zero jumps. This is where I got stuck, so was not able to answer at all.

### FINAL THOUGHTS

Overall I felt that interview went well. Like mostly I1 was very friendly, he was mostly smiling. For example at the end he asked me if I have applied at other places, I told him that I applied to IIT-B, then he asked which one I will prefer. I told him that I will prefer iisc, then he with a smile asked "All your statements are true ?" (I guess he was also hinting that I wanted to PhD, but then opting for Mtech Research and all the statements I made about choosing Mtech Research over PhD.). Also in terms of experience, I also felt very good while giving the interview. I felt that professors and I were in good sync. Just after the end, I was confident about my performance and thought I will get selected.

On 18th May morning (same day writing this blog), they published the provisonal list in the website and I was not in the list. To be honest, I was really sad, don't know what went wrong. Even now, I feel that my interview went well, I understand that it was not perfect but it was good. There were some places that professor had to give hints (like in question of eigenvalues), but I didn't see that as very bad sign (may be because of this they didn't select but not sure though). I felt that I should write an email asking for what was the reason and all that, then controlled myself and at the end didn't write it. (To be honest, it will not change anything.)

## CDS

The interview was on May 17th. It had two parts, one was hacker-earth test and followed by the interview.

The hackerearth test (had to give the round in one of the department computer lab) had 5 MCQ's and 1 programming question. Each MCQ was for 5 marks and programming question was for 15 marks. Overall it was 40 marks. I will write the questions that I remember.

### HACKER-EARTH TEST

Q1. Plot the graph of $f(x) = sin(x)log(|x|)$

A: It had really confusing options (two options were very close).

Q2. 

Q3. Question related to indepedent vectors

Q4. question related to bayes theorem.

Q5. Probability of A winning a game against B is 0.52. They