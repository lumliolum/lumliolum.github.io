---
layout: post
title: ICML 2026 Submission Experience
date: 2026-05-10
---

Recently, I submitted a paper to ICML 2026. I want to share my experience about the whole process. Before going ahead, I want to clarify that I'm not leading author of the paper (although I beleive that I did decent contribution) and paper was rejected. The opinions are totally mine and I'm not sure if my coauthors beleive the same.

Our paper was mainly theoretical and to match, we selected primary area as "Theory -> Probabilisitic Methods". The final scores were 4,4,3,3 with confidence 3,2,5,3 respectively. The pre rebuttal scores were 3,3,3,3 with the same confidence scores. Let's call the reviewers as A, B, C, D.

For each reviewer, I will write a breif things they pointed our and how our rebuttal was. I have used very informal language to describe the discussion. For example, when I say that we replied saying that reviewer is wrong, it shouldn't be taken in literal sense, in real, we actually write in a polite way that we disagree with the reviewer.

In this year ICML, the discussion procedure was as follows
- The reviewers will give initial reviews.
- We as authors give our response which is called as rebuttal.
- Then reviewer acknowledges our response. Basically reviewer has to select an option: (a) that is fully resolved, (b) partially resolved and have follow up question, and (c) partially resolved and the remainings are not easily addressable.
- Then we as authors once again give reply which is called as reply rebuttal.
- At the end, reviewer has write a final justification based on the responses, and score can be adjusted if needed. As one of my coauthor was a reviewer, I know that final justification is compulsory by the reviewer.

The scores mean the following
- 1 : Strong reject
- 2 : Reject
- 3 : Weak Reject
- 4: Weak accept
- 5 : Aceept
- 6 : Strong accept.


## REVIEWER A

This reviewer appreciated the way we wrote the paper and theoretical results. However, he said that novelty is limited and contribution is modest. Also, there were was a proposition that we worked, in which the reviewer asked for relevance. In simpler words, he wanted to ask what is the point of this proposition ? What happens if that result is true ? Other than that, he asked questions about future work.

In the rebuttal, we explained how the proposition result can be used and what it meant. In general, we responded saying that one can say that novelty is limited and there is prior work around the area, but our methodology is very non trivial. Also breifly explained through an example what things that you can do with our analysis and not with prior work. We gave brief answer on the future work question.

After the rebuttal, the reviewer increased the score from 3 to 4. It felt more of that reviwer generally liked the paper but didn't feel that it was significant enough. Because of that he only went to 4 (to be fair, increasing the score by 2 points is very rare).


## REVIEWER B

This reviewer had a confidence score of 2 which is basically acknowledging that he is not familiar with similar work in this area. His main concern that our writing style was very confusing and pointed out one has to go appendix to actually understand some notations we used in the main text.

The writing part is something we agreed. Looking back, the first 9 pages of main text was not sufficient to read as stand alone. We agreed to restructring the paper (moving some sections from appendix to main text).

After the rebuttal, the reviewer increased the score from 3 to 4. It felt more like - he didn't want to discuss much, as he was not familiar with the area. In that sense, even if he keeps the score at 3, I don't think it would have affected much but of course I thank him about increasing the score.


## REVIEWER C

Out of the all reviews, we felt that this reviewer asked some serious questions. The reviewer gave a confidence of 5 which means he was well aware of the prior work and understood our paper. Like Reviewer B, he also mentioned that writing was not good and presentation can be significantly improved also gave some suggestions on how to restructure. In the rebuttal, we agreed to it and said that we will do the required revision.

Another main concern was that he didn't like our motivation. He even gave a way of how our work can be motivated by citing related prior work. The thing is we start our analysis with a loss function which we motivate from prior work. According to him, this motivation was handwavy and he didn't like it. Asked for some theoretical properties of this handwavy argument. Also he felt that if we incorporate his sugestions, one has to rewrite lot of sections of paper and requires another round of review.

In the rebuttal, we said that the final objective after the handwavy argument has desired properties. Also pointed out that prior work he cited has some restrictions and can't be used as primary motivation because it doesn't fit in our setting.

The remaining concerns were

- In final step of the proof one of our theorem, we interchanged limit and expectation without explicity stating. Basically, we were using dominated convergence (DCT) theorem, but it was not easy in the first glance to check if all the assumptions of DCT are satisfied. We replied by saying that this can be done and we showed more specific steps and assured that this change will be updated in the revision. I don't know, somehow I totally missed this. Like I vaguely remember that in discussion with my coauthors, this issue came up, but somehow we didn't think it was needed. The interchange step was directly done, and I didn't realize that we were doing interchange of limit and expecatation.
- Other question was on our experiments. Basically, to support our theorems, we performed some experiment. We didn't specify empirical observations that can inferred from experiment results and a practical guidance on how to apply our loss function. We shared our findings and ended our response wth something on the lines "we would be happy to include this empirical observation in the revision". Personally, I felt that this was good point raised by the reviewer.

Reviewer responded by having a follow up question (He selected option (b)). He was not convinced that our starting point has required theortical properties and also was confused by our response and asked for more clarification. We replied (this time in detail) citing the prior work and how we can arrive at the conclusion.

This completed the discussion phase, the reviewer filled final justification "citing that authors need to rewrite motivation section very carefully. He is okay to accept the paper if authors do the required revision. As such, I will keep the score and pass the decision to AC".

The prior work that he cited was useful to us. Even though we can't use it directly, we were able to generalize our theorem because of the result from prior work. Even the reviewer had a same hunch and he only pointed it out that your theorem can be generalized.

So after all the discussion - the final score was kept at 3 (weak accept).


## REVIEWER D

Reviewer appreciated theoritical foundation of the paper but said that it lacked empirical evaluation and was constrained to low dimensional (For example : 10, 20) and applicability to realisitic scenarios. Also asked about applications to generative modeling. He also felt that we didn't compare our method with more recent and highly relevant baseline (the one we compared was in 2019).

In the rebuttal, we first mentioned that generative modeling is out of scope. About the recent baseline comparision - we said that we think we compared with all the methods and asked the reveiwer to specifically point out the papers he wants us to compare.

There was another question about computational challenge for applying to high dimensial setting, we responded saying that we agree and gave a suggestion of how can choose our loss function so that it can be scaled (this was mentioned in the paper as well).

Reviewer acknowledged the response and marked option (b) [partiall resolved and have a follow up question]. He said that his concerns are partially addressed. He said that experiments that have been conducted in the paper - is low dimensional classical settings and pratical applicability in realisitc scenarios is not addressed. This response came to us at night and before our response , the reviewer changed from option (b) to option (c) [partially resolved and remaining concerns cannot be addressed in short rebuttal]. It would make sense if reviewer changed the option in 10-20 minutes of his comment. But he changed nearly after 12 hours.

We again responsed saying that the settings of parameter estimation that we consider are not just toy examples and have practical relevance by citing 3-4 prior works. We also said that our work is theoretical in nature and we have guarantees on the settings that we considered in the paper. because of this, we performed experiments on very controlled setting and showed that how our method is better. Also in the response, we once again pointed out how generative modeling framework is different.

In the end, he didn't fill the final justification and kept the score same (3 -- weak accept).


## META REVIEW

On April 30, we got the notification that paper was rejected and main reason in the meta review was that "multiple reviewers remained unconvinced that the paper provides sufficient empirical breadth and practical relevance". Also, empirical evaluation is too narrow and it misses the recent baselines. Because of this, AC thinks that is paper in the current form is not ready and empirical support practical signifance is needed to justify the acceptance.

Also, meta review appreciated our theoretical contribution.


## MY THOUGHTS AND FINAL COMMENTS

Overall, the experience was not that good. I was kind of sad for some days even though I was ready for this result. (The average score was 3.5 and it was kind of known that your average score should be 4 to have any hope).

Basically, our rejection was based on the comments given by D. Even though meta review says "multiple reviewers" which I think is false and only reviewer D had a difficulty with our experiment evaluation. I'm surprised that AC concluded that multiple reviewers had a problem with our experiments. Reviewer C didn't say that experiments were not good, he asked question related to it. 

Another frustraing part was AC took comments of D who didn't fill the final justification. This was a compulsory action from D which he didn't perform. So I don't know - how can you take his comments.

In the first comment of D, he pointed out we are not comparing with recent baselines. When we asked for specific papers that he think that we missed - he didn't reply on this comment at all. In the end, meta review cites this as one of the reason (not even thinking that we asked for it and reviewer didn't reply). Just because we are comparing with a 2019 paper - doesn't mean that we are comparing with old method and not "recent baseline". What I feel that is this 2019 paper was the closest one to compare. There was another 2022 paper building on 2019, but that was on different direction. So till date, I think we didn't miss any recent papers.

Also, I didn't understand the argument of practical signifance. We particularly selected our primary area as "Theory->Probabilistic Methods", to reflect the contribution we did. It doesn't make sense to reject this paper - based on empirical results. Its not that we are not running any experiments, just because we do parameter estimation in classical settings - doesn't mean its not practical enough.

The final and unsetting part of D was that he changed the option from (b) to (c) overnight (according to our timezone). One could have changed after our response but he changed it before. So to be honest, I didn't like this behaviour at all.

I feel that only reviewer A and reviewer C understood our contribution. C didn't take any decision and passed it to AC. Sometimes, I feel that C should have taken a side rather than passing to AC (For eg., should have decreased the score). In a world where C rejects the paper citing that one has rewrite a lot and it requires significant changes, I would have argued that it was very nitpicky, but could have got some closure with time. The reason I feel that C should have picked a lane is because he gave a confidence of 5.

I'm also thankful to C because he gave a really good idea on how to generalize our theorems. So in real life, suppose I meet C -- I would have just said that you could have picked a lane.

Note : All reviewers are referred as "he, his" etc. It is easy to write for me this way.
