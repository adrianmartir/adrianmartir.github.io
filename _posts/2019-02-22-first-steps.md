---
layout: post
title:  "How not to use machine learning for crypto price predictions"
excerpt: "This post is me throwing a bunch of machine learning ideas against the problem of cryptocurrency price predictions and reporting the results."
date:   2019-02-27
tags:
  - machine learning
  - cryptocurrency
mathjax: true
---

{% include mathjax.html %}

The aim of this article is to give a high level overview of the approaches that I tried and how this project evolved, it is not hammering some numpy and pandas functions/methods into your head(if you need to know a function, google and the [source code](https://gitlab.com/adrianmarti/ml-experiments-on-cryptos) are your friends :)).


## Philsophy

### Accurate performance estimates are hard

If you have ever tried to seriously make money based on model predictions, you will know, that there are nasty surprises. There will be a huge magnet above your strategy's performance estimate, that will pull your estimate up, without actually pulling the performance up. So be skeptical of your performance metrics.

Here is a noncomprehensive list of things that could widen the gap between estimated and actual performance:
* fees(trading, deposit, withdrawal)
* server downtimes
* your infrastructure failing
* regime changes(structural changes in your time series, e.g. because of a real life event)
* [slippage](https://www.investopedia.com/terms/s/slippage.asp)
* weird bugs(While working on this project, I used a library that had a bug in it and because of it I thought I had built an insanely good model and I didn't notice until much later - a lot of time wasted)
* failure to detect overfitting
* [survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias)

### Failing fast and often

One of the most powerful approaches to learning anything, is trial and error. So why not try as much as possible in the smallest amount of time? If you come up with a complex solution to a problem, to which you haven't even tried the simple solution, you might spend a lot of time implementing something complex, but in the end, if it doesn't work(and it won't work), you will have wasted a lot of time, because you were missing some key insights into your problem. If you would have tried the simpler approach first, you would have gained those insights much faster and with that you would have been able to find the correct solution much faster.

A consequence of the above paragraph is that you should not aim for perfect from the start, you should aim for something basic instead. Although the above might sound obvious, I hate failing at doing something and this causes me to aim for something perfect and with that I fail miserably *and* waste a lot of time.

### Financial time series are not nice at all

Financial time series data is inherently *very noisy* and thus we can't expect our models to explain large parts of the price movements. The goal here is to have an edge by just having a model that is 'good enough' to make profit in the long run. The question is: *How good exactly does our model have to be?* and *Will be able to achieve that level of performance?*. Because of the noise, we will have to restrict ourselves to simple models, because they can deal much better with it(it's very hard to overfit a linear regression). One of my aims is to be able to somehow apply more complex models effectively, since it is much more fun.


## Tooling

### Good old jupyter notebooks

These are excellent for analyzing and plotting data like model and strategy performance metrics. My policy for jupyter notebooks is to keep them as short and concise as possible, because if I mess up some variables or if I open the notebook on a different day I will have to execute a whole bunch of cells and wait forever. For operations that are not instant and I intend to execute more often, I have a different solution:


### Luigi

[*Luigi*](https://luigi.readthedocs.io/en/stable/) offers the abstraction of **Tasks**, which consist of reading a file from disk, doing whatever with it(eg. preprocessing data, generating features or training a model) and then it writes the result again to a file. Within each task you can specify other tasks as dependencies, so that *luigi* check whether the required files have already been generated, and if not, it will generate them with the corresponding tasks. For example when the user of my code first wants to train a model, luigi will detect that the data is not there and will download it.

The benefit of luigi is that I can put the main operations(the ones executed frequently and/or take a long time) in luigi tasks and then I can create a bunch of jupyter notebooks that analyze the output of those tasks, and with that avoiding having huge jupyter notebook files.

If you want to execute a task, say the class `Naive` located in `first_tries.py` in non-debug mode, you can run:

```python
import differences_model
luigi.build([differences_model.Diffed()], local_scheduler=True)
```

When we start saving preprocessing steps to disc and analyzing predictions and stacking models, you will thank me because of the mess luigi avoids.


### A sheet of paper

To keep track of how different models performed.


### Pro tip

The [**ipython shell**](https://ipython.readthedocs.io/en/stable/) is extremely convenient. In the *src/* directory of [this gitlab repo](https://gitlab.com/adrianmarti/ml-experiments-on-cryptos) I put a file called `.ipythonstart` and I have a shell alias that changes to *src/* directory and then runs this:

```bash
PYTHONSTARTUP=.ipythonstart.py ipython
```

This automatically runs `.ipythonstart` when the shell starts and import everything that I might potentially need. Another thing worth noting in the *ipython* shell is that you can start typing something and then press the arrow keys up and down to find lines you executed beginning with the same string.


## Getting the data

We will be using 30 minute data(one candle = 30 min) from *poloniex*, since poloniex is the only exchange I found that does not put a limit to how far back you can pull the data. You can run the following in a jupyter notebook or in an ipython console or wherever you like to run your python to download the data. Make sure you are in the `src/` directory before running this:

```python
import data
luigi.build([data.DownloadPoloniex()], local_scheduler=True)
```

This task basically repeatedly calls the poloniex api for each possible asset pair and stores the json response.


## Framing the problem

For now we will be treating the problem as a classification problem. At every timestep we will be attempting to predict if the price will increase or decrease in the next timestep. More precisely, if $p$ is the price function, that for each timestep $t$ returns the price at that timestep, our *label/target* for the datapoint corresponding to time $t$, would be the sign of $p(t+1) - p(t)$.


## The metric

The metric I'm using for comparing models is the AUC of the ROC. The bottom line is that an AUC of 0.5 means your model is as good as if you were predicting the more common class for every timestep and 1 means that your classifier is perfect.

## Side note on cross validation for time series

If we wanted to predict the future of a time series, we would use all our data and then predict the values on a future interval. To get an estimate of how our model would perform on future data, we can use a specific point in time to split the data into the *training* set, which would be the datapoints before that date and the *validation* set, which will be used to "simulate" how the model will perform on new data.

To get a better estimate, time series cross validation repeats this procedure for n evenly spaced times across our data, to determine where to split the data, and then takes the average of the error on the *validation* set as the estimate. For more details, see the [*sklearn user guide*](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data). Reported metrics in this post will always be calculated via time series cross validation.


## The naive model

The price increased 51.1% of the time, so one could design a classifier that always predicted that the price would increase, and that would hace an accuracy of 51.1%. The corresponding **AUC would be 0.5**.


### The log difference

In machine learning lingo, we call the intput variables of our model *features*. When applying machine learning, it is our goal, that the *features* of our model are reasonable. The most naive thing to do, when designing a classification model on top of a series of prices, would be to using the price at times $t, t-1, ..., t-k$ to predict the target. Why is that a bad idea?

Taking the raw price is pointless, because certain price ranges may only have been hit once or twice in the entire history of our data and any model would have trouble fitting that. The mere fact of *being* at a certain price range like 100, doesn't have obvious relevance for predicting the price. You may be inclined to think that machine learning models should be smart enough to figure out that they have to look at price changes, but in reality machine learning models aren't magic.

![close]({{ site.url }}{{ site.baseurl }}/assets/images/close.png)

![log_diff]({{ site.url }}{{ site.baseurl }}/assets/images/log_diff.png)

*More nice plots [here]({{ site.url }}{{ site.baseurl }}/notebooks/eda.html)*

**From a probability distributions perspective**:
If we view our prices as a distribution, it would change very drastically over time(the mean rises until the peak of the hype train on December 2017 and then drops again with some more peaks and valleys along the way). When we train a model, it gets an idea of how the distribution looks like on the training data and which parts of the distribution correspond to which labels. If the distribution constantly changes, our model will not be able to make sense of new data. Now the variance of the time series is still over the place - further down the line I will try to deal with that.

A way to get more reasonably distributed time series data would be for example by looking at $p(t) / p(t-1)$, because it shows whether the price increased or decreased independently of how high the price currently is. This gives you distribution, where the mean is almost always roughly 0, so at least the mean of the distribution stays the same. Now another common transformation to apply to time series data would be to take $log(p(t) / p(t-1)$)(*logarithmic returns*), because in this space the multiplication of two return values(ie. computing the total return) corresponds to addition, and addition is much nicer. For example an asset always growing at a constant rate would have a price graph of an exponential function, but if we take the logarithm, it is a linear function, which seems more reasonable.

Now [using only one feature]({{ site.url }}{{ site.baseurl }}/notebooks/first_steps.html), the log returns of the last time period and slapping a logistic regression on top of that variable, you get a surprising **AUC of 0.538** and an accuracy of 52.2%.

If you start [shifting features around]({{ site.url }}{{ site.baseurl }}/notebooks/first_steps.html)(including features from timesteps t-1,..,t-k at time t) and apply a logistic regression, you can increase your performance to an **AUC of 0.541**. Gradient boosting on the same features yields slightly better performance. The thing is: the training AUC is around 0.64, which shows that we are overfitting. The pattern hidden in this data(assuming it exists, lol) is to complex to bruteforce with a lightgbm(quite a complex model), since there is not enough data. To improve our model, we have a few options:

* Find smarter features
* Actually get more data
* Somehow better exploit the amount of data by using transfer or metalearning or [probabilistic programming](https://medium.com/@alexrachnog/financial-forecasting-with-probabilistic-programming-and-pyro-db68ab1a1dba) - though doing this comes at the expense of computation power
* Reframe the problem, eg. predict increases on higher timescales

Before we try any of the other options, we will try to go with the first option for a while.

## Smarter shifts

A better way to encode past data, at least better than shifts, is probably using price differences $p(t) - p(t-k)$ ($p(t)$ is the log price at that time) for $k > 1$. This would encode things like whether we are on a local maximum or minimum. This immediately increases the performance to **0.546 AUC**(using light gradient boosting), which is an improvement.

Smarter features seemed like a good idea at the time, but after adding those smart shifts, it doesn't matter what I add, my performance gets worse. I tried volume diffs(volume is not stationary, so I took log diffs), weekday and more stuff and everything worsened performance. This goes to show that overfitting is through the roof and that our features are still not smart enough. 

Of course one could start optimizing hyperparameters and stacking models to solve my problems, but I think it's diminishing returns at this point, we need a fundamentally better approach.


## Technical Indicators

Let's start a new model from scratch with technical indicators.

When designing features from technical indicators, one has to be aware of the requirement, that the distribution at least stays roughly the same, it would be pointless to add some support lines directly to the model as features. It would probably be more wise to use the distance of the current price to the support line. Also while designing my features, I always quickly looked up how the indicator is really traded and tried to design the feature to match that.

A [simple logistic regression]({{ site.url }}{{ site.baseurl }}/notebooks/logistic_technical.html) on the indicators featureset had an **AUC of 0.551** and accuracy of 53.7%. Another significant improvement. Note that here the logistic regression code and the feature generation code is `src/technical.py` and the linked notebook just analyzes the predictions that model does.

Remember my comment on that magnet pulling your performance estimate up? Well it happened to me with the technical indicator featureset, but back when I was using the [*finta*](https://github.com/peerchemist/finta) technical indicator library. It had a line of code that made an indicator use data from the future, which it shouldn't have access to. Thus, the model I trained had *insane* performance(0.60AUC). I opened an [issue](https://github.com/peerchemist/finta/issues/25) on the finta repo and never touched that library again, out of fear that there are more bugs hiding. Now I use [this talib wrapper](https://github.com/mrjbq7/ta-lib).


## A short note on fat tails

For completeness sake, [here]({{ site.url }}{{ site.baseurl }}/notebooks/fat_tails.html) is my fat tails notebook, that illustrates how predictions on a small amount of timesteps with large price differences can have a huge impact on the strategies overall performance.


## Backtesting

Note that the returns are all in Bitcoin, so this may not reflect the actual returns. This was [also included]({{ site.url }}{{ site.baseurl }}/notebooks/strategies.html) for completeness sake.


## Further reading

* Introduction to quantitative trading: [quantstart](https://www.quantstart.com/articles/Beginners-Guide-to-Quantitative-Trading)
* Some approaches people have tried: [hacker news](https://news.ycombinator.com/item?id=16922538)


## Upcoming articles:

* Reframing the problem