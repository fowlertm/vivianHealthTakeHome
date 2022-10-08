## Vivian Health Takehome

### Problem Statement

Use machine learning to try and predict electricity usage in KWH.

### Data

The dataset provided was the `recs2009_public` data, which collects energy-related data from thousands of American households.

### Discussion

#### EDA and feature selection

The natural place to start a project like this is at the beginning. I began by loading in the dataset and poking around. After realizing that there were nearly a thousand columns, I looked for a way to reduce the number of input features.

Since this is a timed takehome assignment not meant to require more than a couple of hours, I made simplifying assumptions all over the place to cut down on input features. Some columns are dominated almost entirely by a single value. Of the ~12,000 values in `'SCALEKER'`, for example, basically all were `-2`, which means 'not applicable'. 

Though I couldn't be *100%* sure that these columns have no predictive value without some more testing, it seems reasonable enough to proceed without them. 

Upon inspecting the data I realized that some of the columns were essentially duplicates of the target `KWH` variable, either containing a subset of the calculation that went into that column or expressing the same quantity in different units. Correlation that's too high can cause metrics like accuracy to be much higher than is warranted, so we want to get rid of these too. 

At this point I trained and tested a basic linear regression, getting a score of 99.9%. This made me suspicious, so I went back in and aggressively removed highly-correlated columns. A more sensible score of ~96%, along with a very muted correlation matrix, justifies my concerns over multicollinearity. 

Finally, I had to scale the features. Here's where some subtlety comes in. The data contained a mix of int, float, and categorical datatypes. You've got to be careful with whether and under what circumstances you standardize a categorical variable, and different techniques exist for scaling e.g. ordinal v.s. cardinal features. 

There's an entire subfield of ML which thinks about handling data for preprocessing--using dimensionality reduction techniques such as principal components analysis and non-negative matrix factorization to reduce the feature space, weighing different normalization methods based on the datatypes, and so on.

For the sake of simplicity I just elided all this by dropping any categorical column and scaling what was left with the sklearn `StandardScaler` defaults :)

In a proper project, of course, a great deal of time would be spent tinkering with these paramaters and discovering what works best. 

#### Models

This seems like a pretty straightforward estimation problem, but even here a lot could be said. Is a linear model appropriate, for example, or would it make more sense to use a cluster- or neighbors-based regressor?

Per the famous `no free lunch theorem`, one must either make assumptions about the underlying data or test every kind of model--there's simply no way to say a priori whether a linear regression or a deep convolutional neural network will do better on a given dataset. 

For this assignment, I decided to go the empirical route and simply try out a couple of basic regression models. I settled on a vanilla linear model, a decision tree regressor, and a knn regressor. 

The linear regressor won hands down, with a very respectable ~97%, handling beating the two alternatives. This would seem to offer support to my belief that a linear model is well-suited to these date, although there are many tests, statistical and otherwise, you'd want to run to make sure. I could plot the residuals and make sure they're normally distributed, for example, but that's a project for another time. 

