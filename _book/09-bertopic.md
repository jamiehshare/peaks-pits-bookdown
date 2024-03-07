# Step 7: BERTopic to find high-level peaks and pits {#step-seven}

So now we have an extremely refined set of posts classified as either peak or pits. The next step is to identify what these moments actually relate to.

To do this, we employ topic modelling via [BERTopicR](https://aoiferyan-sc.github.io/BertopicR/) to identifying high-level topics that emerge within the peak and pit conversation. This is done separately  for each product and peak/pit dataset (i.e. there will be one BERTopic model for product A peaks, another BERTopic model for product A pits, an additional BERTopic model for product B peaks etc). 

As there is already [good documentation on BERTopicR](https://aoiferyan-sc.github.io/BertopicR/) this section will not go into any technical detail for BERTopicR implementation.
