# Enhancing Customer Repurchase Predictions: A Data-Driven Approach with Promotions

The idea of accurately predicting customer behavior is crucial for driving sales and making informed marketing decisions. I proposed an innovative approach that combines data science techniques with business insights to improve predictive models for customer repurchases while focusing on the effectiveness of promotions.

Traditionally, businesses have relied on predictive models that incorporate factors like recency, frequency, and monetary value (RFM) to estimate the likelihood of a customer making a future purchase. However, these models have limitations. They often fail to identify the right customers who would truly benefit from promotional campaigns, leading to suboptimal marketing strategies. Additionally, these models typically predict a single future time point, lacking the ability to capture the evolving nature of customer purchase intentions.

In response, we propose a suitable answer for utilizing customer repurchase predictive modeling that integrates the promotion factor into the model. This approach involves training the model on two distinct groups of customers: one exposed to promotions (promotion=1) and another not exposed (promotion=0). By predicting their future purchasing behavior over a specified time horizon, we can quantitatively assess the effectiveness of promotions.

Furthermore, our method provides a series of probabilities for customer repurchases at various time intervals, enabling us to anticipate when customers are most likely to repurchase and determine the maximum probability of repurchase and the time it takes for a customer to reach that peak probability.

In this analytical project, I used the 'LightGBM' machine learning algorithm within our methodology for customer repurchase predictive modeling.
