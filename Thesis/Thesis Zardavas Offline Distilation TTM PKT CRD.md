  

### Introduction

Deep Reinforcement Learning (DRL) is a rapidly advancing field, offering a powerful framework for creating autonomous agents capable of making complex sequential decisions. Its application in financial markets, particularly for automated trading, holds significant potential.  DRL can learn optimal strategies in fluctuating environments and navigate complex constraints.  However, the inherent characteristics of financial data present practical obstacles that hinder the wide adoption and reliability of DRL agents.

A significant challenge in using DRL in financial markets, especially in volatile sectors like cryptocurrencies, is the inherent instability of DRL agents during training.  Financial data is noisy, non-stationary (its statistical properties change over time), and contains randomness.  These factors lead to high variance in agent performance across different training runs, even with identical settings.  This often results in problems like overfitting to past noise, failure to adapt to new market conditions, and the creation of trading strategies that are unreliable or inconsistent.  Furthermore, most current DRL-based trading agents primarily use quantitative, price-related information, such as OHLCV data and technical indicators.  They often neglect the potentially valuable qualitative information contained in financial texts and the sentiments they convey.

In this context, Knowledge Distillation (KD) appears to be a promising way to mitigate the instability mentioned above and improve generalization in DRL for stock trading. Knowledge distillation (KD) is based on the idea that a "student" agent learns from a more capable "teacher" agent or ensemble of agents. This effectively transfers robust knowledge, leading to more stable training and improved student performance.

The central thrust of this dissertation is to address the dual challenges of instability in Deep Reinforcement Learning (DRL) within the financial domain and the inadequately explored dynamics of qualitative market signals.  First, it aims to enhance the performance of trading agents by incorporating robust, domain-specific financial sentiment signals into the DRL state representation.  This integration provides a more comprehensive view of the market environment, allowing sentiment to serve both as a vital predictive indicator and a stabilizing element, directly addressing issues of resilience and consistency.  Second, the dissertation undertakes a detailed comparison of different knowledge distillation modalities: logit-based distillation (which transfers refined decision-making policies) and feature-based distillation (which transfers structured internal representations).  This systematic study seeks to identify the most beneficial type of knowledge transfer for a student agent in the financial domain, thereby clarifying a more definitive path toward creating DRL agents that exhibit enhanced training stability, superior generalization capabilities, and, ultimately, increased profitability within the demanding environment of financial markets.  This synergistic dynamic, combining qualitative sentiment analysis with advanced knowledge transfer methods, forms the core research contribution and rationale for the work presented herein.

#### 1.2 - Problem Statement

This dissertation addresses a significant research challenge stemming from the inherent difficulties of applying Deep Reinforcement Learning (DRL) to dynamic and fluctuating financial environments, particularly in cryptocurrency trading.  While DRL agents can learn complex trading strategies and alter their behavior to achieve optimal financial outcomes by directly interacting with the market, the core characteristics of these markets—such as high noise levels, rapid shifts in dynamics (non-stationarity), and overall complexity—make it very difficult for DRL to be consistently successful and widely used in automated financial trading.

These inherent market factors frequently cause significant instability during the DRL training process, resulting in policies that exhibit erratic performance and inadequate resilience when applied to real-world trading scenarios.  The resulting unreliability of models trained under such conditions is evident in the substantial variance in outcomes across different training runs, even with identical settings, which undermines confidence in their practical deployment. It also causes problems with overfitting and poor generalization, where models perform well on old data but fail to adapt well to new market conditions.  Furthermore, a significant observation is that the majority of current DRL-based trading agents rely predominantly on quantitative, price-related data, often overlooking valuable qualitative information such as financial sentiment.

This dissertation delineates a specific research problem to address the inherent training instabilities and performance limitations:

The primary research objective is to investigate the integration of advanced Knowledge Distillation (KD) techniques—namely, Probabilistic Knowledge Transfer from intermediate layers (PKT) and online ensemble methods, potentially in conjunction with logits-based distillation (TTM/WTTM)—within a Deep Reinforcement Learning framework (e.g., Proximal Policy Optimization - PPO) to mitigate training instability and enhance the trading performance (e.g., Profit and Loss) of agents in volatile cryptocurrency markets, utilizing hand-engineered, price-based features.

This study is mainly concerned with the specific application and empirical validation of various advanced knowledge distillation techniques.  The goal is to show that they can bypass the known problems with DRL, especially training instability, in the difficult environment of automated cryptocurrency trading, using only hand-crafted, price-based features. The goal of this method is to give the necessary direction and normalization to create DRL agents that are always stable, robust, and profitable.

#### 1.3 - Research Questions

This dissertation is guided by a set of specific research questions that arise directly from the problem statement outlined in Section 1.2. These questions are designed to explore the practical implementation and effectiveness of advanced Knowledge Distillation (KD) methodologies in alleviating training instability and improving the performance of Deep Reinforcement Learning (DRL) agents for automated cryptocurrency trading, using exclusively hand-engineered, price-based features.

The primary research questions addressed are:

*   How can probabilistic knowledge transfer from intermediate layers (PKT) be effectively integrated into a deep reinforcement learning (DRL) framework to enhance the stability and performance of trading agents in volatile cryptocurrency markets?

    This question looks at PKT, a more advanced KD method that does more than just match outputs. It focuses on moving the knowledge stored in the middle layers of a teacher network.  The main goal is to match the probability distribution in the feature space learned by the teacher.  This will move the basic geometry and relationships seen by the teacher's hidden representations to the student model.  The Cosine Coherence kernel is well-known for being used in PKT.  One of the things that makes it useful is that it doesn't need an explicit bandwidth setting.
*   How significantly do online ensemble knowledge distillation techniques, when deployed within a deep reinforcement learning trading context, reduce training instability and improve trading performance?

    This question looks at online distillation, which is a method for training both teacher and student models at the same time. The proposed framework uses an online ensemble mechanism that trains several teacher models at the same time as the student. These teachers have the same architecture but differ because of how they were initially set up with a random seed (see Chapter 4 with TTM.md). The teacher selection process has two steps: first, all teacher models are trained for a certain number of epochs. Then, the best N teachers are chosen based on a performance measure, like cumulative profit and loss (PnL) during training, and online training is continued with this group. Knowledge from the ensemble is unified by averaging the distillation losses calculated for each teacher.
*   Can logits-based distillation methods, specifically Transformed Teacher Matching (TTM) and Weighted TTM (WTTM), effectively regularize the policy of a Deep Reinforcement Learning (DRL) agent and enhance its generalization capabilities across fluctuating financial markets?

    TTM changes the usual KD by taking away the temperature scaling from the student's logits but keeping it for the teacher. This adds a Rényi entropy regularization term that encourages a smoother, less overly confident policy. WTTM builds on TTM by adding an adaptive sampling weight mechanism that uses the smoothness of the teacher's power-transformed output distribution to decide the weights. This gives more weight to the teacher's outputs that are more helpful. These methods seek to impart an enhanced understanding of the teacher's decision boundary and output confidence calibration.
*   What is the comparative effectiveness of logits-based (TTM/WTTM) and feature-based (PKT) knowledge distillation strategies, as well as their integration, in enhancing trading performance (e.g., Profit and Loss, Sharpe Ratio) and ensuring training stability for DRL agents in cryptocurrency markets?

    This comparative study seeks to identify the most beneficial type of knowledge transfer for DRL agents in a financial setting. The Proximal Policy Optimisation (PPO) framework used for training is well-known for achieving the best results. The PPO's goal uses a clipped surrogate objective with a clipping parameter ϵ (usually 0.2) and Generalized Advantage Estimation (GAE). The issue focuses on cryptocurrency markets that are very dynamic and volatile, using hand-engineered, price-based features to represent the market. These features include ratios of high/low/close candle differences, often joined by smoothing windows, and undergo post-clipping normalization. The agent's observation also includes its position in the market, which shows up as a one-hot vector. The goal is to make training more stable, which is often hard to do because financial data is noisy and non-stationary, which leads to policies that are not always consistent or reliable. Success will be measured using standard financial indicators like Profit and Loss (PnL), which is figured out as the cumulative sum of returns or the product of position size and next-period return, and the Sharpe Ratio (risk-adjusted returns), which measures returns earned over the risk-free rate per unit of volatility.

Logits-Based Distillation: Transformed Teacher Matching (TTM) and Weighted TTM (WTTM)

The main goal of this group of knowledge distillation methods is to move the teacher's policy by matching the probabilities or logits of its final result. The student agent learns what to do in a market situation by copying the teacher's established action distribution.

Transformed Teacher Matching (TTM) changes the usual method of matching logits. It uses a transforming power (which is mathematically the same as temperature scaling) on the teacher's output distribution and removes the temperature scaling from the student's logits. This change in architecture automatically adds a Rényi entropy regularization term to the student's learning goal. The goal is to make the student's policy less overly confident and smoother, which is thought to greatly improve the ability to generalize in financial environments that are naturally noisy and volatile.

The TTM framework is extended by Weighted Transformed Teacher Matching (WTTM), which adds an adaptive sample weighting mechanism. The smoothness of the power-transformed teacher's output distribution is what gives these weights their dynamic nature. This smart weighting helps the student focus on learning from the teacher's decisions that are less clear or certain. These kinds of decisions are often thought to have more "dark knowledge," which is information that isn't clearly stated in strict labels.

The main goal of these logits-based strategies is to give the student a better understanding of the teacher's decision boundary and how to adjust the confidence of its output, which will help them make better and more detailed trading decisions.

Feature-Based Distillation: Probabilistic Knowledge Transfer (PKT)

Unlike logits-based methods, feature-based distillation, as shown by Probabilistic Knowledge Transfer (PKT), focuses on distilling the teacher's feature representation or how they understand the market internally. The goal of PKT is to move the helpful information stored in the teacher's middle layers, not just the final action (like buy/sell/hold).

The main idea behind PKT is to match the probability distributions of data samples in the feature space of a chosen intermediate layer in both the teacher's and the student's networks. This process successfully moves the fundamental geometric structure and the complex relationships gained by the teacher's internal representations to the student model. This is a different form of guidance than output-level distillation. It makes sure that if two market states look similar to the teacher, they also look similar to the student, teaching them how to read and understand the market. The Cosine Similarity kernel is usually used to figure out the pairwise similarities between feature vectors, which then describe the probability distributions in conditions that are minimized via KL divergence.

Contrastive Representation Distillation (CRD)

This research mainly focuses on TTM/WTTM and PKT while also recognizing Contrastive Representation Distillation (CRD) as an advanced knowledge distillation method that matches the general goal of clarifying and understanding various forms of knowledge transfer. CRD stands out by maximizing a lower bound on the mutual information between the teacher's and the student's representations.

This goal makes the teacher's and student's models match the same input to similar representations (positive pairs) and different inputs to distant representations (negative pairs) in the same metric space. CRD can improve performance by letting the teacher's representation send a lot of information, such as detailed connections between the probabilities of different categories. This method looks at how to learn and pass on general relational knowledge about data points, which can be very helpful for understanding how complex markets work.

[VISUALISATION PLACEHOLDER: Type=Diagram, Title='Categorization of Knowledge Distillation Strategies', Data='Illustrates conceptual differences between Logit-Based (TTM/WTTM), Feature-Based (PKT), and Contrastive-Based (CRD) distillation, highlighting their primary knowledge transfer targets (output probabilities/policies, intermediate features/representations, and relational embeddings, respectively).']

Proposed Hybrid and Ensemble Mechanism

This dissertation also looks at hybrid methods that use both logits-based (TTM/WTTM) and feature-based (PKT) distillation signals because financial markets can be very hard to work with. This method is based on the idea that the best results in financial DRL may need both a strong internal model of market states and a policy that is less overly confident and more regulated.

Additionally, an online ensemble mechanism is used to improve the student agent's stability and generalizability. In this setup, several teacher models are trained at the same time along with the student agent. The individual extraction losses of these different teachers average to get a single value that represents their knowledge. A dynamic teacher selection process makes sure that the student is always led by the best and most stable teaching policies, which adjust to market changes and help students learn in a steady way.

1.5 - Dissertation Contributions

This section lists the main contributions of this dissertation, which jointly advance the use of Deep Reinforcement Learning (DRL) in the complex and changing field of automated financial trading.

Improved Stability and Performance of DRL Trading Agents: This dissertation presents empirical evidence showing that sophisticated Knowledge Distillation (KD) methodologies can significantly reduce the inherent training instability often encountered by DRL agents in volatile financial markets. When we train DRL models, performance can vary significantly from one run to the next. KD stabilizes the student's training process by giving it a structured learning signal from a teacher model or a group of teachers. This makes the training process more profitable and lowers risk-adjusted returns. The distillation process guides the student using gradients from a "trusted policy," resulting in more consistent and noise-resistant weight updates.

Comparative Analysis of Knowledge Distillation Philosophies: A main contribution is facilitating a direct and empirical comparison of various KD philosophies to systematically determine which form of knowledge transfer is most effective for DRL agents operating within a financial trading environment.

Logits-Based Distillation (Transformed Teacher Matching - TTM/WTTM): This method focuses on analyzing the teacher's final decision-making process or policy and telling the student "what to do." TTM sees temperature scaling as a transforming power, which automatically adds a Rényi entropy regularization term that makes student policies smoother and less certain. This improves generalization, especially in noisy financial environments. Weighted TTM (WTTM) improves this even further by using an adaptive sample grading system to give more weight to the teacher's outputs that are more helpful, which is often called "dark knowledge."

Feature-Based Distillation (Probabilistic Knowledge Transfer - PKT): This method focuses on extracting the teacher's internal understanding or representation of market data, teaching the student "how to interpret" the market. PKT achieves this by matching the probability distributions of pairwise similarities between data samples in the feature space of the middle layers. This successfully moves the geometric structure and the subtle relationships that the teacher's internal representations have learned.

Contrastive Representation Distillation (CRD): This method tries to move a lot of structural knowledge by maximizing the mutual information between the teacher's and student's representations. This encourages the alignment of positive pairs and the separation of negative pairs in a metric space. This method aims to transmit all the information within the teacher's representation, including subtle associations and higher-order output dependencies. This comparative analysis aims to determine whether a regularized policy (TTM), a replicated internal worldview (PKT), or comprehensive relational knowledge (CRD) provides superior resilience in financial trading.

Empirical Validation of Advanced KD Techniques: The dissertation provides empirical validation for the theoretical advantages of advanced KD methodologies, particularly TTM and PKT, in a complex, real-world financial trading environment. For TTM, this means confirming that it is a strong policy regulator that keeps the policy's entropy higher and more stable, which leads to better generalization. For PKT, this means showing that it works to keep the geometry of the teacher's feature space. The study rigorously assesses whether these theoretical benefits translate into measurable improvements in key financial indicators, including Sharpe Ratio and reduced maximum drawdown.

This study critically examines the potential synergies that may arise from integrating various knowledge distillation (KD) methodologies. This entails considering the confluence of both logits-based distillation signals (TTM/WTTM) and feature-based distillation signals (PKT) within a cohesive framework. The underlying hypothesis is that robust internal representations, when combined with well-regulated and regularized actions, can lead to superior and more stable performance in financial DRL.

Development of a Robust Framework for Automated Trading: A significant advance is the creation of a more reliable and useful DRL framework for automated financial trading. This includes establishing an online ensemble mechanism, facilitating dynamic knowledge transfer from a defined group of teacher models to the student during simultaneous training. This online and adaptive method is especially good for environments that change over time, such as financial markets. It gives the student agent strong, flexible, and adaptive guidance. The framework ends in a final combined loss function that combines the standard DRL loss with the average ensemble logits losses and feature-based distillation losses. This allows for full and synergistic guidance.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='Conceptual Overview of Thesis Contributions', Data='A high-level diagram showing the central theme (DRL in Finance), problem (instability), proposed solution (KD strategies), and key contributions (Enhanced Stability, Comparative Analysis, Hybrid/Ensemble, Framework Development).']

This part of the dissertation gives a short summary of the contents of each chapter and explains how the dissertation is structured.

Chapter 1: The Beginning

This chapter introduces the research field, highlighting the promise of Deep Reinforcement Learning (DRL) for automated financial transactions and emphasizing the significant challenge of instability in training in volatile markets such as cryptocurrencies. It encourages the exploration of Knowledge Distillation (KD) as a solution to these challenges. The chapter formally states the problem statement, explains the specific research questions regarding the use of Probabilistic Knowledge Transfer (PKT) and online combined KD, and provides a dissertation summary.

Chapter 2: Background and Previous Research

This chapter lays the foundation for research by defining the necessary theories. It provides an overview of financial markets, with an emphasis on what makes cryptocurrency trading different. It talks about the basics of Deep Reinforcement Learning (DRL), such as the Proximal Policy Optimisation (PPO) algorithm, and breaks down important Deep Learning architectures such as LSTMs (Long Short-Term Memory Recurrent Neural Networks). The chapter also mentions the basic ideas behind Knowledge Distillation. Following this background, it examines other research on the use of DRL in financial trading and previous work on Knowledge Distillation. It then highlights the specific gap that the proposed advanced KD techniques aim to fill.

Chapter 3: Proposed Method

This chapter explains the technical methods created for this dissertation. It shows the proposed DRL framework, which is based on the PPO algorithm, and analyzes in detail how it uses advanced Knowledge Distillation techniques (This chapter shows the specific methodology developed in this dissertation). This includes a detailed explanation of how to use Probabilistic Knowledge Transfer (PKT) from intermediate levels of the network (This section breaks down in detail how to use Probabilistic Knowledge Transfer (PKT)) and the exact steps for using online combined distillation. The chapter breaks down in detail the architecture of the neural network used and how the combined loss function was created for training.

Chapter 4: Testing the Results

This chapter describes the full framework for the empirical evaluation of the proposed method. It explains the exact cryptocurrency dataset used in the study, including the price-based features generated from the OHLCV data. We define the basic methods used for comparison, such as training without KD (Vanilla PPO) and using standard KD methods (Standard Offline Distillation, Online Distillation, Online PKT). The chapter describes the evaluation metrics used, emphasizing trading performance indicators (e.g., PnL, Sharpe Ratio, Maximum Drop) and measurable training stability indicators in several independent executions (Our approach achieved an annual Sharpe index of 2.68 in test data). There are also details on how to implement it, especially for PKT and the online ensemble.

Chapter 5: Discussion and Results

This chapter shows the actual results of the experiments conducted. It contains a quantitative comparison of the proposed advanced KD methods to established benchmarks, examining both training stability and trading performance (PnL) using cryptocurrency market data that has not been seen before. The chapter offers a complete analysis and interpretation of these results, examining their significance and linking the findings to the research questions defined in Chapter 1.

Chapter 6: Summary

This final chapter summarizes all the research work. It reiterates the main results regarding how well the proposed advanced Knowledge Distillation methods work to improve DRL-based cryptocurrency trading agents. The chapter reiterates the main points of the dissertation, gives brief answers to the research questions, mentions any problems with the current work, and suggests new areas of research in this area (The approach can be improved in future work with more computing power to conduct more experiences and better assess the approach). There are many hyperparameters to adjust in our environment, the agent, and the learning process. Finally, the proposed method can be combined and improved with other search methods, such as vision-inspired optimization, pooling strategy evolution, and so on.

#### 2.0 - An Overview of the Background and Related Work

Chapter 2 lays the foundation for the main ideas and provides a complete review of all relevant previous research that supports the methods and contributions of this dissertation. To fully understand the difficulties of using advanced Knowledge Distillation (KD) techniques such as Transformed Teacher Matching (TTM), Probabilistic Knowledge Transfer (PKT), and Inverse Representation Distillation (CRD) in Deep Reinforcement Learning (DRL) in the specific and difficult field of cryptocurrency trading, you must have a good understanding of these basic ideas.

This chapter begins by talking about the basics and ideas behind Artificial Intelligence, Machine Learning, and Deep Learning. It also talks about how important they are to recent technological advances. Following this, it breaks down in more detail the ideas behind Reinforcement Machine Learning (RL) and Deep Reinforcement Machine Learning (DRL). It focuses on DRL as a specific type of RL that uses deep neural networks to represent states and approximate functions, especially algorithms such as Proximity Policy Optimization (PPO) and architectures such as Long-Term and Short-Term Memory (LSTM). The use of DRL in financial markets and algorithmic trading is then examined, focusing on the difficulties arising from the noisy, non-stationary, and complex nature of this field, which often leads to instability in training, overtraining, and poor generalization in DRL agents.

To address the aforementioned challenges, this chapter presents the fundamental principles of Knowledge Distillation (KD) as a viable approach for reducing instability and improving performance. It examines various types of knowledge distillation (KD) from a philosophical perspective, such as logit-based distillation (such as TTM and Weighted TTM), which focuses on moving the teacher's policy and decision threshold, and feature-based distillation (specifically PKT), which attempts to move the structured internal understanding of the market by matching feature representations. The chapter will also talk about Natural Language Processing (NLP) techniques that are useful for analyzing financial sentiments in the context of advanced financial applications. These include Transformer models and domain adaptation strategies, which could help represent the agent's state and make trading decisions more stable.

#### 2.1.1 - A Look at Financial Markets

Financial markets are dynamic and complex places where trading agents do their jobs. In these markets, they trade a wide range of assets, such as stocks, bonds, currencies, and derivatives. The prices of these assets are mainly determined by supply and demand. These markets are constantly changing due to many different economic, political, and social factors, which makes it hard to make decisions. The main goal for participants is to make as much money as possible while also carefully managing the risks involved. This is made harder by the fact that asset values can change very quickly and without warning.

There are a few important things that make these markets what they are:

Noise and Volatility: Financial data often has a lot of randomness and noise in price changes, which makes it hard to tell the difference between real market signals and random changes. This is especially true in cryptocurrency markets, which are known for their high volatility and the presence of a lot of noise. A lot of fuss like this can make training Deep Reinforcement Learning (DRL) agents very unstable. Volatility is the amount of change in a series of trading prices over time, and it is usually measured by the standard deviation of returns.

Non-stationarity: The fundamental dynamics of financial markets, which include volatility and the interconnections between assets, are constantly in flux due to economic cycles, regime shifts, and external influences. This constant change means that financial data is highly dependent on time and goes against the assumption of stability on which many traditional Markov decision processes (MDPs) are based. This limits the usefulness of standard statistical methods.

Complexity: The market is a complex place with many moving parts and many people. The actions of a trading agent can have a big impact on others, especially when many agents are involved. This adds layers of complexity that are hard to predict or fully model.

Risk and Return: The idea that you must choose between risk and return is very important in financial markets. Investors usually don't like to lose money, so good trading strategies must take this risk into account and manage it.

Conventional modeling paradigms, typically organized as a two-stage "predict-then-trade" framework, face significant limitations in this complex and evolving landscape. A significant problem is that the goals of predictive models and investors don't always match up. For example, predictive models mainly want to lower prediction error, which is not the same as an investor's end goal of maximizing risk-adjusted profit (e.g., Sharpe Ratio). This basic gap often causes live trading to not work as well as it could. Furthermore, these methods often ignore important information by relying solely on predictions as input to the trading model and, most importantly, often fail to adequately consider real external constraints such as transaction costs, market liquidity, slippage, and market impact, or fail to effectively address the challenges of temporal credit assignment inherent in sequential financial decision-making.

Reinforcement Learning (RL), and particularly Deep Reinforcement Learning (DRL), are well-suited to overcome these problems. DRL optimizes the entire process simultaneously by combining the tasks of "prediction" and "portfolio construction" in one step. This ensures that the process perfectly matches the investor's overall goals. This integrated approach makes it easier to add important real-world constraints directly to the agent's decision-making process. Some of these constraints are: explicit transaction costs (such as commissions), taking into account market liquidity and slippage to ensure that orders are executed at the right prices, being aware of the market effect of large trading volumes, directly managing risk aversion through risk-adjusted performance functions (such as optimizing for the Sharpe ratio or minimizing maximum drawdown), and following trading rules (such as avoiding negative cash balances).

Because financial markets are noisy, non-stationary, and complex by nature, DRL agents need special training to become more stable and ensure that they operate reliably. A key problem that DRL frameworks must solve is how to balance searching for new strategies with using strategies that have already been proven effective.

[VISUALISATION PLACEHOLDER: Type=Diagram, Title='Characteristics of Financial Markets', Data='A diagram showing the main features: Noise, Non-stationarity, Complexity, Risk & Return, and how they affect DRL agents.']
[VISUALISATION PLACEHOLDER: Type=Flowchart, Title='Comparison: Two-Step vs. Unified DRL Trading', Data='A flowchart showing the 'Predict-then-Trade' model (Prediction -> Trading Action) versus the Unified DRL model (Direct Action from Environment/Observation), highlighting where constraints are incorporated in each.']

#### 2.1.2 - What makes cryptocurrency trading different

This part talks about the unique characteristics of cryptocurrency markets and how these make it difficult for investment agents to do their jobs, which means they need to use more advanced modeling techniques. Cryptocurrency exchanges differ from most traditional financial markets because they are always open and can be affected by a wide range of external factors. This makes them dynamic and interactive.

Some important things about cryptocurrency trading environments are:

High Volatility: The prices of cryptocurrencies are known for their quick and large changes, which offer traders both great opportunities and great risks (raw source, 256, 512, 546). Because prices change so quickly and there is so much noise in the market, it is hard to tell the difference between real market signals and random changes (raw source, 554). The standard deviation of returns is often used to measure volatility (raw source, 546). For example, the cryptocurrency pair ATOM/USDT has seen its instability reach 25%, but it has also fluctuated between 6% and 12% in other periods. It is very hard to make investment decisions when asset values are so unpredictable and change so quickly, especially in such volatile markets (raw source, 255, 256). Also, this natural volatility can cause DRL agents to take on more risks in their trades, which could lead to large financial losses even when they are trying to make more money.

24/7 Operation: Cryptocurrency exchanges differ from most traditional financial markets in that they are open all the time, 24 hours a day, seven days a week. This means that we must find trading strategies that can work nonstop. Decentralized blockchain technology makes it easier to transact by eliminating the need for intermediaries like banks. This makes transactions secure, transparent, and fast all over the world.

Cryptocurrency markets are very sensitive to news, sentiments on social media, changes in regulations, and a wide range of other external events. This high level of sensitivity means that market prices are not only affected by past price data but also by new information that is spread through news articles, social media posts, and public discussions. The market's mood, which is the overall disposition and attitude of investors, can have a big impact on buying and selling pressure. This can cause price changes that cannot be predicted simply by looking at past prices. The mood has a big impact on cryptocurrency markets, where stories and community involvement are important parts of how prices are determined. Adding sentiment analysis to Deep Reinforcement Learning (DRL) frameworks can greatly improve how a unit sees the market, giving it a more complete picture. This integration could also help keep price patterns from becoming too noisy. Studies show that models learn significantly faster and achieve greater profitability when sentiment data is accessible for cryptocurrencies, suggesting its potential as a more reliable predictor of future behavior compared to price data alone (source, 'Multi-source Financial Sentiment Analysis for Detecting Bitcoin Price Change Indications using Deep Learning.pdf' references).

In conclusion, these unique factors—high volatility, continuous operation, and strong sensitivity to external information and emotions—collectively create an extremely dynamic, complex, and interactive environment (raw source, 543, 547). This environment often causes a lot of instability when training DRL agents and makes their performance in real situations unpredictable. So, to succeed in these markets, we need to consider and create new modeling methods, such as DRL, that can quickly adjust to changes, effectively use various sources of information, and manage the noise and non-stationarity that come with them.

#### 2.2.1 - Basic Ideas behind Reinforcement Learning (RL)

Reinforcement Learning (RL) is a powerful subfield of machine learning that offers a computational framework for an agent to acquire an optimal action policy through repeated trial-and-error experiences in a dynamic environment. Reinforcement Learning (RL) works by finding the best long-term strategy. Its main goal is to get the most cumulative reward from interactions, not to learn from static data sets or pre-set labels.

The agent-environment interaction loop is the most important part of the reinforcement learning methodology. This process happens in separate time steps. At each step of time t:

The environment first gives the agent an image of its current state, which is called the state (St). This state quickly summarizes all the important things the agent sees.

The agent then chooses an action (At) from its list of possible actions based on what it thinks is happening. These actions can be either discrete (such as buying, holding, or selling a set amount) or continuous (such as fractional allocations or portfolio weights).

After a time step, the environment changes to a new state (St+1) and gives the agent a scalar feedback signal, the reward (Rt+1), as a direct result of its chosen action. This reward gives a number that shows how desirable or good the agent's action was at that particular time.
This continuous cycle of observation, action, and reward feedback continues until a final state is reached (in episodic tasks) or indefinitely (in continuous tasks).

The agent's policy (π) is a formal way to show its strategy, which is a set of rules or a map for choosing actions. A policy connects the observed states to certain actions or a probability distribution over the actions, which allows for both deterministic and stochastic behaviors. The main goal of Reinforcement Learning (RL) is to find the best policy (π)* that gives the highest expected sum of discounted future rewards, which is often called the "return."

Almost all reinforcement learning algorithms depend on value functions. They help understand how good it is for the agent to be in a certain state or to do something specific in that state. This evaluation is based on the expected set of future rewards that can be collected.

The state value function (Vπ(s)) tells you how much you can expect to get if you follow the policy π from the state s.

The action value function (Qπ(s,a)) calculates the expected return for choosing action an in state s and then following policy π.

The exploration versus exploitation trade-off is a key problem that comes up in Reinforcement Machine Learning. To succeed in the long run, the agent must always balance two goals:

Exploitation: Using what it already knows to choose actions that are thought to give the best rewards right now.

Exploration: Trying new actions that may not be the best to find new, better strategies or to learn more about how the environment works. It is very important to find the right balance between these two things for strong and flexible learning.

The Markov Decision Process (MDP) is the formal mathematical framework most often used to model problems in Reinforcement Learning. A Markov Decision Process (MDP) is formally characterized by a tuple (S, A, P, R, γ), where S represents the state space, A stands for the action space, P denotes the state transition probability function, R indicates the reward function, and γ refers to the discount factor. The Markov Property is an important prerequisite for defining an MDP. It says that the future state only depends on the current state and the action taken, not on the whole history of states and actions that came before. If a Reinforcement Learning (RL) problem follows this rule, it can be clearly defined and studied using the MDP framework.

[VISUALISATION PLACEHOLDER: Type=Diagram, Title='Agent-Environment Interaction Loop in RL', Data='A cyclical diagram showing Agent, Environment, State, Action, Reward, and New State, with arrows showing the flow of information.']


#### **2.2.2 - DRL Algorithms and Architectures**

**Deep Reinforcement Learning (DRL)** represents a cutting-edge field that integrates **Reinforcement Learning (RL)** principles with the powerful representational capabilities of **deep neural networks (DNNs)**. This synergistic combination enables RL agents to effectively handle **complex, high-dimensional, and often unstructured state spaces**, such as raw market data or visual inputs, without the need for manual feature engineering. The overarching objective of DRL is to discover an optimal policy for sequential decision-making problems by learning through trial and error, interacting with an environment to maximize cumulative reward over the long term.

DRL algorithms can be broadly categorized into several main approaches:
*   **Value-based methods**, which focus on learning to estimate a value function (e.g., Q-values) to predict the expected future reward of states or state-action pairs (e.g., Deep Q-Network - DQN).
*   **Policy-based methods**, which directly parameterize and optimize the agent's policy, mapping states to actions or probability distributions over actions (e.g., Policy Gradient, Trust Region methods).
*   **Actor-Critic methods**, which judiciously combine elements of both value-based and policy-based approaches, often leading to enhanced stability and efficiency.

### Actor-Critic Paradigm

**Actor-Critic (AC)** algorithms constitute a prominent class of DRL methods that leverage **two distinct neural networks** working in tandem: the **Actor** and the **Critic**.
*   The **Actor** network is primarily responsible for determining the agent's **actions** and formulating the system's **policy**. It receives the current state of the environment as input and computes the action as its output, frequently as a probability distribution over possible actions.
*   The **Critic** network evaluates the actions proposed or taken by the Actor. It receives the current state (and often the Actor's action) as input and outputs an estimate of the discounted future reward or state value (e.g., V(s) or Q(s,a)). This evaluation by the Critic provides a crucial learning signal, typically in the form of an advantage estimate, which guides the Actor in refining and improving its policy. Both the Actor and Critic components are commonly implemented as neural networks, and their optimization is typically performed jointly.

### Proximal Policy Optimization (PPO)

**Proximal Policy Optimization (PPO)** is a specific and widely adopted Actor-Critic algorithm, highly regarded for its **stability** and robust performance in numerous DRL problems. It is considered a leading contender among contemporary DRL algorithms. PPO achieves its characteristic stability by meticulously limiting the magnitude of policy updates at each step, thereby preventing excessively large changes that could destabilize performance, especially within inherently noisy environments like financial markets. PPO's core mechanism involves optimizing a **clipped surrogate objective function**. This clipped objective ensures that policy updates remain within a reasonable "proximal" region of the previous policy, preventing drastic deviations and contributing to enhanced training stability. The algorithm also allows for multiple epochs of minibatch updates on sampled data, which contributes to its favorable sample efficiency compared to other on-policy methods.

### Long Short-Term Memory (LSTM) Networks

**Long Short-Term Memory (LSTM) networks** are a crucial type of **recurrent neural network (RNN)** architecture widely employed in DRL, particularly for tasks involving sequential data. LSTMs are exceptionally well-suited for processing **sequential data** and, critically, for capturing **long-term temporal dependencies**. This capability is indispensable for applications such as financial trading, where market dynamics and patterns evolve over time, and the retention of past information is vital for informed current decision-making. Unlike simpler RNNs, LSTMs effectively address challenges such as the vanishing gradient problem through their unique internal **gating mechanisms** (comprising input, forget, and output gates). These gates meticulously control the flow of information into and out of the recurrent cell state, enabling LSTMs to store and retrieve relevant historical information over extended periods. Consequently, in DRL for financial trading, LSTMs are frequently integrated into the network architecture of both the Actor and Critic components to robustly process sequential input features, such as historical price data and technical indicators.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='Actor-Critic DRL Architecture', Data='Diagram showing separate Actor and Critic networks receiving state input, Actor outputting action, Critic outputting value/advantage, and the learning signal flow.']
[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='LSTM Cell Gates', Data='Conceptual diagram illustrating the input, forget, and output gates controlling information flow within an LSTM cell.']
**Figure 1: Agent-Environment Interaction in Reinforcement Learning** [Adapted from 479]

```
+----------------+       Action (at)       +---------------+
|     Agent      |<-----------------------|  Environment  |
|  (Policy π,    |                       | (States S,    |
|   Value V)     |------State (st)------->|  Rewards R,   |
|                |<-----Reward (rt)-------| Transitions P)|
+----------------+       Next State (st+1)  +---------------+
```

_(This diagram illustrates the continuous loop where the agent takes an action in a given state, the environment responds with a new state and a reward, and this feedback loop drives the agent's learning process_5..._.)_
**Figure 2: Categories of Reinforcement Learning Methods** [Adapted from 133]

```
                                +---------------------------+
                                |  Reinforcement Learning   |
                                +---------------------------+
                                            |
                  +-------------------------------------------------+
                  |                         |                         |
        +---------+---------+   +-----------+-----------+   +---------+---------+
        | Value-Based Methods |   | Policy-Based Methods  |   | Actor-Critic Methods|
        +---------+---------+   +-----------+-----------+   +---------+---------+
            |                             |                         |
        (e.g., Q-learning, DQN,       (e.g., Policy Gradient,     (Combine both, often
        SARSA) [5, 9]              Trust Region, Evolution)  for continuous action
                                      [5, 10]                spaces) [5, 10]
```

_(This diagram visually organizes the primary approaches in RL, highlighting how Actor-Critic methods bridge value-based and policy-based techniques_58_.)_
**Figure 3: Actor-Critic Architecture** [Adapted from 322, 338]

```
+-----------------------------------+
|         Environment State (st)    |
+-----------------------------------+
                 |
                 V
+-----------------------------------+
|       Shared Feature Extractor    |  (e.g., LSTM Network)
+-----------------------------------+
                 |
     +-----------+-----------+
     |                       |
     V                       V
+-----------------+     +-----------------+
|     Actor       |     |     Critic      |
| (Policy π(a|s;θ))|     | (Value V(s;w)   |
|                 |     |  or Q(s,a;w))   |
+-----------------+     +-----------------+
     |                       |
     V                       V
+-----------------+     +-----------------+
| Action (at)     |     | Value Estimate  |
| (Probability    |     | (for learning)  |
| Distribution)   |     +-----------------+
+-----------------+
```

_(This diagram illustrates the dual network structure of an Actor-Critic model, showing how the Actor determines actions and the Critic evaluates them to provide a learning signal_12..._.)_
**Figure 4: LSTM Network Architecture** [Adapted from 279]

```
Input (xt) ---> [Forget Gate] --- (Cell State Ct-1) ---> (Cell State Ct) ---> [Output Gate] --> Hidden State (ht)
              |                                     ^
              V                                     |
             [Input Gate] <-------------------------(Candidate Cell State)
```

_(This conceptual diagram illustrates the core components of an LSTM cell, including input, forget, and output gates that regulate information flow into and out of the cell state, enabling the network to learn long-term dependencies in sequential data_2427_.)_
**Figure 5: Example of Financial Candlestick Data** [Adapted from 514]

```
   High (H)
     |
   Open (O) --+
     |        |
   Close (C)--+   Body (Open to Close range)
     |
   Low (L)
   Time
   ---------------->
    (Time Periods)
```

_(This diagram illustrates the key components of a candlestick, which summarizes price movements (Open, High, Low, Close) over a specific time period. Sequences of these candlesticks form the temporal data processed by networks like LSTMs_2829_.)_

#### 2.3.1 - General Concept of Knowledge Distillation

Knowledge Distillation (KD) is a machine learning technique designed to transfer learned knowledge from a 'teacher' model to a 'student' model. This process aims to achieve various goals, including model compression, improving student performance, and regularizing the student's training process.

Here are some mathematical equations and diagrams that help illustrate the general concept of Knowledge Distillation:

### Mathematical Equations

The core of Knowledge Distillation often involves training a smaller student model to mimic the behavior or representations of a larger, more complex teacher model.

**1. Soft Target Generation (Classical Logit-Based Distillation)** A fundamental aspect of many KD methods is the use of "soft targets" generated from the teacher's output, often involving a **temperature scaling parameter** ($T$) applied to the logits (raw outputs before final activation). This temperature makes the probability distribution smoother, revealing nuances and relative similarities between different classes or actions as perceived by the teacher, which is typically lost with "hard" one-hot labels.

- **Teacher's Soft Probability Distribution** ($q(a|s)$): $$ q(a|s) = \frac{\exp(y^{(t)}(a|s)/T)}{\sum_{a'} \exp(y^{(t)}(a'|s)/T)} $$ where $y^{(t)}(a|s)$ are the logits (raw outputs) of the teacher model for a given state $s$ and action $a$.
    
- **Student's Soft Probability Distribution** ($p(a|s)$): $$ p(a|s) = \frac{\exp(y^{(s)}(a|s)/T)}{\sum_{a'} \exp(y^{(s)}(a'|s)/T)} $$ where $y^{(s)}(a|s)$ are the logits of the student model.
    

**2. Logit Matching Loss (Classical KD Objective)** The student model is typically trained to minimize the divergence between its soft probabilities and the teacher's soft probabilities. The **Kullback-Leibler (KL) Divergence** is commonly used for this purpose.

- **KL Divergence Loss** ($L_{log}$): $$ L_{log} = \mathbb{E}_{s} \left[ D_{KL}(q(\cdot|s) || p(\cdot|s)) \right] $$ This loss encourages the student's output distribution to match that of the teacher, transferring the teacher's learned decision boundaries and confidence levels. Alternatively, it can be expressed as a cross-entropy loss: $$ L_{log} = - \frac{1}{N} \sum_{s \in S} \sum_{j=0}^{k} q(\alpha_j|s) \log(p(\alpha_j|s)) $$ where $S$ is a set of $N$ states and $k$ is the number of available actions.

**3. Transformed Teacher Matching (TTM) Loss** A variant of KD, TTM, modifies the standard approach by dropping temperature scaling on the student's side, while retaining it for the teacher. This reinterprets the teacher's temperature-scaled output as a **power-transformed probability distribution**. A key insight is that TTM's objective function inherently includes a **Rényi entropy term**, which acts as an additional regularizer, promoting smoother and less confident student policies and leading to improved generalization.

- **TTM Loss** ($L_{TTM}$): $$ L_{TTM} = H(y, q) + \beta D(p_T^t || q) $$ Here, $H(y,q)$ is the cross-entropy loss with the hard labels $y$, $D(p_T^t || q)$ is the divergence between the teacher's power-transformed distribution $p_T^t$ and the student's unscaled output distribution $q$, and $\beta$ is a balancing weight. The Rényi entropy is defined as: $$ H_\alpha(X) = \frac{1}{1− \alpha} \log \sum_{i=1}^{n} p_i^\alpha $$ where $\alpha$ is the order of Rényi entropy.

**4. Probabilistic Knowledge Transfer (PKT) Loss** PKT is a feature-based distillation method that focuses on transferring knowledge from **intermediate layers** of the teacher network. It aims to match the probability distributions of pairwise similarities between data samples within the feature space of the teacher and student, thereby transferring the **local geometric structure** learned by the teacher.

- **Teacher's Conditional Probability Distribution** ($p_{i|j}$): $$ p_{i|j} = \frac{K(x_i, x_j; 2\sigma^2)}{\sum_{k \neq j} K(x_i, x_k; 2\sigma^2)} $$ where $x_i$ and $x_j$ are feature vectors from the teacher's intermediate layer, and $K$ is a symmetric kernel (e.g., **Cosine Similarity kernel**).
- **Student's Conditional Probability Distribution** ($q_{i|j}$): $$ q_{i|j} = \frac{K(x_i, x_j; 2\sigma^2)}{\sum_{k \neq j} K(x_i, x_k; 2\sigma^2)} $$ Similarly, for the student's corresponding intermediate layer.
- **PKT Loss** ($L_{pkt}$): $$ L_{pkt} = \sum_{i} \sum_{j \neq i} p_{i|j} \log \left( \frac{p_{i|j}}{q_{i|j}} \right) $$ This loss minimizes the KL divergence between these conditional probability distributions, encouraging the student's feature space to align with the teacher's.

**5. Combined Loss Function (for DRL with KD)** In Deep Reinforcement Learning (DRL) frameworks, KD is often integrated by combining the standard RL loss ($L_{RL}$) with one or more distillation losses.

- **General Combined Loss:** $$ L = L_{RL} + \beta \cdot L_{dist} $$ Here, $L_{dist}$ is the distillation loss, and $\beta$ is a weighting hyperparameter.
- **Combined Loss with Multiple Distillation Types (e.g., Logit and PKT):** $$ L = L_{RL} + \beta_1 \cdot L_{log} + \beta_2 \cdot L_{pkt} $$ where $\beta_1$ and $\beta_2$ are weighting factors controlling the contribution of each distillation term.

**6. Ensemble Distillation Loss Aggregation** When multiple teacher models (an ensemble) are used, their individual distillation losses are typically averaged to provide a more robust learning signal to the student.

- **Ensemble Distillation Loss** ($L_{dist}$): $$ L_{dist} = \frac{1}{N} \sum_{i=1}^{N} L_{dist_i} $$ where $N$ is the number of teachers in the ensemble and $L_{dist_i}$ is the distillation loss calculated with respect to teacher $i$.

### Diagrams / Visualization Concepts

Several sources describe visual representations that would be beneficial for understanding Knowledge Distillation:

- **General Knowledge Distillation Overview:** A fundamental diagram would show a **Teacher model** and a **Student model**. An arrow from the Teacher's output would lead to the Student's loss function (representing **Logit Matching**). Another arrow from an intermediate layer of the Teacher would lead to the Student's loss (representing **PKT**). The diagram would also illustrate a **Combined loss function** ($L = L_{RL} + \beta \cdot L_{dist}$) that incorporates these distillation terms along with the standard Reinforcement Learning loss.
    
- **Conceptual Distinction: Logit-Based vs. Feature-Based Distillation:** A visualization could illustrate the two primary philosophies:
    
    - **Logit-Based Distillation** (e.g., TTM/WTTM) focuses on the **teacher's final decision** or "outcome" ("what to do" / "trading policy"), resembling an apprentice chef mimicking the master's final dish plating.
    - **Feature-Based Distillation** (e.g., PKT) focuses on the **teacher's internal thought process** or "process and understanding" ("how to think" / "state representation"), akin to an apprentice learning the master's "Mise en Place" – how they organize ingredients and understand relationships before cooking.
- **Probabilistic Knowledge Transfer (PKT) Diagram:** A diagram specifically for PKT would depict how the **knowledge of the teacher model is modeled using a probability distribution**, and then this knowledge is transferred to the student by minimizing the divergence between the probability distributions of the teacher and the student. This shows the matching of data distributions in the feature space rather than raw representations.
    
- **Distillation Settings (e.g., from Contrastive Representation Distillation):** A broader conceptual diagram could illustrate different scenarios where KD is applied, such as:
    
    - **Compressing a large model** into a smaller one.
    - **Transferring knowledge across different sensory modalities** (e.g., from an image processing network to a sound or depth processing network).
    - **Distilling an ensemble of networks** into a single, more efficient network.
- **Visualization of "Dark Knowledge" / Correlations:** While not a core concept definition, a useful visualization shows the **correlation matrices between class logits** of a teacher network and how different distillation methods (e.g., KL divergence, contrastive objectives) help the student network capture these correlations, which represent the "dark knowledge" not obvious from hard labels. This demonstrates the subtle, nuanced knowledge being transferred.


####  2.3.2 - Logit-Based Distillation (Transformed Teacher Matching - TTM / Weighted TTM - WTTM)

This section details advanced **logit-based distillation methods**, specifically Transformed Teacher Matching (TTM) and Weighted TTM (WTTM), which propose sophisticated modifications to the traditional application of temperature scaling in knowledge distillation (KD). These methods are crucial for transferring the teacher's final decision-making policy and its inherent confidence calibration to the student model.

### Core Mechanism of Logit-Based Distillation

Logit-based distillation methods fundamentally focus on transferring knowledge encoded in the teacher's **final output layer**, particularly its action probability distributions or logits. These methods typically employ "**soft targets**" generated from the teacher's output, often involving a **temperature scaling parameter** ($T$) applied to the logits (raw outputs before final activation). This temperature smooths the probability distribution, revealing subtle nuances and relative similarities between different classes or actions as perceived by the teacher, information that is typically lost when relying solely on "hard" one-hot labels.

The soft probability distribution $q(a|s)$ produced by the teacher's logits $y^{(t)}(a|s)$ is calculated using a softmax function with temperature $T$:
$$ q(a|s) = \frac{\exp(y^{(t)}(a|s)/T)}{\sum_{a'} \exp(y^{(t)}(a'|s)/T)} $$
Similarly, in standard KD, the soft probability distribution $p(a|s)$ for the student's logits $y^{(s)}(a|s)$ is also traditionally calculated using the same temperature $T$:
$$ p(a|s) = \frac{\exp(y^{(s)}(a|s)/T)}{\sum_{a'} \exp(y^{(s)}(a'|s)/T)} $$
The loss function commonly utilized for this purpose is the Kullback-Leibler (KL) divergence between the student's and teacher's soft probabilities:
$$ L_{log} = \mathbb{E}_{s} \left[ D_{KL}(q(\cdot|s) || p(\cdot|s)) \right] $$
This loss encourages the student's output distribution to match that of the teacher, thereby transferring the teacher's learned decision boundaries and confidence levels.

### Transformed Teacher Matching (TTM)

**Transformed Teacher Matching (TTM) introduces a pivotal modification by entirely dropping temperature scaling on the student side**, while meticulously retaining it for the teacher's output. This novel approach reinterprets the teacher's temperature-scaled output as applying a **power transform** (with exponent $\gamma = 1/T$) directly to the teacher's output probability distribution. The student is then trained to precisely match this power-transformed teacher distribution.

A key **theoretical insight** of TTM is that its objective function inherently incorporates a **Rényi entropy term**. This Rényi entropy term acts as an **additional regularizer** compared to standard KD, which actively promotes smoother, less overconfident student policies. The definition of Rényi entropy $H_\alpha(X)$ for a probability distribution $p$ is:
$$ H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_{i=1}^n p_i^\alpha $$
This inherent regularization is hypothesized to lead to **trained students with superior generalization capabilities** than those trained with original KD, particularly in highly noisy and unpredictable environments such as financial markets. The TTM loss can be expressed as:
$$ L_{TTM} = H(y, q) + \beta D(p_T^t || q) $$
Here, $H(y,q)$ is the cross-entropy loss with the hard labels $y$, $D(p_T^t || q)$ is the divergence between the teacher's power-transformed distribution $p_T^t$ and the student's unscaled output distribution $q$, and $\beta$ is a balancing weight.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='TTM vs. Standard KD Loss Comparison', Data='A diagram showing the difference in loss functions. Standard KD: KL(Student_softmax(T) || Teacher_softmax(T)). TTM: KL(Student_softmax(1) || Teacher_softmax(T)), highlighting the implicit Rényi entropy regularization.']
[VISUALIZATION PLACEHOLDER: Type=Plot, Title='Effect of Power Transform on Probability Distributions', Data='A small plot illustrating power functions (e.g., $f(x)=x^\gamma$ for different $\gamma$ values on a probability distribution), showing how higher $\gamma$ (lower T) makes distributions sharper and lower $\gamma$ (higher T) makes them smoother.']

### Weighted TTM (WTTM)

**Weighted TTM (WTTM) is introduced as a significant extension of TTM, specifically designed for further performance enhancement**. It integrates a **sample-adaptive weighting coefficient** into the TTM objective function. This sophisticated **weighting mechanism** is dynamically determined by quantifying the smoothness of the teacher's power-transformed probability distribution for each individual sample, often using a power sum $U_\gamma(p)$.

The underlying intuition behind this adaptive weighting is to direct the student's learning focus towards matching the teacher's outputs for samples where the teacher exhibits **less confidence or where its probability distribution is smoother**. Such samples are often considered to contain richer "**dark knowledge**"—subtle, implicit information that is not immediately apparent from crisp, confident predictions. By prioritizing these more informative teacher outputs, WTTM is claimed to provide **further improvement over TTM**, achieving state-of-the-art accuracy in certain domains. WTTM ultimately leads to even smoother student output distributions compared to TTM, thereby facilitating more accurate transformed teacher matching.

### Knowledge Focus

In summary, both TTM and WTTM primarily concentrate on distilling knowledge from the teacher's final output distributions (logits/action probabilities). Their overarching aim is to transfer the teacher's refined **decision boundary** and **output confidence calibration**, essentially instructing the student on "what to do" in a given market situation by precisely mimicking the teacher's optimal action distribution.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='WTTM Adaptive Weighting Concept', Data='A diagram illustrating WTTM. Show input samples, teacher output distributions (some sharp, some smooth). Arrows from smoother distributions indicating higher weights for student learning, representing focus on "dark knowledge."']
[VISUALIZATION PLACEHOLDER: Type=Heatmap, Title='Correlation Matrices Illustrating Dark Knowledge Transfer', Data='Heatmaps comparing correlation matrices of class logits for a Teacher model vs. a Student model (trained with/without KD). Show how KD, especially WTTM, helps the student capture subtle correlations and relative probabilities between non-dominant classes/actions, demonstrating the transfer of "dark knowledge" not obvious from hard labels.']


we need to be extra refined in this chapter if you are reacginf output length break it donw to as many pages as possible nd let me type next to continue 
#### 2.3.3 - Feature-Based Distillation (Probabilistic Knowledge Transfer - PKT)

**Probabilistic Knowledge Transfer (PKT)** is a sophisticated **feature-based distillation method** designed to transfer knowledge between neural networks by focusing on the **geometric structure and relationships** present within the **internal representations** (feature spaces) of the models. Unlike methods that primarily match final output logits, PKT aims to convey the teacher's "internal thought process" or "state representation," teaching the student **"how to think"** about input data. This process is connected to maintaining the teacher's **quadratic mutual information (QMI)** between feature representations and (potentially unknown) label annotations.

The mechanism of PKT involves modeling the interactions between data samples in the feature space as probability distributions and then minimizing the divergence between the teacher's and student's respective distributions.

### Mathematical Mechanism of PKT

The transfer of knowledge through PKT is achieved via the following steps and mathematical formulations:

1.  **Feature Extraction from Intermediate Layers:**
    Feature vectors are extracted from a **chosen intermediate layer** in both the teacher and student networks. In the context of Deep Reinforcement Learning (DRL) frameworks discussed in the sources, this specific layer is often identified as the **linear layer located after the LSTM layer and immediately before the actor and critic heads**.

2.  **Pairwise Similarity Calculation (Kernel Function):**
    To quantify the relationships between data samples, the **similarity between pairs of these extracted feature vectors** is computed using a **symmetric kernel function**, denoted as $K(a,b; \sigma^2)$.
    For the specific implementations described in the sources, the **Cosine Similarity kernel** is widely used due to its effectiveness and its advantage of **not requiring explicit bandwidth tuning** (unlike, for example, the Gaussian kernel which requires careful tuning of its width $\sigma$).
    The Cosine Similarity kernel is defined as:
    $$ K_{cosine}(a, b) = \frac{1}{2} \left( \frac{a \cdot b}{||a||_2 ||b||_2} + 1 \right) $$
    This kernel function measures the cosine of the angle between two vectors and scales the result to the range $[0, 1]$.

3.  **Conditional Probability Distributions:**
    The kernel similarities are then transformed into **conditional probability distributions**. These distributions express the **affinity or likelihood of one data sample being a "neighbor" to another** in the feature space.
    *   For the **teacher model** (where $x_i, x_j$ are feature vectors from the teacher's intermediate layer):
        $$ p_{i|j} = \frac{K(x_i, x_j; 2\sigma^2_t)}{\sum_{k \neq j} K(x_k, x_j; 2\sigma^2_t)} $$
    *   For the **student model** (where $y_i, y_j$ are feature vectors from the student's corresponding intermediate layer):
        $$ q_{i|j} = \frac{K(y_i, y_j; 2\sigma^2_s)}{\sum_{k \neq j} K(y_k, y_j; 2\sigma^2_s)} $$
    In these formulas, $N$ is the total number of objects in the transfer set, and $\sigma^2_t$ and $\sigma^2_s$ represent the bandwidths for the teacher and student kernels, respectively. While different bandwidths can be used, the cosine similarity kernel often simplifies this, as it is considered "angular" and does not explicitly require $\sigma$ tuning. It's notable that **class labels are not required** to minimize the divergence between these distributions, making PKT an unsupervised knowledge transfer method in this specific aspect.

4.  **KL Divergence Loss Function:**
    The difference between the teacher's ($p_{i|j}$) and student's ($q_{i|j}$) feature-space probability distributions is quantified using the **Kullback-Leibler (KL) divergence loss**, denoted as $L_{pkt}$. The student model is trained to minimize this loss:
    $$ L_{pkt} = \sum_{i} \sum_{j \neq i} p_{i|j} \log \left( \frac{p_{i|j}}{q_{i|j}} \right) $$
    Minimizing $L_{pkt}$ encourages the student's feature space to align with the teacher's, preserving the **local geometry** and relationships between data points. The asymmetric nature of KL divergence means it **gives higher weight to minimizing the divergence for neighboring pairs of points instead of distant ones**, prioritizing the maintenance of local neighborhoods over recreating the global geometry of the entire feature space, which offers greater flexibility during student training.

### Knowledge Focus and Advantages of PKT

PKT's primary focus is on transferring the **topology and structural relationships** captured by the teacher's deep representations. If the teacher effectively clusters similar market situations and separates dissimilar ones, PKT aims to transfer this learned topology to the student. This form of knowledge transfer is hypothesized to improve the student's generalization capabilities, especially in noisy and non-stationary environments like financial markets, by helping the student learn features robust to transient noise.

Key advantages of PKT include:
*   **Dimensionality Agnostic**: It can directly transfer knowledge even when the output dimensionalities of the teacher and student networks do not match, without needing additional dimensionality-reduction layers.
*   **Representation Learning**: PKT is effective for various representation and metric learning tasks, distinguishing it from many traditional knowledge distillation methods tailored primarily for classification.
*   **Transfer from Handcrafted Features**: It enables the transfer of knowledge from handcrafted feature extractors (e.g., HoG features) into neural networks, allowing for the exploitation of large amounts of unlabeled training data.
*   **Cross-Modal Transfer**: PKT supports cross-modal knowledge transfer, such as distilling knowledge from textual modality into visual modality representations.
*   **Reduced Hyperparameter Tuning**: It simplifies the distillation process by not requiring meticulous hyperparameter tuning (e.g., softmax temperature).
*   **Incorporating Domain Knowledge**: The probability distributions can be enhanced or estimated using other information sources, including domain knowledge or supervised information. An example is Supervised PKT (S-PKT), where class labels directly construct a probability distribution combined with the teacher's features.
*   **Multi-Layer Transfer (Ladder PKT)**: PKT can be extended to transfer knowledge from multiple layers of neural networks through a ladder-based scheme, guiding the knowledge transfer process more effectively.

### Integration in DRL Framework

In a DRL framework, PKT serves as a **feature-based distillation term** within the student agent's overall combined loss function. This total loss typically integrates the standard DRL objective (e.g., PPO clipped surrogate objective, $L_{RL}$) with distillation losses. The overall training objective is defined as:
$$ L = L_{RL} + \beta_1 \cdot L_{log} + \beta_2 \cdot L_{pkt} $$
Here, $L_{log}$ represents a logit-based distillation loss (such as TTM or WTTM), and $\beta_1$ and $\beta_2$ are weighting factors that control the respective contributions of the logit-based and feature-based distillation terms. For many implementations, $\beta_1$ and $\beta_2$ are typically set to 1. This combined guidance aims to enhance the student agent's training stability and performance in complex, non-stationary environments like financial markets.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='Probabilistic Knowledge Transfer (PKT) Mechanism', Data='A diagram illustrating PKT. Show two feature spaces (Teacher and Student). Within each space, depict data points and arrows representing pairwise similarities (e.g., using Cosine Similarity). Show a mapping or loss function that minimizes the divergence between the probability distributions derived from these similarities in the teacher\'s and student\'s feature spaces. Emphasize the transfer of \'local geometric structure\'.']


### 2.3.4 - Contrastive-Based Distillation (CRD)

**Contrastive Representation Distillation (CRD)** is an advanced Knowledge Distillation (KD) method specifically designed to transfer rich representational knowledge between neural networks. It addresses certain limitations of conventional KD, which primarily minimizes the Kullback-Leibler (KL) divergence between probabilistic outputs and may overlook crucial, subtle structural knowledge embedded within the teacher network's internal representations.

#### Core Idea of CRD

The fundamental principle of CRD is rooted in **contrastive learning**. This paradigm aims to learn representations where data points that are semantically similar are mapped to **close positions in a metric space ("positive pairs")**, while data points that are semantically dissimilar are mapped to **distant positions ("negative pairs")**. In the context of knowledge distillation, CRD trains a student network to:
*   **Pull closer** its representation of an input (`fS(x)`) to the teacher's representation of the *same* input (`fT(x)`). These constitute "positive pairs" (Tian et al., 2019).
*   **Push apart** its representation of an input (`fS(x_i)`) from the teacher's representation of *different*, randomly chosen inputs (`fT(x_j)`). These form "negative pairs" (Tian et al., 2019).

This approach is typically applied to representations extracted from an intermediate layer, specifically the penultimate layer before the final output logits.

#### Objective of CRD

CRD's objective is rigorously formulated to **maximize a lower-bound to the mutual information (MI) between the teacher and student representations** (Tian et al., 2019). By maximizing the term `Eq(T,S|C=1) log q(C = 1|T, S)` with respect to the student network's parameters, CRD effectively increases this lower bound on mutual information. This is achieved by framing the problem as a binary classification task where a "critic" model `h` learns to distinguish whether a pair of teacher and student representations `(T, S)` originated from their joint distribution (`C=1`, indicating they are representations of the same input) or from the product of their marginal distributions (`C=0`, indicating they are representations of different inputs).

The mathematical formulation for the objective to maximize mutual information `I(T;S)` is given by:
$$ I(T ;S) \geq \log(N) + \mathbb{E}_{q(T,S|C=1)} \left[ \log q(C = 1|T, S) \right] $$
where `N` relates to the number of negative samples (Tian et al., 2019). In practice, `N` can be effectively made very large by utilizing a memory buffer to store latent features from previous batches, thereby avoiding the need for excessively large batch sizes during training (Tian et al., 2019). The optimal critic `h*` in this formulation aims to accurately estimate `q(C=1|T, S)`.

#### Benefits of CRD

The design of CRD confers several significant advantages, making it a highly effective distillation strategy:
*   **Capturing Structural and Relational Knowledge:** Unlike original KD, which often treats output dimensions as conditionally independent, CRD explicitly aims to capture **correlations and higher-order output dependencies** within the teacher's representations. This enables it to transfer more comprehensive **"structural"** or **"dark knowledge"**—nuanced information about data relationships not explicitly present in hard labels. Empirical evidence demonstrates its ability to significantly match the correlation structure in the teacher's logits, which translates into reduced error rates (Tian et al., 2019, Figure 2).
*   **Superior Performance:** CRD has consistently demonstrated superior performance compared to other cutting-edge distillation methods, including the original KD. It has achieved state-of-the-art results across various knowledge transfer tasks, such as model compression, cross-modal transfer, and ensemble distillation. For instance, it showed an impressive average relative improvement of 57% over original KD (Tian et al., 2019).
*   **Improved Generalization:** The contrastive objective is hypothesized to improve generalization capabilities by more comprehensively transferring all the information encoded in the teacher's representation. This effectively regularizes the learning problem, leading to student models that not only match but sometimes **even outperform the teacher network when combined with standard KD** (Tian et al., 2019).
*   **Dimensionality Agnostic:** CRD proves effective across very different teacher and student architectures, overcoming a limitation of some other methods that necessitate highly similar network structures or spatial resolutions of feature maps (Tian et al., 2019).

#### Knowledge Focus

Fundamentally, CRD focuses on transferring **relational knowledge** about the data. By ensuring that the student learns to perceive the relationships between different data points in its feature space in a manner consistent with the teacher's understanding, CRD offers a distinct form of knowledge transfer compared to logit-based distillation, which primarily focuses on the teacher's final decision-making policy and its confidence.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='Contrastive Representation Distillation (CRD) Principle', Data='A conceptual diagram illustrating CRD. Show an input sample being processed by both Teacher and Student. Depict "positive pairs" (same input, teacher/student representations pulled closer) and "negative pairs" (different inputs, teacher/student representations pushed apart) in a metric space. Highlight the role of a "critic" distinguishing between joint/marginal distributions.']
[VISUALIZATION PLACEHOLDER: Type=Heatmap, Title='CRD Effect on Logit Correlation Matrices', Data='Recreate/adapt Figure 2 from the source (Tian et al., 2019, page 7/16), showing heatmaps of the differences in correlation matrices between class logits for a teacher and various student models (e.g., vanilla, KD, CRD). Emphasize how CRD results in smaller differences, indicating better transfer of "dark knowledge" (correlations between logits).']
### 2.3.5 - Offline Distillation Methods

This section explains the **offline knowledge distillation paradigm**, a foundational approach in the field of knowledge transfer.

### Core Idea

**Offline knowledge distillation (KD)** is fundamentally a **two-stage process**. In this paradigm, **teacher models are first fully trained and then frozen**. These pre-trained, static teacher(s) subsequently provide guidance to a student model, allowing for efficient knowledge transfer without requiring the teacher to continue learning or adapt.

### Stage 1 (Teacher Training)

In the initial stage, **one or more teacher models are trained to convergence** on the specific task. In the context of Deep Reinforcement Learning (DRL), these teachers are often trained using robust algorithms like the **Proximal Policy Optimization (PPO) algorithm**. Once these teacher models achieve their desired performance, their parameters are fixed and are **not updated during the student's subsequent training process**, rendering them static knowledge providers.

### Stage 2 (Student Distillation)

In the second stage, the **pre-trained, static teacher(s) provide a structured learning signal to the student agent**. The student learns by minimizing a **combined loss function** that integrates both its own task objective (e.g., the standard DRL loss, $L_{RL}$) and a distillation loss derived from the fixed teacher(s).

This distillation loss can be derived from different parts of the teacher model:
*   **Logit-Based Distillation (Classical KD):** This approach, notably introduced by Hinton et al. (2015), involves transferring knowledge from the teacher's final output layer. It typically aims to match the **soft probability distributions** (logits transformed by a softmax function with a temperature parameter $T$) of the teacher's output. The objective is for the student to mimic the teacher's learned decision boundaries and confidence levels. The loss ($L_{log}$) is commonly the Kullback-Leibler (KL) divergence between the student's and teacher's soft probabilities.
*   **Feature-Based Distillation (e.g., Probabilistic Knowledge Transfer - PKT):** This method focuses on transferring knowledge from the **intermediate layers** of the teacher network. For **Probabilistic Knowledge Transfer (PKT)**, the core idea is to match the **probability distribution of pairwise similarities** between data samples within the feature space of a chosen intermediate layer in both the teacher and student networks. This is often achieved using a kernel (e.g., Cosine Similarity) to calculate conditional probabilities between samples and then minimizing their KL divergence ($L_{pkt}$). This aims to transfer the geometric structure and relationships learned by the teacher's internal representations.

When an **ensemble of teacher models** is used in offline distillation, the knowledge transferred from them is typically aggregated by **averaging their individual distillation losses**. The student's overall training objective then combines the DRL loss ($L_{RL}$) with these averaged distillation terms, weighted by hyperparameters ($\beta_1$, $\beta_2$). The final combined loss function is structured as:
$$ L = L_{RL} + \beta_1 \cdot L_{log} + \beta_2 \cdot L_{pkt} \quad \text{(Eq 3.14)} $$

### Contrast with Online KD and Bridging the Gap

Offline distillation is fundamentally distinct from **online knowledge distillation** because in the latter, **teachers and students are trained simultaneously**. Online KD is characterized by a continuous, dynamic learning process, whereas offline KD is a batch-oriented, two-phase process. This simultaneous training in online methods allows models to learn from each other dynamically and often simplifies the overall training process by avoiding the separate, two-phase training and additional hyperparameter tuning typically required for offline KD. Consequently, online KD methods often **obtain better performance than offline ones**.

While offline distillation can effectively leverage existing high-performance models, it frequently demonstrates **inferior performance compared to online methods** due to this static nature. Research suggests that the **essential factor contributing to this performance gap lies in the absence of "reversed distillation" from student to teacher** in the traditional offline paradigm. Offline distillation can achieve competitive performance by **fine-tuning the pre-trained teacher to adapt to the student through such reversed distillation**, though this process can be computationally costly.

To bridge this performance gap, the **SHAdow KnowlEdge transfer (SHAKE)** framework has been proposed. SHAKE introduces an **extra shadow head on the backbone of the student network to mimic the pre-trained teacher's predictions, effectively acting as a "proxy teacher"**. This proxy teacher then performs **bidirectional distillation with the student "on the fly"**. This mechanism allows SHAKE to:
*   Dynamically update the student-aware proxy teacher with knowledge from the pre-trained model.
*   Optimize the computational costs associated with augmented reversed distillation.
*   Effectively **reduce the teacher-student capability gap** through this reversed distillation.
*   Combine the advantages of both offline (leveraging pre-trained teachers, alleviating unstable optimization) and online (dynamic, mutual learning) KD methods.
*   Achieve a favorable trade-off between accuracy and training budget.
*   It can also be extended to scenarios involving multiple teacher models by employing multiple shadow heads.

### Rationale and Benefits

Offline knowledge distillation is a **standard and widely adopted approach** in machine learning and DRL, particularly valuable for **leveraging existing high-performance models as teachers** without requiring their retraining. The benefits of knowledge distillation, including its offline variants, are multifaceted:
*   **Model Compression:** It facilitates the training of smaller, faster student models that can achieve performance comparable to that of larger, more computationally expensive teachers.
*   **Improved Student Performance:** Students can attain better accuracy, generalization, or robustness than if trained solely on original data, benefiting from the richer, more nuanced "dark knowledge" embedded in the teacher's output or internal representations. Empirical studies frequently show that "Offline Distillation" yields high performance.
*   **Regularization:** The distillation loss term itself acts as a powerful regularizer, which inherently **improves the generalization capabilities** of the student model. This is especially crucial in noisy financial environments to **mitigate overfitting** to historical data.
*   **Stabilizing Training:** A well-trained, static teacher provides a consistent and more stable learning signal, guiding student updates and actively **preventing oscillations or divergence during training**. This directly **addresses the inherent instability of DRL agents** in complex, non-stationary environments like financial markets.
*   **Reduced Data Reliance:** It is possible to utilize an already trained teacher agent on a dataset, reducing the need for direct access to the entire, potentially private or sensitive, dataset during student training.

[VISUALIZATION PLACEHOLDER: Type=Flowchart, Title='Offline Knowledge Distillation Pipeline', Data='A two-stage flowchart. Stage 1: Teacher Training (Environment/Data -> Teacher Model). Stage 2: Student Distillation (Teacher Model (frozen) + Environment/Data -> Student Model).']

This section rationalizes the strategic decision to primarily focus on **offline ensemble knowledge distillation** within this thesis, while acknowledging the significant potential of other advanced online and hybrid distillation methodologies for future research. This approach is rooted in the current empirical findings and the necessity to establish a robust foundation for addressing the inherent challenges of Deep Reinforcement Learning (DRL) in financial markets.

### Rationale for Prioritizing Offline Ensemble Distillation

The core problem addressed by this thesis is the **training instability and inconsistent performance** of DRL agents in volatile financial environments, particularly cryptocurrency markets. This instability leads to high variance in outcomes, overfitting, poor generalization, and ultimately undermines confidence in the practical application of DRL for automated trading. Knowledge Distillation (KD) is proposed as a promising solution to mitigate these issues.

The primary rationale for prioritizing **offline ensemble distillation** in this current work is its proven effectiveness in directly addressing this core problem:

1.  **Direct Mitigation of Instability:** Offline ensemble distillation has empirically demonstrated its capability to **significantly mitigate the inherent training instability** of DRL agents in the complex financial domain. By providing a structured and consistent learning signal from pre-trained, stable teacher models, this paradigm effectively guides the student's learning process.
2.  **Empirical Validation and Robust Performance:** In preliminary experiments conducted for this thesis (as documented in internal "Thesis First Pass 1" and "Thesis Draft" analyses), the offline distillation implementation yielded a mean Profit and Loss (PnL) of **17.36%**. This figure not only represents a substantial improvement over the baseline DRL agent (which often exhibits lower and more variable performance) but also closely aligns with the target performance (e.g., 18.66% PnL) observed in related state-of-the-art work (Moustakidis_Vasileios_Thesis.pdf). Crucially, this method demonstrated a high degree of stability, which is a primary objective of this research.
3.  **Foundation for Advanced Research:** Focusing on a demonstrably effective and stable method like offline ensemble distillation provides a **strong, reliable baseline and a clear reference point** for future, more complex investigations. It allows for a systematic and controlled evaluation of the incremental benefits introduced by more dynamic or intricate KD components. The two-stage nature (teacher pre-training, then student distillation) also offers a more controlled experimental setup for initial studies.

### Rationale for Future Work: Exploring Other Online and Hybrid Methods

While offline ensemble distillation provides a robust solution to the immediate problem of DRL instability, other advanced online and hybrid distillation methods offer significant potential for further performance enhancement, adaptability, and theoretical exploration. These are considered promising avenues for future research due to their unique characteristics and the complexities they introduce, which warrant dedicated investigation building upon the stable foundation established by offline methods:

1.  **General Online Distillation:** This paradigm involves the simultaneous training of teacher(s) and student(s), potentially offering dynamic adaptation and reduced overall computational overhead by avoiding a separate teacher pre-training phase. However, initial implementations of "Online Distillation" in the current research demonstrated a **higher standard deviation (1.93%)** and a slightly lower mean PnL (16.84%) compared to offline distillation. This suggests that while promising, its stability benefits might require further optimization in the financial context.
2.  **Online Probabilistic Knowledge Transfer (PKT):** This feature-based method aims to transfer the geometric structure of the teacher's internal representations. Although conceptually powerful for learning "how to think," the "Online PKT" implementation in preliminary analyses exhibited the **highest standard deviation (2.30%)** and a lower PnL (16.43%) than the baseline. This indicates potential instability and highlights the need for substantial future work to stabilize and optimize this specific approach for DRL in financial markets.
3.  **Hybrid Knowledge Distillation Strategies:** The proposed hybrid method, combining logit-based (e.g., TTM/WTTM) and feature-based (PKT) distillation within an online ensemble framework, aims to leverage the synergistic benefits of both policy regularization and structured internal representations. While theoretically compelling, early results for this "Proposed Method" showed performance close to the baseline (16.94% PnL) but with a **considerably higher standard deviation (1.40%)** compared to the target stability from the Moustakidis thesis (0.02% standard deviation). Fully realizing the benefits and stability of this complex hybrid approach necessitates dedicated future research and refinement.
4.  **Transformed Teacher Matching (TTM) & Weighted TTM (WTTM):** These logit-based methods introduce a Rényi entropy regularizer to promote smoother, less overconfident policies. Their theoretical advantages in improving generalization in noisy environments make them excellent candidates for detailed future empirical investigation into their specific impact on financial DRL performance and stability.
5.  **SHAKE (SHAdow KnowlEdge Transfer):** This framework proposes to bridge the gap between offline and online KD by using a "shadow head" as a proxy teacher for bidirectional distillation. This method's potential to reduce the teacher-student capability gap and optimize reversed distillation costs, offering a favorable trade-off between accuracy and training budget, positions it as a highly efficient and promising avenue for future exploration.
6.  **Diversity-Driven Knowledge Distillation:** This approach involves training teachers on diverse data subsets to create a "diverse curriculum" for the student, thereby improving generalization and reducing reliance on direct access to private datasets. Its potential for building more generalizable financial DRL agents makes it a compelling direction for future work.
7.  **Deep Mutual Learning (DML):** DML is an online KD method enabling two-way knowledge transfer. While it can outperform standard KD, its training curve can exhibit dynamic oscillation, which poses a significant challenge in highly volatile financial environments. This dynamic aspect makes DML a prime candidate for future dedicated study on stability and optimization within the financial domain.
8.  **Centralized Training for Decentralized Execution:** This paradigm, commonly used in Multi-Agent Reinforcement Learning (MARL), where a centralized critic guides decentralized actors, is a critical online multi-model approach in interactive environments like financial markets with multiple traders. While not a direct "ensemble distillation" technique for single-agent DRL, it represents a valuable direction for extending the research to multi-agent financial scenarios.

In conclusion, by establishing a strong, stable foundation with offline ensemble distillation, this thesis systematically addresses its primary research problem. This strategic focus then allows for the rigorous and dedicated exploration of more dynamic and complex online and hybrid methods as key directions for future research, building incrementally upon the established success and insights.



### 2.4 - Related Work and Research Gap

This section reviews the existing literature on Deep Reinforcement Learning (DRL) applications in financial markets and the evolving field of Knowledge Distillation (KD). Its primary purpose is to contextualize the current research by identifying the persistent challenges faced by DRL in this domain and articulating the specific research gap that this thesis aims to address.

### Deep Reinforcement Learning in Finance

Deep Reinforcement Learning has emerged as a **prominent paradigm for developing automated trading strategies**, particularly within dynamic and volatile environments like financial and cryptocurrency markets. DRL agents are capable of learning complex trading policies directly through interaction with market simulations, adapting their behavior based on received rewards to optimize financial outcomes. This approach offers significant advantages by combining the financial asset price "prediction" step and the "allocation" step into a unified process, bypassing the need for explicit, supervised labels often required by traditional machine learning methods. Various DRL algorithms have been explored for financial applications, including **value-based methods** like Q-learning and Deep Q-Networks (DQN) and its variants (Double DQN, Dueling DQN), and **policy gradient methods**, particularly **Actor-Critic algorithms** such as PPO, A2C/A3C, Deep Deterministic Policy Gradient (DDPG), Soft Actor-Critic (SAC), and TD3. For instance, PPO has achieved state-of-the-art results for agents in financial trading due to its balance between exploration and exploitation and its ability to handle continuous actions. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are often integrated into DRL architectures to capture temporal dependencies and extract higher-level features from noisy financial time-series data. Sentiment analysis of news releases and technical indicators have also been incorporated into the state representation to improve agent performance.

Despite its potential, the successful application of DRL in the financial domain is significantly impeded by several **inherent challenges**:

*   **Training Instability and High Performance Variance:** Financial markets are characterized by significant **noise**, **non-stationarity** (where market dynamics change over time), and overall **complexity**. These factors often lead to **substantial instability during the DRL training process**, resulting in learned policies that may exhibit **inconsistent performance** and **lack robustness** across different training runs or when encountering unseen market conditions. This unreliability significantly undermines confidence in the resulting models and hinders their practical application.
*   **Overfitting and Poor Generalization:** The noisy and non-stationary nature of financial data makes DRL agents susceptible to overfitting to historical data, leading to **poor generalization** to future market conditions.
*   **"Deadly Triad" Issue:** The combination of off-policy learning, non-linear function approximation (deep neural networks), and bootstrapping (using estimated future values) can lead to **instability and divergence** in DRL algorithms.
*   **Sample Efficiency and Reward Design:** Training DRL agents typically requires a large number of interactions, which can be challenging to obtain in financial markets. Furthermore, while profits can be modeled as rewards, the immediate Profit and Loss (PnL) signal is often **sparse and noisy**, making it difficult for the agent to reliably associate actions with long-term profitable outcomes.

### Knowledge Distillation in RL and Finance

**Knowledge Distillation (KD)**, a technique where a "student" model learns from a more capable "teacher" model or an ensemble of teachers, provides a structured learning signal and offers a **promising strategy to mitigate these issues and improve overall performance**. The general idea of KD involves transferring knowledge from a larger, more complex model (teacher) to a smaller, more efficient one (student) without significant loss of generalization power.

In DRL contexts, KD has been applied to various problems, including:
*   **Improving training reliability and stability** for DRL trading agents, especially in noisy financial environments.
*   Transferring knowledge between agents trained on different tasks.
*   Avoiding the forgetting of older experiences in continual learning settings.
*   Enhancing generalization, particularly by using diversified ensembles of teachers.
*   Refining recommendation lists by boosting top positions in recommender systems.

Two philosophically distinct families of KD approaches are primarily investigated:
*   **Logit-Based Distillation (e.g., Transformed Teacher Matching - TTM, Weighted TTM - WTTM):** This method focuses on transferring the teacher's **final decision boundary** and **output confidence calibration** by matching output probabilities (logits). TTM introduces Rényi entropy regularization for smoother, less overconfident policies, while WTTM uses sample-adaptive weighting to prioritize "dark knowledge" from ambiguous teacher outputs.
*   **Feature-Based Distillation (e.g., Probabilistic Knowledge Transfer - PKT):** This method focuses on transferring the teacher's **internal thought process** and **state representation** by matching the probability distributions of data samples in the feature space of intermediate network layers. This aims to transfer the geometric structure and relationships learned by the teacher's internal representations.

KD can also be categorized into **online versus offline distillation**. While traditional KD is often "offline" (teachers are pre-trained and static), "online" KD involves simultaneous training of teachers and students, which can be more efficient and adaptive, particularly in non-stationary environments. Diversity-driven KD has also been explored in financial contexts, where teachers are trained on diverse subsets of data to improve student generalization.

### Identifying the Research Gap

Despite the growing body of work in DRL for financial trading and the application of KD in various RL contexts, a critical challenge remains: **the high performance variance and inherent instability of DRL agents in stochastic and non-stationary financial markets significantly undermine their reliability and widespread adoption**.

Prior research has acknowledged the instability of financial DRL and explored some basic KD applications. However, there is a distinct **lack of a systematic, comparative study of advanced and distinct KD philosophies—specifically, logit-based (such as TTM/WTTM) versus feature-based (such as PKT)—to comprehensively address DRL's stability and performance issues in the challenging domain of cryptocurrency trading, utilizing solely engineered price-based features**. While some prior work points towards combining online and probabilistic KD with DRL for cryptocurrency trading, a thorough empirical investigation of which *type* of knowledge transfer is most effective is still needed.

Therefore, the central goal of this thesis is to **fill this specific research gap** by rigorously investigating whether sophisticated KD techniques, including Probabilistic Knowledge Transfer from intermediate network layers and online ensemble distillation approaches, can effectively serve as powerful stabilization mechanisms and lead to measurably superior trading outcomes in this challenging domain. This research seeks to determine which **type of knowledge transfer** (i.e., mimicking a refined trading policy output or learning to perceive and structure market states similarly to an expert) is most beneficial, and to empirically validate these advanced methods for achieving more stable and profitable DRL agents in volatile cryptocurrency markets.

[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='DRL Challenges in Financial Markets', Data='A conceptual diagram illustrating the challenges: Noise, Non-stationarity, Complexity, Overfitting, Training Instability, High Variance.']
[VISUALIZATION PLACEHOLDER: Type=Diagram, Title='Knowledge Distillation for DRL Stabilization', Data='A conceptual diagram showing how different KD types (Logit-based, Feature-based) provide different 'guidance signals' to a DRL agent to counter market challenges.']
