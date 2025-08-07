

**Thesis Objective:** To investigate and enhance the performance, stability, and generalization of Deep Reinforcement Learning (DRL) agents for financial trading by systematically comparing and potentially combining distinct families of Knowledge Distillation (KD) techniques: logit-based distillation (specifically Transformed Teacher Matching - TTM, and Weighted TTM - WTTM) and feature-space distillation (Probabilistic Knowledge Transfer - PKT).

### 1. Introduction and Motivation

My research addresses the inherent challenges in applying DRL to financial markets, which are characterized by significant noise, non-stationarity, and complexity. These factors often lead to DRL agents exhibiting training instability, overfitting, and poor generalization to unseen market conditions. Knowledge Distillation, where a "student" agent learns from a more capable "teacher" agent or an ensemble, offers a promising strategy to mitigate these issues.

The core of this thesis is to conduct a comparative study of two distinct distillation philosophies to understand what _type_ of knowledge—refined output decision boundaries or structured internal representations—is most beneficial for a student agent to learn in the financial domain. This investigation will be carried out within the `MarketExperimentDistillation` framework, which is designed for training and evaluating such agents.

### 2. Core Distillation Approaches Under Investigation

My investigation will focus on comparing and contrasting the following distillation methodologies:
![[The-differences-between-feature-based-a-and-logit-based-b-distillation-approaches.png]]
- **Logit-Based Distillation ( what TTM/WTTM build upon):**
    
    - **What it does:** This method focuses on the **final decision** of the teacher. The student looks at the teacher's output probabilities (e.g., "70% chance to buy, 20% to hold, 10% to sell") and tries to make its own output probabilities match.
    - **Analogy:** This is like an apprentice chef learning by watching the master chef's final plating. The apprentice tries to make their final dish look exactly like the master's. They are focused on the **outcome**.
    - **In your scope:** The goal is to transfer the teacher's **trading policy**. The student learns _what to do_ in a given market situation by mimicking the teacher's action distribution.
- **Feature-Based Distillation :**
    
    - **What it does:** This method focuses on the teacher's **internal thought process**. It doesn't look at the final buy/sell decision. Instead, it looks at the hidden layers of the teacher's neural network to understand how the teacher _represents_ or _understands_ the market data. It then tries to make its own internal representations have the same structure.
    - **Analogy:** This is like the apprentice chef learning the master's "Mise en Place" — how they chop the vegetables, organize their ingredients, and understand the relationships between different flavors _before_ they even start cooking. The focus is on the **process and understanding**.
    - **In your scope:** The goal is to transfer the teacher's **state representation**. The student learns _how to see and interpret_ the market like the teacher does. It ensures that if two market situations look similar to the teacher internally, they also look similar to the student

**A. Logit-Based Distillation: Transformed Teacher Matching (TTM) & Weighted TTM (WTTM)**

These methods operate at the **output level** of the teacher network, refining traditional logit-matching KD.

- **Transformed Teacher Matching (TTM):**
    
    - **Core Idea:** TTM modifies standard KD by removing temperature scaling from the student's logits while retaining it for the teacher. This reinterprets the teacher's temperature-scaled output as a power-transformed probability distribution.
        
    - **Theoretical Insight:** This approach implicitly introduces a **Rényi entropy regularization** term into the student's learning objective. This regularization encourages the student to develop a smoother, less overconfident policy, which is hypothesized to improve generalization, particularly in noisy environments like financial markets.
        
- **Weighted Transformed Teacher Matching (WTTM):**
    
    - **Core Idea:** WTTM extends TTM by incorporating a **sample-adaptive weighting mechanism**. The weights are derived from the smoothness of the teacher's power-transformed output distribution (quantified by the power sum `U_γ(p)`).
        
    - **Rationale:** Instances where the teacher's output is less peaked (i.e., the teacher is less certain, or the decision is more ambiguous) might contain richer, more nuanced "dark knowledge." WTTM guides the student to prioritize matching these more informative teacher outputs.
        
- **Knowledge Focus:** TTM/WTTM aim to transfer a refined understanding of the teacher's **decision boundary** and **output confidence calibration**.
    
- **Implementation:** Within the `MarketExperimentDistillation` framework, the `_calculate_ttm_loss` and `_calculate_wttm_loss` methods will implement these, using student and teacher logits.
    

**B. Feature-Space Distillation: Probabilistic Knowledge Transfer (PKT)**

This approach operates on the **intermediate representations (features)** learned by the networks.

- **Core Idea:** Instead of matching raw feature vectors, PKT focuses on matching the **probability distribution of pairwise similarities** between data samples within the feature space of the teacher and student.
    
- **Mechanism:**
    
    1. Extract feature vectors from teacher and student models.
        
    2. For each model, construct a matrix of pairwise affinities (e.g., using cosine similarity or an RBF kernel).
        
    3. Normalize these affinity matrices to form conditional probability distributions (representing the likelihood of samples being "neighbors" in the feature space).
        
    4. Train the student to minimize a divergence metric (e.g., KL divergence) between its feature-space probability distribution and that of the teacher.
        
- **Knowledge Focus:** PKT encourages the student to learn a feature space that preserves the **local geometric structure** of the teacher's feature space. If the teacher has learned to effectively cluster similar market states and separate dissimilar ones, PKT aims to transfer this learned topology.
    
- **Advantages:** Capable of handling knowledge transfer between layers of different dimensionalities and effective for representation learning.
    
- **Implementation:** The `prob_loss` function within the `MarketExperimentDistillation` framework (used via `feature_distillation_weight * final_feature_distill_loss`) will embody this concept, ideally using kernel-based probability matching as described in the PKT research.
    

### 3. Rationale for Comparing TTM/WTTM with PKT in Financial Markets

This comparative study is scientifically valuable for several key reasons:

1. **Disentangling Knowledge Types & Levels of Abstraction:**
    
    - TTM/WTTM transfer knowledge about the teacher's final output mapping (state-to-action probabilities) and its confidence.
        
    - PKT transfers knowledge about the teacher's internal representation of states and the structural relationships it has learned.
        
    - **Key Question:** For financial DRL, is it more beneficial to mimic a refined version of the teacher's trading policy output, or to learn to perceive and structure market states similarly to the teacher?
        
2. **Addressing Unique Financial Data Characteristics:**
    
    - **Noise and Non-stationarity:**
        
        - The regularization from TTM/WTTM may lead to student policies less sensitive to spurious correlations or minor input perturbations.
            
        - PKT, by enforcing a consistent feature space geometry learned from (potentially more stable) teachers, might help the student learn features robust to transient noise.
            
    - **Generalization:**
        
        - TTM/WTTM aim for generalization via smoother policies.
            
        - PKT aims for generalization by ensuring the student learns essential relationships between market states.
            
3. **Exploring Complementarity and Synergies:**
    
    - These approaches are not mutually exclusive. PKT could help the student build robust internal state representations, and TTM/WTTM could then guide the mapping of these features to well-calibrated actions. The `MarketExperimentDistillation` framework is designed to test such combinations (e.g., `distillation_mode = 'TTM_PKT'`).
        
4. **Empirical Validation of Theoretical Benefits:**
    
    - The TTM/WTTM paper highlights improved generalization and the role of Rényi entropy. The PKT paper discusses preserving feature space geometry. This thesis will empirically test if these translate to tangible gains (e.g., improved Sharpe ratio, reduced drawdown, stable PnL) in financial DRL.
        
5. **Leveraging Teacher Ensembles:**
    
    - TTM/WTTM can average the (transformed) decision policies of an ensemble, leading to a robust target policy.
        
    - PKT can average the feature space geometries, allowing the student to learn common structural insights.
        

### 4. Experimental Design and Methodology

The `MarketExperimentDistillation` class will serve as the primary framework.

- **Baselines:**
    
    - Student trained alone (no distillation).
        
    - Standard Knowledge Distillation (KD).
        
- **Comparisons:**
    
    - TTM vs. Baselines.
        
    - WTTM vs. Baselines and vs. TTM.
        
    - PKT vs. Baselines.
        
    - Direct comparison: Best TTM/WTTM variant vs. PKT.
        
    - Combined strategies: TTM+PKT, WTTM+PKT vs. individual methods and baselines.
        
- **Teacher Ensemble:** The impact of the teacher ensemble (size, selection strategy) will be considered.
    
- **Self-Distillation:** The framework also supports exploring self-distillation in conjunction with these methods.
    

### 5. Key Metrics for Evaluation

A comprehensive evaluation will involve:

- **A. Financial Performance Metrics:** Cumulative PnL, Annualized Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), Calmar Ratio, Win Rate, Profit Factor, Average PnL per Trade, Number of Trades, Average Holding Period.
    
- **B. Training Stability and Robustness Metrics:** Mean and Standard Deviation of key financial metrics across multiple random seeds, convergence speed.
    
- **C. Distillation-Specific Diagnostic Metrics:**
    
    - For TTM/WTTM: Average Shannon Entropy of student's output logits (`q`), Average KL Divergence `D(p_T_teacher || q_student)`.
        
    - For PKT: PKT Loss value (`final_feature_distill_loss`) convergence.
        
    - Standard PPO training metrics (policy loss, value loss, policy entropy).
        



### **Executive Summary: The Core Difference**

The fundamental reason for the performance difference lies in **what is being distilled**.

*   **Transformed Teacher Matching (TTM)** distills the teacher's **policy**. It teaches the student *what to do*.
*   **Probabilistic Knowledge Transfer (PKT)** distills the teacher's **feature representation**. It teaches the student *how to think*.

In a noisy and non-stationary domain like financial markets, teaching a student a **regularized, cautious policy (TTM)** is a far more robust strategy for generalization than teaching it to perfectly mimic a **potentially overfitted internal worldview (PKT)**.

---

### **The Analogy: The Veteran Trader and the Apprentice**

Imagine your pre-trained teacher is a veteran trader with years of experience. You want this veteran to train a new apprentice (the student model). You have two ways to do this:

1.  **The PKT Method (Teaching the "Map"):** You hook up a machine to the veteran's brain and copy their entire neural map of the market—every connection, every heuristic, every gut feeling. The apprentice now has a perfect replica of the veteran's brain.
2.  **The TTM Method (Teaching the "Rules"):** You ask the veteran to write down their trading rules. But you add a condition: "Explain your rules, but for every decision, also explain why you *might* have considered the other options." The apprentice learns the veteran's decisions but also learns about the uncertainty and the alternatives, making them inherently more cautious.

Now, imagine the market regime changes slightly (which it always does).

*   The PKT apprentice, with the veteran's exact brain map, is brittle. That map was perfectly tuned for the old market. When new patterns appear, the map is wrong, and the apprentice fails catastrophically. It has overfitted to the veteran's specific experiences.
*   The TTM apprentice, who learned the rules along with the inherent uncertainties, is more robust. Their cautious, regularized policy is not overly dependent on the specific noise of the old market. They can adapt and continue to perform well.

---

### **Deep Dive: Technical Breakdown**

Let's analyze how this plays out in the code and the research.

#### **1. Transformed Teacher Matching (TTM): The Policy Regularizer**

*   **What it Distills:** The final `logits` of the teacher model. This is the direct precursor to the action probabilities—it is the model's policy.
*   **Core Mechanism:** The key is the asymmetric use of a transform (equivalent to temperature scaling). By "softening" the teacher's probability distribution, it creates a less opinionated target. For example, instead of the teacher saying "Action A is 95% correct, B is 3%, C is 2%," the softened target might say "Action A is 70% correct, B is 20%, C is 10%."
*   **The Theoretical "Why" (The Secret Sauce):** The ICLR 2024 paper proves that this process is mathematically equivalent to adding a **Rényi entropy regularizer** to the student's learning objective.
    *   **Regularization Prevents Overfitting:** Entropy regularization is a classic technique in machine learning. It penalizes the model for being too certain. By forcing the student to maintain some uncertainty (higher entropy) in its own policy, it prevents the student from latching onto spurious correlations in the training data.
    *   **Result:** The student learns a smoother, more robust policy that generalizes better to the unseen test data, which has different noise characteristics. **Your TTM agent's high test PnL is direct evidence of successful regularization.**

#### **2. Probabilistic Knowledge Transfer (PKT): The Feature-Space Mimic**

*   **What it Distills:** The intermediate `features` of the teacher model (the output of the LSTM/base layers, before the final action head).
*   **Core Mechanism:** PKT calculates a similarity matrix between all pairs of samples in a batch, based on their feature representations. It then uses a KL divergence loss to force the student's similarity matrix to match the teacher's. In essence, it says: "If the teacher model sees state `s_i` and state `s_j` as being very similar, the student model must also learn a feature representation where `s_i` and `s_j` are very similar."
*   **The Theoretical "Why" (The Hidden Trap):** The goal is to transfer the "geometry" of the teacher's learned representation space.
    *   **The Flaw is Overfitting:** The teacher learned this geometry based *only* on the training data. This geometry inevitably encodes not just the true, underlying signals of the market but also the **noise, biases, and specific regime characteristics of the training period.**
    *   **Result:** PKT forces the student to meticulously copy this entire, potentially flawed, representation. The student becomes an expert on the training set's specific noise profile. When presented with the test set, which has a different noise profile and market regime, the student's overfitted feature representation is no longer effective. **Your PKT agent's low test PnL is direct evidence of this feature-level overfitting.**

### **Conclusion: Why the Same Teachers Yield Different Results**

The teachers are the same, but TTM and PKT are using them for fundamentally different purposes.

*   **TTM uses the teachers as a source of a *regularized policy signal*.** It effectively filters the teachers' output, keeping the general decision-making trend while adding a layer of protective uncertainty.
*   **PKT uses the teachers as a source of a *detailed feature blueprint*.** It attempts to copy the blueprint exactly, warts and all, including all the biases learned from the training data.

Your experimental results are not an anomaly; they are a powerful validation of modern distillation theory. For noisy, non-stationary domains like finance, methods that promote **policy regularization (like TTM)** will almost always outperform methods based on **strict representation mimicry (like PKT)** when using teachers trained on the same domain.


####  **Financial Performance Analysis**

This is the ground truth: which agent is a better trader? We are comparing the methods on the following industry-standard, risk-adjusted metrics:

- **Sharpe Ratio:** This is our primary metric. It answers the question, "How much return did the agent generate for each unit of risk (volatility) it took on?" A higher Sharpe Ratio is unequivocally better and is the standard for comparing trading strategies.
    
- **Sortino Ratio:** A crucial refinement of the Sharpe Ratio. It only penalizes for "bad" volatility (downside risk), ignoring "good" volatility (upside swings). This is superior for evaluating strategies that may have asymmetric returns.
    
- **Calmar Ratio:** This measures return relative to the worst-case scenario (Annualized Return / Maximum Drawdown). It directly answers the question, "Is the return worth the biggest potential loss I might have to endure?" This is critical for risk management.
    
- **Maximum Drawdown:** The single largest peak-to-trough loss. This metric quantifies the "pain" an investor would have experienced and is a key indicator of risk. We compare this to see which agent is less prone to catastrophic losses.
    

**How to Explain:** "Professor, we are no longer just looking at profit. We are rigorously comparing the **risk-adjusted performance**. Our system automatically calculates the Sharpe, Sortino, and Calmar ratios, allowing us to prove which agent is not just more profitable, but also more stable and efficient."

#### **2. RL Dynamics Analysis**

This is the most important part of our academic investigation. It allows us to look inside the "black box" of the training process and understand why the agents behave differently.

- **Policy Entropy (The Key to TTM vs. PKT):** This is our most critical diagnostic metric.
    
    - **What it Measures:** The "uncertainty" or "randomness" of the agent's policy. High entropy means the agent is exploring and is not overconfident. Low entropy means the agent is highly certain about its decisions.
        
    - **The Comparison:** We are directly testing the core hypothesis of the TTM paper. TTM is designed to be a **regularizer** that increases policy entropy. We can now plot the entropy curves for the Baseline, PKT, and TTM agents. **We expect to see that the TTM agent maintains a higher and more stable entropy throughout training.** This provides direct, empirical evidence that its regularization is working, which in turn explains its superior generalization to the test set.
        
- **Training Stability:** We measure the variance of the reward curve in the later stages of training. A lower variance indicates a more stable and reliable learning process. This helps us argue that one method produces more consistent agents than another.
    
- **Distillation Loss Trajectory:** By plotting the distillation_loss_history, we can visually confirm that the student is successfully learning from the teacher in both the TTM and PKT experiments. We can see how quickly and how deeply the student is able to minimize this loss.
    

**How to Explain:** "Professor, we have moved beyond just looking at the final output. We are now analyzing the **learning process itself**. Crucially, we are tracking the agent's **policy entropy**. This allows us to empirically validate the theoretical claims of the TTM paper—that its method acts as a powerful regularizer. We can now produce plots that show why TTM is creating a more robust agent that doesn't overfit, which is a significant contribution."

#### **3. Behavioral Statistics Analysis (The "How")**

This analysis characterizes the trading "style" of each agent.

- **Turnover Rate / Trade Frequency:** This tells us how active each agent is. An agent with extremely high turnover might be sensitive to transaction costs and likely has a "nervous" policy. We can compare which distillation method leads to a more patient or more active agent.
    
- **Position Concentration:** This measures how much the agent diversifies its risk. A high concentration might indicate a strategy that takes large, risky bets.