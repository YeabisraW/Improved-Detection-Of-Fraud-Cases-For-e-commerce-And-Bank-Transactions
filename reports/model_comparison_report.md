# Task 2: Model Interpretability & Business Strategy Report

## 1. Feature Importance Comparison
| Feature Rank | Built-in (Gini) Importance | SHAP (Impact) Importance |
| :--- | :--- | :--- |
| **Top 1** | V17 (Structural) | V17 (Directional) |
| **Top 2** | V14 | Transaction Amount |
| **Top 3** | V12 | Time/Velocity Features |

**Key Insight:** While the Built-in Gini importance identifies which PCA components (like V17) are best for splitting the data, **SHAP** reveals that 'Amount' has a non-linear relationship where specific high-value thresholds significantly spike fraud probability.

## 2. Concrete Business Actions
Based on SHAP insights from the Credit Card and Ecommerce models:
1. **Automated Step-up Auth:** Trigger mandatory 3D-Secure or Biometric verification for transactions where SHAP indicates the 'Amount' or 'Velocity' exceeds the 95th percentile of the user's normal behavior.
2. **Threshold-based Monitoring:** Implement a real-time rule to flag accounts showing high SHAP values for 'Time' features (e.g., rapid-fire transactions in off-peak hours).
3. **Geo-Fence Logic:** For the Ecommerce model, tie shipping/billing address mismatches (a key SHAP driver) to a manual review queue for orders over $200.