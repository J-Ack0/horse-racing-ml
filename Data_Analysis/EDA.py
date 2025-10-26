import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


df = pd.read_csv("Horse_Racing_ML/Pt2/merged_horse_racing_data.csv")


def run_chi_squared_test(df, cat_feature, min_threshold=30):
    """Performs a Chi-Squared Test on the relationship between the categorical feature and 'winner'."""

    # 1. Identify high-cardinality categories to keep
    # We group all low-count categories into an 'Other' group for the test
    category_counts = df[cat_feature].value_counts()
    keep_categories = category_counts[category_counts >= min_threshold].index

    # 2. Create a modified feature for the test
    df['temp_cat_for_test'] = np.where(
        df[cat_feature].isin(keep_categories),
        df[cat_feature],
        'Other'
    )

    # 3. Create the Contingency Table
    # The table shows the counts of Wins (1) and Losses (0) for each category.
    contingency_table = pd.crosstab(df['temp_cat_for_test'], df['winner'])

    # 4. Run the Chi-Squared Test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"--- Chi-Squared Test for {cat_feature} ---")
    print(f"Degrees of Freedom (dof): {dof}")
    print(f"Chi-Squared Statistic: {chi2:.2f}")
    print(f"P-Value: {p_value:.10f}") # Show high precision for p-value

    # Evaluate the result
    if p_value < 0.05:
        print(f"Conclusion: The p-value is less than 0.05. We reject the null hypothesis.")
        print(f"There is a **statistically significant relationship** between {cat_feature} and the `winner` (i.e., they are not independent).")
    else:
        print(f"Conclusion: The p-value is not less than 0.05. We fail to reject the null hypothesis.")
        print(f"There is no statistically significant relationship between {cat_feature} and the `winner`.")

# Example Usage
run_chi_squared_test(df, 'jockey_clean', min_threshold=30)