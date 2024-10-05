import hypors
import polars as pl

# Create example Polars Series
group1 = pl.Series("Group 1", [2.0, 3.0, 3.0, 5.0, 6.0])
group2 = pl.Series("Group 2", [3.0, 4.0, 4.0, 6.0, 8.0])
group3 = pl.Series("Group 3", [5.0, 6.0, 7.0, 8.0, 9.0])

# Call the ANOVA function
result = hypors.anova([group1, group2, group3], alpha=0.05)
print(result.f_statistic, result.p_value)
