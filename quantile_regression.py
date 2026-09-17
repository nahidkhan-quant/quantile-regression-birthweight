import pyreadr

# Read the RData file
result = pyreadr.read_r("Bwght.RData")

# Show the object names stored in the RData file
print(result.keys())

# Extract the dataset
df = list(result.values())[0]

# Preview the dataset
print(df.head())
print(df.columns)

# Convert to CSV
df.to_csv("Bwght.csv", index=False)

print("Bwght.csv was created successfully.")
### part b)###

import pandas as pd
import statsmodels.api as sm

# Load the birth-weight dataset
df = pd.read_csv("Bwght.csv")

# Define the dependent variable
y = df["Weight"]

# Define the explanatory variables
X = df[["PrenatalCare", "Married", "Boy"]]

# Add an intercept to the regression
X = sm.add_constant(X)

# Estimate the OLS model
ols_model = sm.OLS(y, X).fit()

# Display the regression results
print(ols_model.summary())
### part c) ###

ols_model = sm.OLS(y, X).fit(cov_type="HC1")
print(ols_model.summary())
###part d) ###
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# Load the dataset
df = pd.read_csv("Bwght.csv")

# Define dependent variable
y = df["Weight"]

# Define explanatory variables
X = df[["PrenatalCare", "Married", "Boy"]]

# Add intercept
X = sm.add_constant(X)

# Quantiles required by the assignment
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

# Store results
quantile_results = {}

print("\nQUANTILE REGRESSION RESULTS\n")

# Estimate one quantile regression for each tau
for tau in quantiles:
    model = QuantReg(y, X)
    result = model.fit(q=tau)

    quantile_results[tau] = result

    print("=" * 70)
    print(f"Quantile Regression: tau = {tau}")
    print("=" * 70)
    print(result.summary())
    print()

### creating table##
# Create a compact table of coefficient estimates
coef_table = pd.DataFrame({
    tau: quantile_results[tau].params
    for tau in quantiles
}).T

coef_table.index.name = "Quantile"

print("\nCoefficient Estimates Across Quantiles")
print(coef_table)

####part f) ####
### PART (f): Quantile coefficient plots ###

import matplotlib.pyplot as plt

# Quantiles used in part (d)
taus = [0.10, 0.25, 0.50, 0.75, 0.90]

# Extract coefficient estimates from the quantile regression results
intercept_coef = [quantile_results[t].params["const"] for t in taus]
prenatal_coef = [quantile_results[t].params["PrenatalCare"] for t in taus]
married_coef = [quantile_results[t].params["Married"] for t in taus]
boy_coef = [quantile_results[t].params["Boy"] for t in taus]

# -------------------------------------------------
# Figure 1: Quantile regression intercept
# -------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    taus,
    intercept_coef,
    marker="o"
)

plt.xlabel("Quantile (tau)")
plt.ylabel("Intercept coefficient")
plt.title("Quantile Regression Intercept")
plt.xticks(taus)
plt.grid(True)

# Save figure
plt.savefig(
    "quantile_intercept.png",
    dpi=300,
    bbox_inches="tight"
)

# Display figure
plt.show()


# -------------------------------------------------
# Figure 2: Quantile regression slope coefficients
# -------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    taus,
    prenatal_coef,
    marker="o",
    label="PrenatalCare"
)

plt.plot(
    taus,
    married_coef,
    marker="o",
    label="Married"
)

plt.plot(
    taus,
    boy_coef,
    marker="o",
    label="Boy"
)

plt.xlabel("Quantile (tau)")
plt.ylabel("Coefficient")
plt.title("Quantile Regression Slope Coefficients")
plt.xticks(taus)
plt.legend()
plt.grid(True)

# Save figure
plt.savefig(
    "quantile_slope_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)

# Display figure
plt.show()


# Print confirmation
print("\nPart (f) figures created successfully.")
print("Saved as:")
print("1. quantile_intercept.png")
print("2. quantile_slope_coefficients.png")

### part i) CDF comparison###
### PART (i): CDF comparison ###

import matplotlib.pyplot as plt

# Quantile probabilities
taus = [0.10, 0.25, 0.50, 0.75, 0.90]

# Reference-group quantiles from the intercept
cdf_no_care = [2509, 2855, 3174, 3487, 3771]

# Counterfactual quantiles if the same group receives prenatal care
cdf_with_care = [2589, 2930, 3231, 3546, 3825]

# Plot both CDFs
plt.figure(figsize=(8, 6))

plt.plot(
    cdf_no_care,
    taus,
    marker="o",
    label="No Prenatal Care"
)

plt.plot(
    cdf_with_care,
    taus,
    marker="o",
    label="With Prenatal Care"
)

plt.xlabel("Birth Weight (grams)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Birth Weight: Effect of Prenatal Care")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(
    "birthweight_cdf_prenatalcare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Part (i) CDF figure saved as birthweight_cdf_prenatalcare.png")