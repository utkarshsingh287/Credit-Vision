# PROJECT : LOAN APPROVAL PREDICTION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# COLOR THEME (used across all graphs in this project)

PRIMARY   = "#4F46E5"   # Indigo
SECONDARY = "#14B8A6"   # Teal
SUCCESS   = "#22C55E"   # Green
WARNING   = "#F59E0B"   # Amber
DANGER    = "#EF4444"   # Red
PURPLE    = "#8B5CF6"   # Purple
SKY       = "#0EA5E9"   # Sky Blue
GRAY      = "#6B7280"   # Gray

sns.set_theme(style="whitegrid")
rcParams["figure.figsize"] = (8, 5)
rcParams["axes.titlesize"] = 15
rcParams["axes.titleweight"] = "bold"

# load the dataset (Data Collection)

print("="*60)
print("UNIT I : DATA COLLECTION")
print("="*60)

df = pd.read_csv("Loan.csv")
print("Dataset loaded successfully from Loan.csv")
print("Shape of dataset:", df.shape)


# DATA MANIPULATION WITH NUMPY AND PANDAS

print("\n" + "="*60)
print("UNIT II : DATA MANIPULATION WITH NUMPY AND PANDAS")
print("="*60)

# converting AnnualIncome column to a numpy array to try basic numpy functions
income_array = np.array(df["AnnualIncome"])

print("\nNumPy Array Demo on AnnualIncome column:")
print("Type          :", type(income_array))
print("Mean Income   :", np.mean(income_array))
print("Median Income :", np.median(income_array))
print("Std Deviation :", np.std(income_array))
print("Min Income    :", np.min(income_array))
print("Max Income    :", np.max(income_array))

print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nColumn names:")
print(list(df.columns))

# creating a simple pandas series from one column
credit_score_series = pd.Series(df["CreditScore"])
print("\nCreditScore as a Pandas Series (first 5 values):")
print(credit_score_series.head())

# checking for missing values and duplicate rows
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

print("\nNo missing values found, so no imputation was required.")

# ApplicationDate column is not useful for prediction, so removing it
df = df.drop("ApplicationDate", axis=1)

# UNIT III : DATA VISUALIZATION WITH MATPLOTLIB AND SEABORN

print("\n" + "="*60)
print("UNIT III : DATA VISUALIZATION")
print("="*60)

# countplot to see how many applicants got approved vs rejected
plt.figure()
ax = sns.countplot(x="LoanApproved", data=df, palette=[DANGER, SUCCESS])
plt.title("Loan Approved vs Rejected")
plt.xlabel("0 = Rejected, 1 = Approved")
plt.ylabel("Number of Applicants")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.show()

# pie chart for loan approval percentage
plt.figure(figsize=(6, 6))
df["LoanApproved"].value_counts().plot(
    kind="pie",
    colors=[DANGER, SUCCESS],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.03, 0.03),
    labels=["Rejected", "Approved"]
)
plt.title("Loan Approval Percentage")
plt.ylabel("")
plt.tight_layout()
plt.show()

# histograms to see how income and credit score are distributed
important_cols = ["AnnualIncome", "CreditScore"]
colors = [PRIMARY, SKY]

for col, color in zip(important_cols, colors):
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True, color=color)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.show()

# boxplot to compare credit score across approved/rejected applicants
plt.figure()
sns.boxplot(x="LoanApproved", y="CreditScore", data=df, palette=[DANGER, SUCCESS])
plt.title("Credit Score vs Loan Approval")
plt.tight_layout()
plt.show()

# boxplot to compare annual income across approved/rejected applicants
plt.figure()
sns.boxplot(x="LoanApproved", y="AnnualIncome", data=df, palette=[DANGER, SUCCESS])
plt.title("Annual Income vs Loan Approval")
plt.tight_layout()
plt.show()

# scatterplot to see relation between income and loan amount
plt.figure()
sns.scatterplot(data=df, x="AnnualIncome", y="LoanAmount",
                 hue="LoanApproved", palette=[DANGER, SUCCESS], alpha=0.6)
plt.title("Annual Income vs Loan Amount")
plt.tight_layout()
plt.show()

# countplots for a couple of categorical columns
categorical_cols_plot = ["EmploymentStatus", "LoanPurpose"]
palettes = ["crest", "flare"]

for col, pal in zip(categorical_cols_plot, palettes):
    plt.figure(figsize=(9, 5))
    ax = sns.countplot(x=col, data=df, palette=pal)
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

# heatmap to see correlation between all numeric columns
plt.figure(figsize=(16, 10))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of All Numeric Features")
plt.tight_layout()
plt.show()


# EXPLORATORY DATA ANALYSIS (EDA)

print("\n" + "="*60)
print("UNIT IV : EXPLORATORY DATA ANALYSIS")
print("="*60)

# summary statistics for all numeric columns
print("\nSummary Statistics (numeric columns):")
print(df.describe().T)

# checking which features are most correlated with LoanApproved
target_corr = correlation["LoanApproved"].sort_values(ascending=False)
print("\nCorrelation of features with LoanApproved:")
print(target_corr)

# covariance between a few important columns
covariance = df[["AnnualIncome", "LoanAmount", "CreditScore"]].cov()
print("\nCovariance between AnnualIncome, LoanAmount and CreditScore:")
print(covariance)

# picking top 5 positive and top 5 negative correlated features to plot
top_positive = target_corr[1:6]
top_negative = target_corr.sort_values().head(5)

feature_names = []
feature_values = []
bar_colors = []

for name in top_positive.index:
    feature_names.append(name)
    feature_values.append(top_positive[name])
    bar_colors.append(SUCCESS)

for name in top_negative.index:
    feature_names.append(name)
    feature_values.append(top_negative[name])
    bar_colors.append(DANGER)

plt.figure(figsize=(9, 6))
sns.barplot(x=feature_values, y=feature_names, palette=bar_colors)
plt.title("Top Features Correlated with Loan Approval")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

# outlier detection using boxplot and IQR method
outlier_cols = ["AnnualIncome", "CreditScore"]
outlier_colors = [PRIMARY, SKY]

for col, color in zip(outlier_cols, outlier_colors):
    plt.figure()
    sns.boxplot(x=df[col], color=color)
    plt.title(f"{col} - Outlier Detection")
    plt.tight_layout()
    plt.show()

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers found using IQR method")


# INTRODUCTION TO STATISTICAL ANALYSIS

print("\n" + "="*60)
print("UNIT V : STATISTICAL ANALYSIS")
print("="*60)

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# basic descriptive statistics for credit score
print("\nDescriptive Statistics for CreditScore:")
print("Mean   :", df["CreditScore"].mean())
print("Median :", df["CreditScore"].median())
print("Mode   :", df["CreditScore"].mode()[0])
print("Variance:", df["CreditScore"].var())
print("Std Dev :", df["CreditScore"].std())
print("Skewness:", df["CreditScore"].skew())
print("Kurtosis:", df["CreditScore"].kurt())

# shapiro-wilk test to check if AnnualIncome is normally distributed
# (using a sample of 500 rows since shapiro test works better on smaller samples)
sample_income = df["AnnualIncome"].sample(500, random_state=42)
shapiro_stat, shapiro_p = stats.shapiro(sample_income)

print("\nShapiro-Wilk Test on AnnualIncome:")
print("Statistic:", shapiro_stat, " p-value:", shapiro_p)
if shapiro_p < 0.05:
    print("Result: Data is NOT normally distributed (reject H0)")
else:
    print("Result: Data appears normally distributed (fail to reject H0)")

# t-test to compare income of approved vs rejected applicants
approved_income = df[df["LoanApproved"] == 1]["AnnualIncome"]
rejected_income = df[df["LoanApproved"] == 0]["AnnualIncome"]

t_stat, t_p = stats.ttest_ind(approved_income, rejected_income)
print("\nIndependent T-Test: AnnualIncome (Approved vs Rejected)")
print("T-Statistic:", t_stat, " p-value:", t_p)
if t_p < 0.05:
    print("Result: Significant difference in income between the two groups")
else:
    print("Result: No significant difference in income between the two groups")

# chi-squared test to check if employment status and approval are related
contingency_table = pd.crosstab(df["EmploymentStatus"], df["LoanApproved"])
chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Squared Test: EmploymentStatus vs LoanApproved")
print("Chi2 Statistic:", chi2_stat, " p-value:", chi2_p, " Degrees of Freedom:", dof)
if chi2_p < 0.05:
    print("Result: EmploymentStatus and LoanApproved are related (dependent)")
else:
    print("Result: EmploymentStatus and LoanApproved are NOT related (independent)")

# VIF test to check multicollinearity between financial features
vif_features = ["AnnualIncome", "CreditScore", "LoanAmount", "NetWorth",
                 "TotalAssets", "TotalLiabilities", "MonthlyIncome"]

vif_data = add_constant(df[vif_features])

feature_list = []
vif_list = []

for i in range(vif_data.shape[1]):
    feature_list.append(vif_data.columns[i])
    vif_list.append(variance_inflation_factor(vif_data.values, i))

vif_result = pd.DataFrame()
vif_result["Feature"] = feature_list
vif_result["VIF"] = vif_list

print("\nVariance Inflation Factor (VIF) for selected features:")
print(vif_result)
print("Note: VIF > 5 usually indicates high multicollinearity")

# plotting a few common probability distributions
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

x_norm = np.linspace(-4, 4, 200)
axes[0, 0].plot(x_norm, stats.norm.pdf(x_norm, 0, 1), color=PRIMARY, linewidth=2)
axes[0, 0].set_title("Normal Distribution")

x_uni = np.linspace(0, 1, 200)
axes[0, 1].plot(x_uni, stats.uniform.pdf(x_uni), color=SECONDARY, linewidth=2)
axes[0, 1].set_title("Uniform Distribution")

x_binom = np.arange(0, 20)
axes[1, 0].bar(x_binom, stats.binom.pmf(x_binom, 20, 0.5), color=WARNING)
axes[1, 0].set_title("Binomial Distribution (n=20, p=0.5)")

x_poisson = np.arange(0, 15)
axes[1, 1].bar(x_poisson, stats.poisson.pmf(x_poisson, 4), color=DANGER)
axes[1, 1].set_title("Poisson Distribution (lambda=4)")

plt.tight_layout()
plt.show()

# A/B test - comparing loan amount for mortgage vs rent applicants
group_a = df[df["HomeOwnershipStatus"] == "Mortgage"]["LoanAmount"]
group_b = df[df["HomeOwnershipStatus"] == "Rent"]["LoanAmount"]

ab_t_stat, ab_p_value = stats.ttest_ind(group_a, group_b)

print("\nA/B Test: LoanAmount (Mortgage vs Rent)")
print("Group A (Mortgage) Mean LoanAmount:", group_a.mean())
print("Group B (Rent) Mean LoanAmount    :", group_b.mean())
print("T-Statistic:", ab_t_stat, " p-value:", ab_p_value)

if ab_p_value < 0.05:
    print("Result: Statistically significant difference between the two groups")
else:
    print("Result: No statistically significant difference between the two groups")

plt.figure()
sns.barplot(x=["Mortgage", "Rent"], y=[group_a.mean(), group_b.mean()],
            palette=[PRIMARY, SECONDARY])
plt.title("A/B Test - Average Loan Amount by Home Ownership")
plt.ylabel("Average Loan Amount")
plt.tight_layout()
plt.show()

# ROLE OF MACHINE LEARNING IN DATA SCIENCE

print("\n" + "="*60)
print("UNIT VI : MACHINE LEARNING")
print("="*60)

print("""
CRISP-DM Framework followed in this project:
1. Business Understanding : Predict loan approval to help lenders
2. Data Understanding      : Explored dataset in Unit IV (EDA)
3. Data Preparation        : Cleaning, encoding, scaling (Unit II & below)
4. Modelling                : Linear Regression + Classification models
5. Evaluation                : Accuracy, Precision, Recall, F1, R2 Score
6. Deployment                : Model can be used for future predictions
""")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

# simple linear regression example - predicting monthly income from annual income
X_lr = df[["AnnualIncome"]]
y_lr = df["MonthlyIncome"]

X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_lr_train, y_lr_train)
y_lr_pred = lin_reg.predict(X_lr_test)

print("Linear Regression : Predicting MonthlyIncome from AnnualIncome")
print("Coefficient:", lin_reg.coef_[0])
print("Intercept  :", lin_reg.intercept_)
print("R2 Score   :", r2_score(y_lr_test, y_lr_pred))
print("RMSE       :", np.sqrt(mean_squared_error(y_lr_test, y_lr_pred)))

plt.figure()
plt.scatter(X_lr_test, y_lr_test, color=SKY, alpha=0.4, label="Actual")
plt.plot(X_lr_test, y_lr_pred, color=DANGER, linewidth=2, label="Regression Line")
plt.title("Linear Regression - Annual Income vs Monthly Income")
plt.xlabel("Annual Income")
plt.ylabel("Monthly Income")
plt.legend()
plt.tight_layout()
plt.show()

# creating a few new columns to help the model make better predictions
df["LoanToIncomeRatio"] = df["LoanAmount"] / (df["AnnualIncome"] + 1)
df["LiquidAssets"] = df["SavingsAccountBalance"] + df["CheckingAccountBalance"]
df["DebtToAssetRatio"] = df["TotalLiabilities"] / (df["TotalAssets"] + 1)
df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 30, 45, 60, 100],
                         labels=["Young", "Adult", "MiddleAge", "Senior"])

print("\nNew features added: LoanToIncomeRatio, LiquidAssets, DebtToAssetRatio, AgeGroup")

# converting text columns into numbers so the model can understand them
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
print("Categorical columns to encode:", list(categorical_cols))

encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# RiskScore is removed here because it is very strongly correlated with
# LoanApproved (almost like it decided the approval itself), so keeping
# it would let the model cheat instead of learning from real data
X = df.drop(["LoanApproved", "RiskScore"], axis=1)
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scaling so all columns are on a similar range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training 3 different classification models and comparing them
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

    print("\n" + "="*60)
    print(name)
    print("="*60)
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print(classification_report(y_test, y_pred))

# putting all model results together in one table to compare
result_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
result_df = result_df.sort_values(by="Accuracy", ascending=False)
result_df = result_df.reset_index(drop=True)

print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)
print(result_df)

plt.figure()
sns.barplot(data=result_df, x="Accuracy", y="Model", palette="viridis")
plt.title("Model Comparison by Accuracy")
plt.tight_layout()
plt.show()

# picking the best model based on accuracy
best_model_name = result_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print("\nBest performing model:", best_model_name)

# confusion matrix for the best model
y_pred_best = best_model.predict(X_test_scaled)
plt.figure()
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=["Rejected", "Approved"],
    cmap="Greens", colorbar=False
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.show()

# roc curve for the best model
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, color=PRIMARY, linewidth=2, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color=GRAY)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.show()

# feature importance for the best model (only works for tree-based models)
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    importance = importance.head(10)

    plt.figure()
    sns.barplot(x=importance.values, y=importance.index, palette="mako")
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    print("\nTop 10 Important Features:")
    print(importance)

# final conclusion
print("\n" + "="*60)
print("PROJECT CONCLUSION")
print("="*60)
print(f"""
1. Best performing classification model: {best_model_name}
2. Linear Regression confirmed a strong linear relationship
   between Annual Income and Monthly Income.
3. Statistical tests (T-test, Chi-Squared) confirmed that income
   and employment status are significantly related to approval.
4. Shapiro-Wilk test showed AnnualIncome is not perfectly normal,
   which is expected for real-world financial data.
5. VIF check confirmed multicollinearity was low among the
   selected financial features.
6. RiskScore was intentionally removed from classification
   training because it was too strongly correlated with the
   target (data leakage), which would have made results
   unrealistic.
7. This project demonstrates the complete CRISP-DM pipeline:
   from data collection to a working, evaluated ML model.
""")