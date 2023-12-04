import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import os
import csv
from scipy.stats import chi2
from math import floor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sns.set(rc={'axes.edgecolor':'gray', 
            'axes.labelcolor': 'gray', 
            'xtick.color': 'gray', 
            'ytick.color': 'gray', 
            'text.color': 'gray',
            'figure.figsize': (20, 10), 
            'legend.fontsize': 12, 
            'font.size': 12, 
            'legend.title_fontsize': 14, 
            'axes.labelsize': 14,
            'axes.titlesize': 24}, 
        style='white')
color_dict = {"normal":"#A8AA79",
              "fire":"#EF812E",
              "water":"#6991F0",
              "grass":"#7AC852",
              "electric":"#F6D030",
              "ice":"#9AD7D9",
              "fighting":"#C12F27",
              "poison":"#A0429F",
              "ground":"#BCA23B",
              "flying":"#A991F0",
              "psychic":"#F85887",
              "bug":"#A7B822",
              "rock":"#B99F38",
              "ghost":"#6D5947",
              "dark":"#70589A",
              "dragon":"#6B3EE3",
              "steel":"#B6B8D0",
              "fairy":"#FF65D5"
             }

fulldata = pd.read_csv("../Python/pokemon.csv")
df = fulldata[["name","type1","type2","weight_kg","hp","generation","is_legendary"]].copy()
# df.info()
missing_data = df[df["weight_kg"].isna()]
# print(missing_data)

def dfix(index_no, type2, weight):
    df.loc[index_no, "type2"] = type2
    df.loc[index_no, "weight_kg"] = weight
    return 0

dfix(18, np.nan, 7.7)
dfix(19, np.nan, 18.5)
dfix(25, np.nan, 30)
dfix(26, np.nan, 12)
dfix(27, np.nan, 29.5)
dfix(36, np.nan, 9.9)
dfix(37, np.nan, 19.9)
dfix(49, np.nan, 0.8)
dfix(50, np.nan, 33.3)
dfix(51, np.nan, 4.2)
dfix(52, np.nan, 32)
dfix(73, "ground", 20)
dfix(74, "ground", 105)
dfix(75, "ground", 300)
dfix(87, np.nan, 30)
dfix(88, np.nan, 30)
dfix(102, "psychic", 120)
dfix(104, np.nan, 45)
dfix(491, np.nan, 2.1)
dfix(554, np.nan, 92.9)
dfix(719, "dark", 490)
dfix(744, np.nan, 25)

#print(df[df["name"].isin(missing_data["name"])])
#df.info()

type_list = pd.Series(color_dict.keys())
for i in type_list:
    df[i] = 0

for i in range(0,len(df)):
    type1_to_add = df.loc[i, "type1"]
    df.loc[i, type1_to_add] = 1
    type2_to_add = df.loc[i, "type2"]
    if type2_to_add is not np.nan:
        df.loc[i, type2_to_add] = 1

# g = sns.scatterplot(x="weight_kg",y="hp",data=df,hue="type1",legend="full",palette=color_dict)
# g.set_title("Pokemon by Weight and Base HP")
# plt.show()

# a = round(df["weight_kg"].corr(df["hp"]),3)
# print("The correlation coefficient between weight and base HP for all pokemon is "+ str(a))

df_type = pd.DataFrame(columns=["type","corr_coef"])
for i in range(0,len(type_list)):
    value_to_add = df.groupby(type_list[i])[["weight_kg","hp"]].corr().loc[1,"hp"]["weight_kg"]
    df_type.loc[len(df_type.index)] = [type_list[i],value_to_add]

df_type.set_index("type", inplace=True)
round(df_type.sort_values(by="corr_coef", ascending=False),3)
# print(df_type)

for i in range(0, len(type_list)):
    df_type.loc[type_list[i],"type_count"] = (sum(df.loc[:,type_list[i]]))

# print(df_type["type_count"].sort_values(ascending=False))


b = 0
for i in range(0, len(type_list)):
    b = b + df_type.loc[type_list[i],"corr_coef"]* df_type.loc[type_list[i],"type_count"]

b = round(b/sum(df_type.loc[:,"type_count"]),3)
# print("The weighted average correlation coefficient of all types of pokemon using weights equal to the number of pokemon with that type is "+ str(b))
# print("This exceeds the unweighted average by " + str(round((b-a),3)))

filtered_data = []
for i in range(0, len(type_list)):
    filtered_data.append(df[df[type_list[i]] == 1])
    df_type.loc[type_list[i], "weight_mean"] = (filtered_data[i])["weight_kg"].mean()
    df_type.loc[type_list[i], "weight_stdev"] = (filtered_data[i])["weight_kg"].std()
# print(round(df_type[["weight_mean", "weight_stdev"]].sort_values(by="weight_mean", ascending=False),2))

# h = sns.barplot(y="weight_mean", data=df_type.sort_values(by="weight_mean",ascending=False), x= df_type.sort_values(by="weight_mean",ascending=False).index, palette=color_dict)
# h.set_title("Average Pokemon Weight by Type")
# plt.show()

# g = sns.boxplot(data = df, x = "type1", y = "weight_kg", palette = color_dict, showfliers=False)
# g.set_title("Pokemon Weights by Primary Type")
# plt.show()

# g = sns.lmplot(x="corr_coef",y="type_count",data=df_type, legend=False, height=8, aspect=2, scatter_kws={"s":10*df_type["type_count"], "color":list(color_dict.values())}, line_kws={"linewidth":8,"color":"purple"})
# g.fig.suptitle("Correlation Coefficients vs. Number of Pokemon of that Type", fontsize=20, y=0.8)
# plt.show()

# g = sns.lmplot(x="corr_coef",y="weight_mean",data=df_type, legend=False, height=10, aspect=2, scatter_kws={"s":10*df_type["type_count"], "color":list(color_dict.values())}, line_kws={"linewidth":8,"color":"purple"})
# g.fig.suptitle("Correlation Coefficients vs. Average Weight of Pokemon of that Type", fontsize=20, y=0.8)
# plt.show()
# fig, ((ax0, ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10, ax11), (ax12, ax13, ax14, ax15, ax16, ax17)) = plt.subplots(3, 6)
# g = ((ax0, ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10, ax11), (ax12, ax13, ax14, ax15, ax16, ax17))
# for i in range(0,len(type_list)):
#     sns.histplot(data=filtered_data[i], x="hp", ax=g[floor(i/6)][i % 6], color= list(color_dict.values())[i], binrange=[0,255], bins=13).set(title=list(color_dict.keys())[i])
# fig.tight_layout()
# fig.suptitle("Distribution of Base HP by Type", fontsize=50, y = 1.1)
# plt.show()
# for i in range(0, len(type_list)):
#     df_type.loc[type_list[i], "hp_mean"] = (filtered_data[i])["hp"].mean()
#     df_type.loc[type_list[i], "hp_stdev"] = (filtered_data[i])["hp"].std()
# print(round(df_type[["hp_mean", "hp_stdev"]].sort_values(by="hp_mean", ascending=False),2))

uni_outliers_by_type = pd.DataFrame(columns=["outlier_count"])

for i in range(0, len(type_list)):
    value_to_add = ((df.sort_values(by="weight_kg", ascending=False).head(30))[type_list[i]]).sum()
    uni_outliers_by_type.loc[type_list[i],"outlier_count"] = value_to_add   
    
# print(uni_outliers_by_type.sort_values(by="outlier_count", ascending=False))

def maha(x=None, data=None, cov=None):
    x_minus_mu = x - data.mean()
    cova = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cova)
    left = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left, x_minus_mu.T)
    return mahal.diagonal()

for i in range(0, len(type_list)):
    df_x = filtered_data[i][["weight_kg", "hp"]]
    df_x.loc[df_x.index,"mahala"] = maha(x=df_x, data=df_x)
    filtered_data[i] = pd.merge(filtered_data[i], df_x["mahala"], left_index=True, right_index=True)
    filtered_data[i]["p_value"] = 1 - chi2.cdf(filtered_data[i]["mahala"], 1)

bivar_outliers = pd.DataFrame(columns= list(filtered_data[0].columns))
for i in range(0, len(type_list)):
    out_to_add = filtered_data[i][filtered_data[i]["p_value"] < .001]
    bivar_outliers = pd.concat([bivar_outliers, out_to_add])

outlier_dupes = bivar_outliers[bivar_outliers.duplicated(subset=["name"],keep="first")]
bivar_outliers = bivar_outliers[bivar_outliers["type2"].isna()]
bivar_outliers = pd.concat([bivar_outliers, outlier_dupes])
# print(bivar_outliers.sort_values(by="name"))
df_no_out = df[~df["name"].isin(bivar_outliers["name"])]

# sns.scatterplot(x="weight_kg",y="hp",data=df_no_out,color="#152558").set_title("Pokemon by Weight and Base HP, Outliers Highlighted in Red")
# sns.scatterplot(x="weight_kg",y="hp",data=bivar_outliers,color="#F82517")

# plt.show()

X = df_no_out[["weight_kg","normal","fire","water","grass","electric","ice","fighting","poison","ground","flying","psychic","bug","rock","ghost","dark","dragon","steel","fairy"]]
y = df_no_out["hp"]
kf = KFold(n_splits=10, shuffle=True, random_state=135)
cv_scores = cross_val_score(LinearRegression(), X=X, y=y, cv=kf, scoring="r2")
cv_results = cross_validate(LinearRegression(), X=X, y=y, cv=kf, return_estimator=True)
cv_coefs = []
cv_intercepts = []
for model in cv_results["estimator"]:
    cv_coefs.append(model.coef_)
    cv_intercepts.append(model.intercept_)

for i in range(0,len(df)):
    pred_hp_int, pred_hp_coef = 0, 0
    for j in range(0, 10):
        pred_hp_list = []
        pred_hp_int = pred_hp_int + cv_intercepts[j]
        pred_hp_coef = pred_hp_coef + cv_coefs[j][0]*df.loc[i,"weight_kg"]
        for k in range(0, len(type_list)):
            pred_hp_coef = pred_hp_coef + df.loc[i,type_list[k]]*cv_coefs[j][k+1]
        pred_hp_list.append(pred_hp_coef + pred_hp_int)
    df.loc[i,"predicted_hp"] = sum(pred_hp_list)/10

# plt.scatter(df["predicted_hp"], df["hp"], color="blue")
# plt.xlabel("predicted HP")
# plt.ylabel("actual HP")
# plt.plot([0,255],[0,255], color="red",linestyle="dashed")
# plt.title("Predicted vs. Actual HP Values")
# plt.show()

# g = sns.histplot((df["hp"] - df["predicted_hp"]),bins=60, color="blue")
# g.set_title("Residuals")
# plt.show()

# plt.scatter(df["predicted_hp"], (df["hp"]-df["predicted_hp"]), color="blue")
# plt.xlabel("predicted HP")
# plt.ylabel("residuals")
# plt.title("Predicted HP vs. Residuals")
# plt.show()

print("Mean Absolute Error:", round(metrics.mean_absolute_error(df["hp"], df["predicted_hp"]),3))
print("Mean Squared Error:", round(metrics.mean_squared_error(df["hp"], df["predicted_hp"]),3))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(df["hp"], df["predicted_hp"])),3))
print("R2 Score:", round(metrics.r2_score(df["hp"], df["predicted_hp"]),3))