# 口袋妖怪重量和基础HP关系分析
### 1.介绍
数据来自http://serebii.net/
1. 这是一份探索性的数据分析，分析了宝可梦的体重和基础HP之间的关系。
2. 最初的假设是：宝可梦的重量越大，则基础HP越高，这是符合我们的正常认知的。
3. 同时，我认为宝可梦的属性（如：火，草，龙，幽灵）会对重量和基础HP之间的关系产生影响。
4. 宝可梦共有18种属性。
5. 每一个宝可梦都有一个主要的属性，有一部分宝可梦还有一个副属性，主属性和副属性都是18种之一。

需要用到的库和属性：

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

### 2.数据清洗
    fulldata = pd.read_csv("../Python/pokemon.csv")
    df = fulldata[["name","type1","type2","weight_kg","hp","generation","is_legendary"]].copy()
    df.info()

![1](https://github.com/WangXueFei11/homework/assets/144666483/63566b1f-3009-4c8f-841b-3273596ad769)

1. 从中我们可以看出，有20个宝可梦的重量数据丢失了。
2. 事实证明，大部分数据丢失是因为宝可梦有不同的形态。
3. 其中一些宝可梦的"type2"栏中的副属性是不正确的，因为进化后的属性跟原本的属性不相同。
4. 在出现这种差异的地方，只使用与系列中引入的主属性相关的信息。
5. 在大多数情况下，这包括删除宝可梦阿罗拉形态中的错误属性信息和添加缺失的体重。

下面展示缺失“weight_kg”的数据框：

    missing_data = df[df["weight_kg"].isna()]
    print(missing_data)

![2](https://github.com/WangXueFei11/homework/assets/144666483/401c6744-7c4b-45c3-aae0-bba4840f1ead)

下面创建一个函数，用来修正和补全数据，然后使用正确的的数据作为参数，调用这个函数20次，最后检验一下修改后的数据：

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

    df[df["name"].isin(missing_data["name"])]

![3](https://github.com/WangXueFei11/homework/assets/144666483/d24cee0f-9895-4bd3-a2e7-95787df600b1)

再次检查，确保数据得到正确的处理：

    df.info()

![4](https://github.com/WangXueFei11/homework/assets/144666483/f3d8097c-bbd5-406a-8fc0-3204bb3447e3)

可以看到，weight_kg有801，跟宝可梦数量相同。

### 3.数据分析
创造18个新列，用于存放18种属性的宝可梦，使用二进制数据1或0,1表示是这种属性，0表示不是这种属性，初始状态全部为0：

    type_list = pd.Series(color_dict.keys())
    for i in type_list:
        df[i] = 0

检查每一个宝可梦的主属性和副属性，将相应属性位置的二进制数设为1：

    for i in range(0,len(df)):
        type1_to_add = df.loc[i, "type1"]
        df.loc[i, type1_to_add] = 1
        type2_to_add = df.loc[i, "type2"]
        if type2_to_add is not np.nan:
            df.loc[i, type2_to_add] = 1

对于我们本次的主要任务而言，主属性和副属性的地位同等重要。

    
    g = sns.scatterplot(x="weight_kg",y="hp",data=df,hue="type1",legend="full",palette=color_dict)
    g.set_title("Pokemon by Weight and Base HP")
    plt.show()

![散点图](https://github.com/WangXueFei11/homework/assets/144666483/54ef3a7a-6fd0-447e-8d25-47c4c9e01b68)

1. 散点图基本上显示了宝可梦的重量跟基础HP之间的正相关关系；
2. 可以看出，大部分宝可梦的重量都在100KG以下；
3. 100KG以下宝可梦的基础生命值的极差很大。

下面计算相关系数：

    a = round(df["weight_kg"].corr(df["hp"]),3)
    print("The correlation coefficient between weight and base HP for all pokemon is "+ str(a))

![5](https://github.com/WangXueFei11/homework/assets/144666483/9ebb34e4-772f-4ce1-b442-16810f0e59d6)

可知相关系数为0.425，0.425说明相关性是比较弱的（正相关）。


下面考察每种属性的宝可梦的重量和基础HP之间的相关性，有两种属性的宝可梦将会被包含在这两种属性的桶中。

    df_type = pd.DataFrame(columns=["type","corr_coef"])
    for i in range(0,len(type_list)):
        value_to_add = df.groupby(type_list[i])[["weight_kg","hp"]].corr().loc[1,"hp"]["weight_kg"]
        df_type.loc[len(df_type.index)] = [type_list[i],value_to_add]
    
    df_type.set_index("type", inplace=True)
    round(df_type.sort_values(by="corr_coef", ascending=False),3)
    print(df_type)

创建“df_type”，这是一个使用宝可梦属性作为索引的数据框架，并将使用这些属性的汇总数据填充。在数据框创建之后，对具有特定属性的宝可梦进行分组，并获得重量和基础HP之间的相关系数。

![6](https://github.com/WangXueFei11/homework/assets/144666483/5d0e7366-8813-47c8-a00e-032ab8b593ae)


1. 按照属性分类之后，可以看出在大部分属性的宝可梦中，重量跟基础HP有更强的正相关性；
2. 其中只有毒系（poison,0.391）、飞行系（flying,0.378）、地面系（ground,0.359）和超能系（psychic,0.074）的相关系数小于所有属性的平均值（0.425）。


下面统计每种属性的宝可梦的数量：

    df_type = pd.DataFrame()
    for i in range(0, len(type_list)):
        df_type.loc[type_list[i],"type_count"] = (sum(df.loc[:,type_list[i]]))

    print(df_type["type_count"].sort_values(ascending=False))

通过计算列中对应属性的1的总和来计算每种特定属性的宝可梦数量。

![7](https://github.com/WangXueFei11/homework/assets/144666483/cccef413-3055-451b-90b5-53c39cb62a48)


可以看出，水系，普通系和飞行系的宝可梦最多。
因为有的宝可梦有两种属性，所以各种类型的宝可梦的数量和要多于宝可梦的数量，多了401，即有两种属性的宝可梦的数量。

    b = 0
    for i in range(0, len(type_list)):
        b = b + df_type.loc[type_list[i],"corr_coef"]* df_type.loc[type_list[i],"type_count"]

    b = round(b/sum(df_type.loc[:,"type_count"]),3)
    print("The weighted average correlation coefficient of all types of pokemon using weights equal to the number of pokemon with that type is "+ str(b))
    print("This exceeds the unweighted average by " + str(round((b-a),3)))


![8](https://github.com/WangXueFei11/homework/assets/144666483/06bde6b5-8cc1-4fbe-8333-85ac7d9461c4)

可以看出，所有属性的宝可梦的加权平均相关系数（权重为该属性宝可梦的数量）为0.483，比未加权高了0.058。


求取所有属性的宝可梦的重量的平均值和标准差：

    df_type = pd.DataFrame()
    filtered_data = []
    for i in range(0, len(type_list)):
        filtered_data.append(df[df[type_list[i]] == 1])
        df_type.loc[type_list[i], "weight_mean"] = (filtered_data[i])["weight_kg"].mean()
        df_type.loc[type_list[i], "weight_stdev"] = (filtered_data[i])["weight_kg"].std()
    print(round(df_type[["weight_mean", "weight_stdev"]].sort_values(by="weight_mean", ascending=False),2))

创建1个包含18个数据框的列表，每个数据框只包含具有一种特定属性的宝可梦。然后，分别求得所有属性的宝可梦的重量的平均值和标准差，并将这些信息记录到“df_type”中。

![9](https://github.com/WangXueFei11/homework/assets/144666483/eef15d3a-d9cc-41b1-a9fd-2de3cda35a53)

一般来说，宝可梦的重量跟它的属性有关，龙系、钢系和地面系的宝可梦是最重的。

    g = sns.barplot(y="weight_mean", data=df_type.sort_values(by="weight_mean",ascending=False), x= df_type.sort_values(by="weight_mean",ascending=False).index, palette=color_dict)
    g.set_title("Average Pokemon Weight by Type")
    plt.show()

![柱状图](https://github.com/WangXueFei11/homework/assets/144666483/1a0ce104-3f59-4536-a48f-bcaea6b8d735)
柱状图展示了各种属性的宝可梦的平均重量。

    g = sns.boxplot(data = df, x = "type1", y = "weight_kg", palette = color_dict, showfliers=False)
    g.set_title("Pokemon Weights by Primary Type")
    plt.show()

![箱形图](https://github.com/WangXueFei11/homework/assets/144666483/e841414b-171e-492b-9480-0b6a82827a02)

这些箱形图只涉及宝可梦的主属性，因此值与上面的条形图有些不同。

    g = sns.lmplot(x="corr_coef",y="type_count",data=df_type, legend=False, height=10, aspect=2, scatter_kws={"s":10*df_type["type_count"], "color":list(color_dict.values())}, line_kws= 
    {"linewidth":8,"color":"purple"})
    g.fig.suptitle("Correlation Coefficients vs. Number of Pokemon of that Type", fontsize=20, y=0.8)
    plt.show()

![宝可梦数量与相关系数的关系](https://github.com/WangXueFei11/homework/assets/144666483/df1613d9-b141-4665-bba1-162e8beb65ef)

点的大小跟这种类型的宝可梦的数量相对应，这张图的下降趋势表明，随着宝可梦的主属性变得越来越不常见，重量和基础HP之间的相关性变得越来越强。

    g = sns.lmplot(x="corr_coef",y="weight_mean",data=df_type, legend=False, height=10, aspect=2, scatter_kws={"s":10*df_type["type_count"], "color":list(color_dict.values())}, line_kws= 
    {"linewidth":8,"color":"purple"})
    g.fig.suptitle("Correlation Coefficients vs. Average Weight of Pokemon of that Type", fontsize=20, y=0.8)
    plt.show()

![宝可梦平均重量与相关系数的关系](https://github.com/WangXueFei11/homework/assets/144666483/c6b6a439-4636-49aa-a7f7-5ce6ead5c1ac)

该图的轻微上升趋势表明，随着特定属性宝可梦的平均重量上升，重量和基础HP之间的相关性变得越来越强。

    fig, ((ax0, ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10, ax11), (ax12, ax13, ax14, ax15, ax16, ax17)) = plt.subplots(3, 6)
    g = ((ax0, ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10, ax11), (ax12, ax13, ax14, ax15, ax16, ax17))
    for i in range(0,len(type_list)):
        sns.histplot(data=filtered_data[i], x="hp", ax=g[floor(i/6)][i % 6], color= list(color_dict.values())[i], binrange=[0,255], bins=13).set(title=list(color_dict.keys())[i])
    fig.tight_layout()
    fig.suptitle("Distribution of Base HP by Type", fontsize=50, y = 1.1)

![各个系HP的分布](https://github.com/WangXueFei11/homework/assets/144666483/775a28f0-73fb-4301-82c4-8458de7eec81)

1. 这些直方图显示了特定范围内宝可梦的基础HP；
2. 与重量的分布相比，基础HP的变化范围较小；
3. 可能与游戏设定最大的基础HP为255有关；
4. 而重量主要是一种描述性特征，对宝可梦战斗的影响微乎其微，所以重量基本上是没有上限的。

下面计算每种属性宝可梦的基础HP的平均值和标准差：

    for i in range(0, len(type_list)):
        df_type.loc[type_list[i], "hp_mean"] = (filtered_data[i])["hp"].mean()
        df_type.loc[type_list[i], "hp_stdev"] = (filtered_data[i])["hp"].std()
    print(round(df_type[["hp_mean", "hp_stdev"]].sort_values(by="hp_mean", ascending=False),2))

![10](https://github.com/WangXueFei11/homework/assets/144666483/93f38b94-eaf5-4c99-82b0-99f8cb1ab3da)
![9](https://github.com/WangXueFei11/homework/assets/144666483/53721bdb-bff9-496e-bd33-383b42ec4094)

结合每种属性宝可梦的平均重量的平均值可以看出：
1. 龙系，冰系和地面系的宝可梦的平均基础HP是最高的；
2. 钢系和岩石系的宝可梦有很高的平均重量，但平均HP位于中等水平；
3. 妖精系宝可梦的平均重量最低，但平均HP高于钢系。
4. 虫系和毒系宝可梦的平均重量和平均HP都较低。

# 4.识别和去除异常值

下面找出重量最大的30个宝可梦，并给出每一个属性拥有几个：

    uni_outliers_by_type = pd.DataFrame(columns=["outlier_count"])
    for i in range(0, len(type_list)):
        value_to_add = ((df.sort_values(by="weight_kg", ascending=False).head(30))[type_list[i]]).sum()
        uni_outliers_by_type.loc[type_list[i],"outlier_count"] = value_to_add   
    
    print(uni_outliers_by_type.sort_values(by="outlier_count", ascending=False))

![11](https://github.com/WangXueFei11/homework/assets/144666483/3c573576-c10b-4c07-92cf-ecb7c7cc8102)

1. 地面系，龙系和钢系的宝可梦构成了30个最重的宝可梦的很大一部分；
2. 仅通过权重去除异常值将导致这些属性的宝可梦的代表性不足；
3. 因此考虑移除多元异常值，即与其他同属性的宝可梦相比，其重量与基础HP的比率异常的宝可梦。
4. 通过计算与其马氏距离相关的p值来识别多元变量异常值。

这是一个计算给定输入和数据集的马氏距离的函数：
    def maha(x=None, data=None, cov=None):
        x_minus_mu = x - data.mean()
        cova = np.cov(data.values.T)
        inv_covmat = sp.linalg.inv(cova)
        left = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left, x_minus_mu.T)
        return mahal.diagonal()
        
这个循环使用数据帧列表，其中包含按属性过滤的每个宝可梦，并计算马氏距离和相关的p值：（p值的公式基于使用1个自由度的卡方分布(因为变量的数量为2)。）

    for i in range(0, len(type_list)):
        df_x = filtered_data[i][["weight_kg", "hp"]]
        df_x.loc[df_x.index,"mahala"] = maha(x=df_x, data=df_x)
        filtered_data[i] = pd.merge(filtered_data[i], df_x["mahala"], left_index=True, right_index=True)
        filtered_data[i]["p_value"] = 1 - chi2.cdf(filtered_data[i]["mahala"], 1)

找到所有具有异常低的p值(小于0.001)的宝可梦，并将它们添加到异常值列表中：（P < 0.001是确定数据点是否为离群值的标准。）
    bivar_outliers = pd.DataFrame(columns= list(filtered_data[0].columns))
    for i in range(0, len(type_list)):
        out_to_add = filtered_data[i][filtered_data[i]["p_value"] < .001]
        bivar_outliers = pd.concat([bivar_outliers, out_to_add])

    outlier_dupes = bivar_outliers[bivar_outliers.duplicated(subset=["name"],keep="first")]
    bivar_outliers = bivar_outliers[bivar_outliers["type2"].isna()]
    bivar_outliers = pd.concat([bivar_outliers, outlier_dupes])
    print(bivar_outliers.sort_values(by="name"))

1. 双属性宝可梦包含在两个特定属性的数据框架中，而不是其中一个。
2. 因此，双属性宝可梦可能是其中一种属性宝可梦中的异常值，而不是另一种属性宝可梦中的异常值。
3. 选择在两种属性的宝可梦中都具有p < 0.001时才将其作为异常值。
4. 找到在离群值列表中列出两次的双属性宝可梦(因此，它们是两种属性的离群值)。
5. 然后清除双属性宝可梦的异常值列表，并只重新添加那些双属性宝可梦的异常值。

以下20个宝可梦的重量和基础HP与所有与其具有相同属性的宝可梦的重量和基础HP分布相比是异常的。

![12](https://github.com/WangXueFei11/homework/assets/144666483/61ff18ab-e737-41b2-b83d-45259bf40048)

创建一个新的数据框架，删除掉异常值：

    df_no_out = df[~df["name"].isin(bivar_outliers["name"])]
    sns.scatterplot(x="weight_kg",y="hp",data=df_no_out,color="#152558").set_title("Pokemon by Weight and Base HP, Outliers Highlighted in Red")
    sns.scatterplot(x="weight_kg",y="hp",data=bivar_outliers,color="#F82517")
    plt.show()

![异常值](https://github.com/WangXueFei11/homework/assets/144666483/47c4fda6-ac59-462d-9629-46c949f126d4)

大多数被视为异常值的宝可梦，它们的重量或基础HP也是重量或基础HP的单变量异常值。由于每种属性的重量和基础HP分布不同，有些宝可梦不被认为是异常值，尽管它们的重量或基础HP比其他异常值的宝可梦更极端。

# 5.线性回归
开发一个具有10倍交叉验证的线性回归模型，尝试根据宝可梦的重量和属性来预测其基础HP。

1. 训练模型的输入是宝可梦的重量和属性(用与所有18种可能属性对应的二进制数据列识别属性)。
2. 训练目标是宝可梦的基础HP。
3. 将异常值去除以训练模型，但该模型的预测也将应用于包括异常值在内的数据集。
4. 系数和截距分别存储在名为“cv_coefs”和“cv_intercepts”的变量中。

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

使用线性回归模型的系数和截距来预测每个宝可梦的基础HP，每个预测都是通过交叉验证中使用的单独模型的10个预测的平均值来实现的：

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

    plt.scatter(df["predicted_hp"], df["hp"], color="blue")
    plt.xlabel("predicted HP")
    plt.ylabel("actual HP")
    plt.plot([0,255],[0,255], color="red",linestyle="dashed")
    plt.title("Predicted vs. Actual HP Values")
    plt.show()

预测结果：

![预测](https://github.com/WangXueFei11/homework/assets/144666483/36b44f92-b3fc-4bbd-8e4a-e65798fd878f)

1. 红色虚线表示y=x;
2. HP预测偏向于平均值，大多数预测在60-80范围内;
3. 这可能是由于重量相对较低的宝可梦有较大的HP值范围。

预测误差：

![预测误差](https://github.com/WangXueFei11/homework/assets/144666483/23dab869-7d1d-4f10-ae5c-c1d9b2346a72)

    g = sns.histplot((df["hp"] - df["predicted_hp"]),bins=60, color="blue")
    g.set_title("Residuals")
    plt.show()

大多数预测值与实际值相差不超过20 HP。

下面展示预测误差与预测值之间的关系：

    plt.style.use("seaborn-whitegrid")
    plt.scatter(df["predicted_hp"], (df["hp"]-df["predicted_hp"]), color="blue")
    plt.xlabel("predicted HP")
    plt.ylabel("residuals")
    plt.title("Predicted HP vs. Residuals")
    plt.show()

![误差与预测值之间的关系](https://github.com/WangXueFei11/homework/assets/144666483/625f3e19-ce02-49ae-a4f2-36c631289584)

下面数据化误差：

    print("Mean Absolute Error:", round(metrics.mean_absolute_error(df["hp"], df["predicted_hp"]),3))
    print("Mean Squared Error:", round(metrics.mean_squared_error(df["hp"], df["predicted_hp"]),3))
    print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(df["hp"], df["predicted_hp"])),3))
    print("R2 Score:", round(metrics.r2_score(df["hp"], df["predicted_hp"]),3))

![数据化误差](https://github.com/WangXueFei11/homework/assets/144666483/ff849c82-e1f6-4995-a2fb-fdeef7ed2f48)

1. R2分数相对较低，这表明只有19.3%的基础HP可以用它的重量和属性来解释；
2. 这就意味着还有其他因素，如宝可梦的其他信息，如身高或进化形态的等，可以帮助我们做出更准确的预测；
3. 然而，正如相关系数和R2分数所显示的那样，宝可梦的体重与其基础HP之间通常存在正相关关系。
