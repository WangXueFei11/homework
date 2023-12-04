# 口袋妖怪重量和基础HP关系分析
### 1.介绍
1. 这是一份探索性的数据分析，分析了宝可梦的体重和基础HP之间的关系。
2. 最初的假设是：宝可梦的重量越大，则基础HP越高。
3. 同时，我认为宝可梦的属性（如：火，草，龙，幽灵）会对重量和基础HP之间的关系产生影响。
4. 宝可梦共有18种种族。
5. 每一个宝可梦都有一个主要的属性，有一部分宝可梦还有一个副属性，主属性和副属性都是18种之一。
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
    g.fig.suptitle("Correlation Coefficients vs. Number of Pokemon of that Type", fontsize=40, y=1.05)
