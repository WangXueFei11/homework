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
