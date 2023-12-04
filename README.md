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
