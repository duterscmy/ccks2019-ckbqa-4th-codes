# ccks2019-ckbqa-4th-codes
中文知识库问答代码，CCKS2019 CKBQA评测获得第四名解决方案
## 任务介绍
  这个评测任务主要是基于中文开放域知识库的智能问答，主办方是北京大学的邹磊、胡森老师。知识库是他们搭建的PKUBASE，语料在https://github.com/pkumod/CKBQA 上。我们主要是参考了ccks2018 COQA评测第二名的方法，在此基础上加入了基于Bert的序列标注、语义匹配等模型，以及桥接等回答复杂问题的环节。具体方法在评测论文里，和本次比赛前三名的方法一起放在了/pdf下供参考。  
  
  注意：因为这个程序依赖自己在本地服务器搭的知识库，所以不搭好知识库的话，entity_extractor及之后的程序都是无法运行的。
## 代码介绍
  ### 文件结构：
  - src: 主要的程序文件
  - helper: 用于生成一些字典/语料集预处理的程序文件
  - data: 存放各种字典，中间文件及最终答案
  - corpus: 原始语料集文件
  - PKUBASE: 评测方提供的知识库文件
  ### 依赖字典：
  这些字典均可通过helper中的程序生成
  - segment_dic.txt: 分词词典
  - prop_dic.pkl: 知识库中的属性值字典
  - mention2entity_dic.pkl: 实体链接字典，实际就是评测方提供的文件存成了pkl格式
  - char_2_prop.pkl：知识库中字符to属性值的倒排索引，用于属性值模糊匹配
  ### 将PKUBASE知识库搭建在本地服务器
  1. 下载neo4j图数据库的linux版本，地址https://neo4j.com/ ，上传到服务器上解压即可。配置好conf，具体参考https://blog.csdn.net/u013946356/article/details/81736232 
  2. 使用大规模导入工具 ./neo4j-admin import 来导入知识库，这里需要把PKUBASE知识库源文件处理成一定格式，参考：生成neo4j文件和各种词典.ipynb
  ### 训练流程（按顺序）：
  - train_ner.py 根据语料集反向标注问题，训练一个NER模型
  - mention_extractor.py: 同时使用jieba分词和bert序列标注来抽取问题中的实体mention
  - prop_extractor.py: 使用规则抽取问题中的属性值
  - entity_extractor.py: 将实体mention和属性值链接到知识库，得到候选实体，并为每个实体生成一些人工特征
  - entity_filter.py: 根据特征，对候选实体进行逻辑回归筛选，保存逻辑回归模型
  - similiary.py 根据该语料微调bert文本匹配模型；需要配置src/data/train.xls等文件。正例就是每个问题和对应的正确关系路径，负例可以随机选n个错误路径。
  - tuple_extractor.py: 根据候选实体从知识库中抽取两跳内的关系作为候选答案，并根据微调后的bert文本匹配模型，生成问题+候选答案的相似度特征
  - tuple_filter.py: 根据bert相似度特征对候选答案进行排序筛选
  - kb.py：连接知识库，定义了一些查询函数。由于连接的是自行搭建的数据库，无法直接执行。
  ### 测试流程
  - answer_bot.py: 该程序会依次调用上述文件，并针对双实体问题进行桥接，最终输出基于神经网络的方法可提交的结果
  - 语义解析模块.ipynb：针对上述基于神经网络方法难以回答且结构清晰的问题，写了一些正则式进行补充，主要还是得到最终可提交结果
  ### 主要参考
  - 预训练的bert文本匹配： https://github.com/terrifyzhao/bert-utils
