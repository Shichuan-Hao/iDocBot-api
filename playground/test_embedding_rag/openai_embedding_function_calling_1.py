import os
import openai

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



openai.api_key = os.getenv("OPENAI_API_KEY")

sql_inter_description = "用于执行一段SQL代码，并最终获取telco_db数据库数据查询结果，\
核心功能是将输入的SQL代码传输至MySQL环境中进行运行，\
并最终返回SQL代码运行结果。需要注意的是，本函数是借助pymysql来连接MySQL数据库。"

q1 = "请帮我下telco_db数据库中所有用户的性别和年龄信息。"

q2 = "请帮我介绍下什么是机器学习？"

text_tuples = (sql_inter_description, q1, q2)
res = openai.Embedding.create(
  model="text-embedding-ada-002",
  input=text_tuples,
  encoding_format="float"
)

cosine_similarity([res.data[0].embedding, res.data[1].embedding, res.data[2].embedding])

