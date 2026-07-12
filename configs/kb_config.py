# 更新以下字段为你本地数据库的实际用户名、密码和数据库名
username = 'root'
hostname = '127.0.0.1'
database_name = 'i_doc_bot'

from urllib.parse import quote

# 使用 quote 函数对密码进行编码
password_encoded = quote('123456')

# 现在使用编码后的密码构建连接字符串
SQLALCHEMY_DATABASE_URI = f"mysql+asyncmy://{username}:{password_encoded}@{hostname}/{database_name}?charset=utf8mb4"