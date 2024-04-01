## Install
conda install python=3.11
pip install -r requirements.txt

If Elasticsearch is used:

为了建立document store (for BM25 retriever), 我们需要先安装Elasticsearch

在Ubuntu系统上安装Elasticsearch 8.0及以上版本的步骤如下：

步骤 1: 导入Elasticsearch的公钥，打开终端并运行以下命令以导入Elasticsearch的公钥，这是为了确保下载的软件包是官方的：

```
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
```

步骤 2: 添加Elasticsearch到APT源, 将Elasticsearch的APT仓库添加到您的系统中：

```
sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" > /etc/apt/sources.list.d/elastic-8.x.list
```

步骤 3: 更新APT索引并安装Elasticsearch
首先，更新您的APT包索引：
```
sudo apt-get update
```

然后，安装Elasticsearch：
```
sudo apt-get install elasticsearch
```

步骤 4: 启动Elasticsearch服务
安装完成后，启动Elasticsearch服务：

```
sudo systemctl start elasticsearch.service
```

为了使Elasticsearch在系统启动时自动运行，请启用服务：
```
sudo systemctl enable elasticsearch.service
```

步骤 5: 验证Elasticsearch是否正在运行
通过运行以下命令来检查Elasticsearch是否成功启动并在运行：
```
curl -X GET "localhost:9200/" 把这个整理成md格式的文本，只输出文本
```
