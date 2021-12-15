#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import gensim
import src.base

stock_name = '农夫山泉'
#file_name = '3ccb918a76ffecbbbad21f3ac23348d9'
input_content1 = '2020年已经过去了，已经到来的新一年，你是不是已经做好准备来面对了呢？在过去的一年，突如其来的疫情，让全球不少富豪的财富快速增长，这是在经济宽松的背景下达成的（主要因素），而你知道2020年全球最能赚钱的大佬吗？据彭博亿万富豪指数实时数据，截至2020年12月31日，马斯克资产增加1400亿美元，位居年度资产增加最多的富豪首位，贝索斯以769亿美元排在第二，农夫山泉钟睒睒以709亿美元位列第三，其也是今年中国，乃至亚洲赚钱最多的人碧桂园杨惠妍损失73.4亿美元为"最惨"富豪。据彭博亿万富翁指数，农夫山泉董事长钟睒睒超越印度信实工业集团董事长穆克什-安巴尼，成为亚洲首富，在全球富豪榜上名列第11位。钟睒睒的财富今年暴涨709亿美元，达到778亿美元。这是历史上最快的财富积累之一，考虑到直到今年他在中国以外还鲜为人知，这就更加令人瞩目。钟睒睒的财富主要来自两个互不相关的领域。除农夫山泉以外，他还持有北京万泰生物药业股份有限公司的20%股份。今年4月，万泰生物药业在A股上市，几个月后，农夫山泉成为香港最热门的上市公司之一。自上市以来，农富山泉股价已上涨155%，万泰药业涨幅超过2000%。据悉，今年8月31日，农夫山泉赴港上市，IPO发行价21.5港元。也是从那时候开始，低调的创始人钟睒睒才渐为人所知。'
input_content2 = '1年赚50亿，农夫山泉揭开暴利的卖水生意。一瓶矿泉水的生意：毛利过半，卖水一年赚50亿。农夫山泉IPO一事，终于敲定了。4月29日晚间，农夫山泉正式向港交所递交招股说明书，中金公司和摩根士丹利担任联席保荐人，募资规模预计为10亿美元。至此，一个庞大的瓶装水帝国浮出水面。'
input_content3 = '农夫山泉：2020年净利润52.8亿元 同比增长6.6%财联社3月25日讯，农夫山泉2020年营收228.8亿元人民币，同比下降4.8%，市场预估235.7亿元人民币；净利润52.8亿元人民币，同比增长6.6%，市场预估51.1亿元人民币。农夫山泉2020年每股基本盈利为人民币0.48元，同比增加4.3%，建议派发期末股息每股普通股人民币0.17元。'
input_content4 = '【2019安心奖】旨在评选出新的消费环境下，锐意进取，不断创新发展的优秀企业，以鼓励和致敬他们促进中国消费结构转型，推动中国经济高质量发展所做出的卓越贡献。经过3个月的激烈角逐，通过评委会的层层筛选，综合大众的投票意见、媒体评审和专家评审的意见，最终有50家企业以细分品类中的出色表现脱颖而出，斩获大奖。农夫山泉以对大自然的敬畏的品质保证获得“年度饮品企业”大奖。“农夫山泉有点甜”，是他对消费者的郑重承诺；坚持"天然、健康"的产品理念，倡导以健康的生活方式满足消费者补充营养元素的要求，是他低调的品牌性格；注重产品研发以及新标准、新概念的推广，农夫山泉致力于行业标准和规则的制定和引领；创新渠道管理系统，他是行业销售终端数字化信息化升级的助力者。农夫山泉，以其“农夫”般的朴素与执著，书写着属于他的华章。'


seg_dir = '/segmentation/news/content/' + stock_name + '/'
save_dir = '/similarity/model/news/content/' + stock_name + '/doc2vec/vec_20_min_2_epo_20/'
data_dir = '/data/news/' + stock_name + '/'

stoplists = src.base.get_stoplists()

#f = open(data_dir + file_name + '.txt')
#contents = f.read()
#dictionary = json.loads(contents)
#input_content = dictionary['content']
#f.close()

model = gensim.models.doc2vec.Doc2Vec.load(save_dir + '/default.d2v')

file_names_json = os.listdir(seg_dir)



def stock_sim(input_content,stoplists,model,file_names_json,data_dir):
    input_words_list = src.base.segment_process(input_content, stoplists)
    inferred_vector = model.infer_vector(input_words_list)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    outcome = []
    output_file_name = file_names_json[sims[0][0]]
    output_f = open(data_dir + output_file_name.replace('.json','') + '.txt')
    output_contents = output_f.read()
    output_dictionary = json.loads(output_contents)
    output_content = output_dictionary['content']
    output_f.close()
    outcome.append(input_content)
    outcome.append(sims[0][1])
    outcome.append(output_file_name)
    outcome.append(output_content)
    return outcome


result1 = stock_sim(input_content1,stoplists,model,file_names_json,data_dir)
result2 = stock_sim(input_content2,stoplists,model,file_names_json,data_dir)
result3 = stock_sim(input_content3,stoplists,model,file_names_json,data_dir)
result4 = stock_sim(input_content4,stoplists,model,file_names_json,data_dir)

def takeSecond(element):
    return element[1]

result_total = [result1,result2,result3,result4]
result_total.sort(key = takeSecond,reverse = True)

print('排序结果：')
for i in result_total:
    print('原文：')
    print(i[0])
    print('\n')
    print('most_similar')
    print(i[1:3])
    print(i[3])
    print('\n')
