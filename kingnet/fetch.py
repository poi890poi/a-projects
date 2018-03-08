import urllib.request
from urllib.parse import quote
import json

keywords = [u'頭痛', u'感冒', u'發燒', u'牙痛', u'懷孕', u'月經', u'頭髮', u'關節', u'腸病毒', u'手腳',
            u'皮膚', u'胃痛', u'肚子痛', u'拉肚子', u'咳嗽', u'心臟', u'血壓', u'血糖', u'酸痛', u'痠痛',
            u'經痛', u'憂鬱', u'焦慮', u'失眠', u'睡眠', u'頭暈', u'想吐', u'嘔吐', u'血脂', u'牙齒',
            u'聽力', u'勃起', u'白帶', u'感染', u'身高', u'體重', u'減重', u'無力', u'青春痘', u'紅疹',
            u'癢', u'感冒', u'發燒', u'潰瘍', u'腸胃', u'肝', u'腎', u'腫瘤', u'鼻塞', u'分泌物',
            u'耳朵', u'發炎', u'喉嚨', u'便秘', u'小便', u'背', u'麻', u'頭皮', u'保險套', u'頻尿',
            u'病毒', u'細菌', u'乳房', u'囊腫', u'破裂', u'斷裂', u'角膜', u'眼睛', u'痛', u'脫出',
            u'紅腫', u'肩頸', u'肌肉', u'膝蓋', u'甲狀腺', u'新陳代謝', u'肥胖', u'食慾', u'手術', u'卵巢',
            u'子宮', u'燙傷', u'住院', u'外傷', u'化膿', u'粘膜', u'叮咬', u'過敏', u'痘痘', u'傷口',
            u'包皮', u'大便', u'中暑', u'運動', u'呼吸', u'更年期', u'精神', u'水腫', u'沙啞', u'出血',]
print(keywords)

#category = 'inquiry'
category = 'news'

with open('kn_news_small_50x20.json', mode='w', encoding='utf-8') as f:
    f.write('[')
    for keyword in keywords:
        print(quote(keyword))

        with urllib.request.urlopen('https://www.kingnet.com.tw/api2017/omni-search.php?f='+category+'&n=20&keyword='+quote(keyword)) as response:
            json_text = response.read()
            json_obj = json.loads(json_text.decode('utf-8'))
            json_obj['sources'][category]['keyword'] = keyword
            json.dump(json_obj['sources'][category], f)
            f.write(',')

    f.write('{"eof":"eof"}]')
