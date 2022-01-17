# 실제 주소에 따른 위도 경도 값 저장
import requests
from urllib.parse import urlparse

input = []
with open('input.txt', 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()

        if not line: break

        input.append(line.strip())


output = []
for i in input:
    url = 'https://dapi.kakao.com/v2/local/search/address.json?&query=' + i

    result = requests.get(urlparse(url).geturl(),
                          headers={"Authorization": "KakaoAK .."})

    json_obj = result.json()

    for document in json_obj['documents']:
        val = document['y'] + ',' + document['x'] + '\n'
        print(val)

        output.append(val)

with open('output.txt', 'w') as f:
    for o in output:
        f.write(o)

print('Finish..')
