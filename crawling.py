'''

    - Hyunwoong Choi -

    Complex in certain region housing price predict crawling program.



    This program collects complex information in certain city for predicting real-estimate price using machine learning.

    which can be used to sell or to buy reasonable real-estimate price,

    and it can help buyers to form the right spending habits by predicting housing prices.

'''
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.common import exceptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common import exceptions

print('Complex in certain region price predict crawling program - made by Hyunwoong Choi\n\n')

CHROME_PATH = './chromedriver.exe'
URL_PATH = 'https://new.land.naver.com/complexes'
SEARCH_KEYWORD = '서울시 강남구 개포동'
SEARCH_BAR_CLASS_NAME = 'search_input'
COMPLEX_LIST_CSS_SELECTOR_NAME = '#region_filter > div > a > span.area.type_complex'
COMPLEX_AREA_LIST_NAME = 'area_list--complex'


print('Chrome driver loading...')
driver = webdriver.Chrome(CHROME_PATH)
driver.get(URL_PATH)
time.sleep(0.5)
# 코드 중간마다 sleep를 하는 이유는, 웹 페이지가 아직 다 로딩되지 않은 상태에서
# 추가적인 작업을 요청하게 되면 웹 인터럽트가 발생하여 프로그램이 정상적으로 동작하지 못함.


search_bar = driver.find_element_by_class_name(SEARCH_BAR_CLASS_NAME)
search_bar.send_keys(SEARCH_KEYWORD)
search_bar.send_keys(Keys.ENTER)
time.sleep(4)
# 검색 키워드를 자동으로 입력해서 특정 지역으로 포커싱을 맞춤
# 검색 결과가 전부 갱신이 완료될 때 까지, 충분히 기다림

complex_list_view_button = driver.find_element_by_css_selector(COMPLEX_LIST_CSS_SELECTOR_NAME)
driver.execute_script('arguments[0].click();', complex_list_view_button)
# 특정 지역으로 이동한 후, "단지"라는 버튼을 눌러야 판매 중인 주택 목록이 등장함
# 해당 버튼은 javascript로 동작하기 때문에, 이에 해당하는 function 실행

driver.implicitly_wait(5)

all_complex_list = driver.find_element_by_class_name(COMPLEX_AREA_LIST_NAME)
complexs_info = all_complex_list.find_elements_by_xpath('//*[@id="region_filter"]/div/div/div[3]/ul/li')
# 모든 주택 정보를 읽어와서 complexs_info에 저장함

object_complexs = []
total = 0
count = 0
for c in complexs_info:
    time.sleep(0.01)
    sales = c.find_elements_by_class_name('quantity')

    total += 1

    if int(sales[0].text) == 0:# 매매 건에 대해서만 수집하도록 함
        count += 1
        continue

    print('대상 : ', c.text)
    object_complexs.append(c)
# 실제 매매 건수가 존재하는 아파트만 포함하도록 함


object_complexs.reverse()

print('Price Predict Object : ', total - count)
print('Not Price Predict Object : ', count)

#################################################
###### 실제 데이터 수집 시작
#################################################
print('Start crawling....')
for obj in object_complexs:
    ###########################################################
    ###### 앞서 뽑은 아파트 목록 하나씩 소스코드를 갱신하며 가져옴
    ###########################################################

    time.sleep(1)
    body = driver.find_element_by_id('app')
    # 갱신된 웹 페이지 내용을 새로 가져옴 - 가장 중요한 부분
    time.sleep(0.3)

    print(obj.text)
    obj.click()

    filter_button = body.find_elements_by_class_name('list_filter_btn')[0]
    filter_button.click()
    time.sleep(0.1)
    body.find_element_by_xpath('//*[@id="complexOverviewList"]/div/div[1]/div[2]/div/div[1]/div/div/ul/li[2]/label').click()
    # 필터 클래스를 추출하여, 매매 대상 매물만 보이도록 조정함
    body.find_element_by_xpath('// *[ @ id = "complexOverviewList"] / div / div[1] / div[2] / div / div[1] / div / button').click()
    time.sleep(1)

    obj_lists = body.find_element_by_xpath('//*[@id="articleListArea"]')
    objs = obj_lists.find_elements_by_class_name('item_inner')

    for o in objs:
        try:
            time.sleep(0.5)
            if o.text.__contains__('부동산114') or o.text.__contains__('부동산114제공')\
                    or o.text.__contains__('공인중개사협회매물'):
                # 외부 링크로 연결되는 매물의 경우에는 데이터 수집을 하지 않음.
                continue

            print(o.text)

            o.click()

            new_body = driver.find_element_by_id('app')
            # 새롭게 갱신된 웹 페이지 내용을 다시 가져옴
            time.sleep(1)

            complex_information = new_body.find_element_by_class_name('complex_feature')
            infos = complex_information.text.split('\n')

            build_time = None

            for i in range(0, len(infos)):
                if infos[i].__contains__('준공'):
                    build_time = infos[i+1] + '01'
                    build_time = build_time.replace('.', '-')
                    # timestamp 형식에 맞게 하기 위해 ####.##.01 형식으로 지정
                    break

            detail_contents = new_body.find_element_by_class_name('detail_contents_inner')

            strings = detail_contents.text.split('\n')

            ## 데이터 수집시 중복 제거에 사용되는 Flag 변수 ##
            subwayFlag = True
            areaFlag = True
            roomFlag = True
            floorFlag = True
            parkingFlag = True
            totalTaxFlag = True
            # 해당 정보는 본문에도 있고, 공인중개사 코멘트에도 포함되어 있을 수도 있기 때문에,
            # 공인중개사 코멘트 정보는 포함하지 않고 수집하기 위한 Flag 변수
            ###############################################


            windDirection = 0
            # 각 주택의 풍향 정보
            # 0 - 알 수 없음, 1 - 남, 2 - 동, 3 - 서, 4 - 북, 5 - 남동, 6 - 남서, 7 - 북서, 8 - 북동
            address = SEARCH_KEYWORD + ' ' + strings[1] # 매매 대상 주소
            tax = None # 취득세 값
            subway = None # 역까지의 거리 값
            wind = None # 풍향 정보
            room = None # 방 개수
            bathRoom = None # 욕실 개수
            totalParking = None # 총 주차 공간 개수
            personalParking = None # 세대 당 주차 공간 평균
            objFloor = None # 매매 대상 층 수
            totalFloor = None # 해당 건물 전체 층 수
            supplyArea = None # 공급 면적
            actualArea = None # 전용 면적
            totalPrice = None # 매매 가격
            perPrice = None # 평당 가격
            areaRatio = None # 전용률

            # 매매 가격 및 평당 가격 추출
            # 실제 데이터 : 1. 매매##억 #,###(#,###만원/3.3㎡)
            # 실제 데이터 : 2. 매매##억(#,###만원/3.3㎡)
            price = strings[2].split('(')
            totalPrice = price[0]
            # 매매##억 #,###
            totalPrice = totalPrice[2:]
            # ##억 #,###

            if totalPrice.__contains__(' '):
                totalPrice = totalPrice.split('억 ')[0] + totalPrice.split('억 ')[1]
            else:
                totalPrice = totalPrice[:-1] + '0000'
            # ###,###
            totalPrice = totalPrice.replace(',', '')
            # ######
            totalPrice = int(totalPrice)

            perPrice = price[1].split('만')[0]
            perPrice = perPrice.replace(',', '')
            perPrice = int(perPrice)

            print('------------------------------------')
            print('매매 주소 : ', address)
            print('준공년월 : ', build_time, type(build_time))
            print('매매 가격 : ', totalPrice, type(totalPrice))
            print('평당 가격 : ', perPrice, type(perPrice))

            for s in strings:
                # string[1]은 무조건 매물 주소
                # string[2]는 무조건 매매+가격(평당 가격)
                if s.__contains__('역까지') and subwayFlag:
                    # 실제 데이터 : #향역까지 #분 공급/전용 면적:###.##㎡/##.##㎡
                    subway = s.split('분')[0]

                    if s.__contains__('향'):
                        # 풍향 정보가 있는 경우,
                        windDirection = s.split('향')[0]

                        # 0 - 알 수 없음, 1 - 남, 2 - 동, 3 - 서, 4 - 북, 5 - 남동, 6 - 남서, 7 - 북서, 8 - 북동
                        if windDirection == '남':
                            wind = 1
                        elif windDirection == '동':
                            wind = 2
                        elif windDirection == '서':
                            wind = 3
                        elif windDirection == '북':
                            wind = 4
                        elif windDirection == '남동' or windDirection == '동남':
                            wind = 5
                        elif windDirection == '남서' or windDirection == '서남':
                            wind = 6
                        elif windDirection == '북서' or windDirection == '서북':
                            wind = 7
                        elif windDirection == '북동' or windDirection == '동북':
                            wind = 8
                    else:
                        # 풍향 정보가 없는 경우
                        wind = 0

                    subway = subway.split('역까지 ')[1]
                    subway = int(subway)

                    print('풍향 : ', wind, type(wind))
                    print('역까지 거리(분) : ', subway, type(subway))

                    subwayFlag = False


                if s.__contains__('공급/전용면적') and areaFlag:
                    # 실제 데이터 : 공급/전용면적 ##.##㎡/##.##㎡(전용률72%)
                    area = s.split('㎡/')

                    supplyArea = area[0].split(' ')[1]
                    supplyArea = float(supplyArea)

                    actualArea = area[1].split('㎡(')[0]
                    actualArea = float(actualArea)

                    areaRatio = (actualArea / supplyArea) * 100

                    print('공급 면적 : ', supplyArea, type(supplyArea))
                    print('전용 면적 : ', actualArea, type(actualArea))
                    print('전용률 : ', areaRatio, type(areaRatio))

                    areaFlag = False

                if s.__contains__('방수/욕실수') and floorFlag and roomFlag:
                    # 해당 분기에서 해당층/총층 , 방수/욕실수 둘 다 필터링이 됨.
                    # 실제 데이터 : 해당층/총층 ##/##층 방수/욕실수 #/#개

                    floor = s.split('층 방수/욕실수 ')[0]
                    # 해당층/총층 ##/##
                    room_full = s.split('층 방수/욕실수 ')[1]
                    # #/#개

                    floor = floor.split('층 ')[1]

                    totalFloor = floor.split('/')[1]
                    totalFloor = int(totalFloor)

                    objFloor = floor.split('/')[0]
                    if objFloor == '저':
                        objFloor = int(totalFloor * (1 / 3)) - 1
                    elif objFloor == '중':
                        objFloor = int(totalFloor * (2 / 3)) - 1
                    elif objFloor == '고':
                        objFloor = totalFloor - 1
                    # 현재 층이 저 중 고로 표기된 경우, 구분이 되는 임의의 정수 값으로 대체

                    objFloor = int(objFloor)

                    room = room_full.split('/')[0]
                    room = int(room)

                    bathRoom = room_full.split('/')[1]
                    bathRoom = bathRoom[:-1]
                    bathRoom = int(bathRoom)

                    print('매매 대상 층 수 : ', objFloor, type(objFloor))
                    print('건물 총 층 수 : ', totalFloor, type(totalFloor))
                    print('매매 대상 방 수 : ', room, type(room))
                    print('매매 대상 욕실 수 : ', bathRoom, type(bathRoom))

                    roomFlag = False
                    floorFlag = False

                if s.__contains__('총주차대수') and parkingFlag:
                    # 실제 데이터 : 입주가능일 2개월이내 총주차대수 41대(세대당 0.71대)

                    park = s.split('총주차대수 ')[1]

                    if park.__contains__('-'):
                        totalParking = 0
                        personalParking = 0
                    else:
                        totalParking = park.split('대')[0].strip()
                        totalParking = int(totalParking)

                        park = park.split('(세대당 ')[1]
                        personalParking = float(park[:-2])

                    print('총 주차 대수 : ', totalParking, type(totalParking))
                    print('세대당 주차 대수 평균 : ', personalParking, type(personalParking))

                    parkingFlag = False

                if s.__contains__('취득세') and totalTaxFlag:
                    # 실제 데이터 : 취득세 총 약 #,###만원
                    tax = s.split('약 ')[1]

                    millionFlag = False
                    if tax.__contains__('억'):
                        tax = tax.split('억 ')[0] + tax.split('억 ')[1]
                        millionFlag = True

                    if tax.__contains__('만원'):
                        tax = tax.split('만원')[0].strip()

                    if tax.__contains__('만'):
                        tax = tax.split('만')[0]

                    if tax.__contains__(','):
                        tax = tax.split(',')[0] + tax.split(',')[1]
                    else:
                        if millionFlag:
                            tax = tax[0] + '0' + tax[1:]

                    tax = int(tax)
                    # 가공 데이터 : #### -> type = int

                    print('총 취득세 정보 : ', tax, type(tax))

                    totalTaxFlag = False

            print('------------------------------------\n')

            print('Ready for write on csv file what crawled information in web..')

            with open('korea_housing.csv', 'a') as f:
                one_line = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(address, build_time, totalPrice,
                                                                                    perPrice, wind, subway, supplyArea,
                                                                                    actualArea, areaRatio, objFloor,
                                                                                    totalFloor, room, bathRoom,
                                                                                    totalParking, personalParking, tax)

                f.write(one_line)
            print('Finish writing..')
        except exceptions.StaleElementReferenceException as e:
            print(e)
            continue

    time.sleep(0.3)
    close_button = body.find_element_by_xpath('// *[ @ id = "ct"] / div[2] / div[2] / div / button')
    close_button.click()

    time.sleep(0.3)
    driver.execute_script('arguments[0].click();', complex_list_view_button)

    # except exceptions.StaleElementReferenceException as e:
    #     print(address + ' : ', e)
    #     pass

print('Finish Crawling program.')
