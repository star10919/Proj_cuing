import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from itertools import chain

# 영양성분 크롤링
 
class Nutrition(object):
    # url = 'https://terms.naver.com/list.naver?cid=59320&categoryId=59320'
    url = None
    driver_path = 'c:/Program Files/Google/Chrome/chromedriver'
    dict = {}
    df = None
    food_name = []
    food_nut = []
    new_food_nut = []
    new_food_gram = []
    new_food_kcal = []
    final_food_nut = []
    

    def scrap_name(self):

        for i in range(1, 3):
            self.url = f'https://terms.naver.com/list.naver?cid=59320&categoryId=59320&page={i}'  # 1 페이지부터 3페이지 차례대로 링크로 이동
            driver = webdriver.Chrome(self.driver_path)
            driver.get(self.url)
            all_div = BeautifulSoup(driver.page_source, 'html.parser')
            ls1 = all_div.find_all("div", {"class": "subject"})     # name
            for i in ls1:
                self.food_name.append(i.find('a').text) # 현재 페이지에 보이는 음식 이름 하나씩 가져오기
            # print(self.food_name)


            ls2 = all_div.find_all("p", {"class": "desc __ellipsis"})   # nutrition
            for i in ls2:
                self.food_nut.append(i.text) # 현재 페이지에 보이는 영양성분 정보(탄수화물, 단백질, 지방, 당류, 나트륨 등) 하나씩 가져오기
            
            # ls3 = all_div.find_all("div", {"class": "related"})
            ls3 = all_div.find_all("span", {"class": "info"})        # 1회 제공량, 칼로리
            
            for i, j in enumerate(ls3):
                # print(f'{i} // {j.text}')
                if '1회제공량' in j.text:
                    self.new_food_gram.append(j.text)
                elif '칼로리' in j.text:
                    self.new_food_kcal.append(j.text)
                else:
                    pass
            
            # print(len(self.food_name))  # 16
            # print(len(self.food_nut))   # 15
            self.food_name.remove('인문과학')     # 불필요한 요소 제거
            # print(len(self.food_name))  # 15
            
            for i in self.food_nut:
                temp = i.replace('\n', '').replace('\t', '').replace(' ', '').replace('[영양성분]', '').replace('조사년도', '').replace('지역명전국(대표)', '').replace('자료출처식약처영양실태조사', '')     # 불필요한 요소 제거
                self.new_food_nut.append(temp)
            # print(self.new_food_nut)
            
            for i, j, k in zip(self.new_food_nut, self.new_food_gram, self.new_food_kcal):
                temp = i + ',' + j + ',' + k
                self.final_food_nut.append(temp)                # nutrition 1회 제공량, 칼로리, 영양성분을 하나의 변수로 합침
            # print(self.final_food_nut)
            
            # print(self.new_food_gram)
            # print(self.new_food_kcal)
            # print(len(self.food_name))
            # print(len(self.new_food_gram))
            # print(len(self.new_food_kcal))
            
            for i, j in enumerate(self.food_name):
                # print('i,j :\n',i, j)
                # print('name :\n',self.food_name[i])
                # print('nutrition :\n',self.food_nut[i])
                self.dict[self.food_name[i]] = self.final_food_nut[i] # 음식 이름과 영양성분을 병합해서 딕셔너리로 만듬 {'고구마': '- 탄수화물...'}
            driver.close()




        # 영양성분을 하나의 똑같은 틀에 넣기 위한 작업
        food_ls = []
        unique_ls = [] # 유니크 값을 확인하기 위한 배열 -> 모든 음식에서 기재되어있는 영양성분이 다르기 때문에 유니크 값을 뽑아내기 위함
        for key, value in self.dict.items(): # 크롤링 한 음식 가지수만큼 반복
            nut_tr = self.dict[key].split('-')[:-1] # 1회 제공량이나 칼로리부터는 '-'문자열이 없고, ','로 구분 돼서 그 전까지 분리 ['', '탄수화물:23g', '단백질:123g']
            nut_tr = ' '.join(nut_tr).split() # 앞 공백 요소 제거
            print('nut_tr', nut_tr)
            kal_tr = self.dict[key].split('-')[-1].split(',') # 1회 제공량, 칼로리는 ','로 각 요소가 분리돼서 따로 분리
            kal_tr = [x.replace(' ', ':') for x in kal_tr] # 제공량, 칼로리는 해당 영양성분에 대한 내용이 띄어쓰기로 분류되어있어서 한 번에 바꾸기 위해 replace시켜줌
            print('kal_tr', kal_tr)
            ls = nut_tr[:] + kal_tr # 둘이 합침
            print('ls', ls)
            new_dict = {sub.split(":")[0]: sub.split(":")[1] for sub in ls[:]} # {'탄수화물':5g, '단백질':4g} 처럼 모든 영양성분을 키와 값으로 나눔
            new_dict.update({'음식명' : key}) # '음식명'(컬럼 추가)도 영양성분 dictioanry와 합침('음식명' : '고구마', '탄수화물': 5g ~)  /  dictionary는 update(list에서 append)
            print('new_dict',new_dict)
            unique_ls.append(list(new_dict.keys())) # 유니크 값을 확인하기 위한 배열에 키값을 추가함.
            unique_value = ['나트륨', '포화지방산', '당류', '음식명', '1회제공량', '지방', '콜레스테롤', '탄수화물', '단백질', '트랜스지방', '칼로리'] # 크롤링한 데이터에서의 유니크값 들
            for i, j in enumerate(unique_value): # 유니크값만큼 돌아서 비교함.
                if (j in list(new_dict.keys())) == False: # 영양성분이 존재하지 않으면 추가
                    new_dict.update({j : '0g'}) # 데이터프레임으로 만들기 위해 해당성분에 없는 영양성분은 0g으로 추가해줌 -> 선생님이 물어보실듯

            food_ls.append(new_dict) # 만들어진 음식 dictionary를 하나씩 리스트 자료형에 추가

        # 모든영양성분(컬럼값) 누락하지 않기 위해서
        unique_ls = list(chain.from_iterable(unique_ls)) # 2d -> 1d(리스트로 바꿔야 셋으로 바꿀 수 있음)
        test = set(unique_ls) # 유니크 값 출력(set : 중복값 허용되지 않음)
        # test = set(unique_ls) # 유니크 값 출력
        # print('test', test)
        for i in food_ls:
            print('testtest',i)
        
        food = pd.DataFrame([i for i in food_ls], columns=['음식명', '나트륨', '포화지방산', '당류', '1회제공량', '지방', '콜레스테롤', '탄수화물', '단백질', '트랜스지방', '칼로리']) # dataframe 자료형으로 변환
    
        food.to_csv('food.csv', index=False)
    '''
        for i in ls:
            print()
            self.food_nut.append(i.find("p").text)
        driver.close()
    def insert_dict(self):
        for i, j in zip(self.food_name, self.food_nut):
            self.dict[i] = j
            print(f'{i}:{j}')
    def dict_to_dataframe(self):
        dt = self.dict
        self.df = pd.DataFrame.from_dict(dt, orient='index')
        print(self.df)
    def df_to_csv(self):
        path = './data/food_nutrition.csv'
        self.df.to_csv(path, sep=',', na_rep='Nan')
    '''
    @staticmethod
    def main():
        nut = Nutrition()
        nut.scrap_name()

Nutrition.main()