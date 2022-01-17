import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

HOUSING_PATH = 'korea_housing.csv'

def load_housing_data(filePath=HOUSING_PATH):
    return pd.read_csv(filePath, encoding='CP949')

def display_scores(scores):
    print('Scores : ', scores)
    print('Means : ', scores.mean())
    print('Standard deviation : ', scores.std())

def display_plot_feature_importance(model):
    n_features = housing.shape[1]
    plt.barh(np.arange(n_features), sorted(model.feature_importances_), align='center')
    plt.yticks(np.arange(n_features), housing.feature_names)
    plt.xlabel('Random')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

print('Read csv...')
housing = load_housing_data(HOUSING_PATH)
print(housing.head)

print('Housing Information')
print(housing.info())

housing = housing.drop(['address', 'build_date'], axis=1)

print('After Preprocess Information')
print(housing.info())

housing.hist(bins=50, figsize=(50, 50))
plt.show()

train, test = train_test_split(housing, test_size=0.2, random_state=1)

housing.plot(kind='scatter', x='longtitude', y='latitude', alpha=0.4,
             colorbar=True, sharex=False)
plt.show()

corr_matrix = housing.corr()


print(corr_matrix)

print('---- sale price corr ----')
print(corr_matrix['sale_price'])

print('---- obj_floor corr ----')
print(corr_matrix['obj_floor'])

print('---- distance_from_subway corr ----')
print(corr_matrix['distance_from_subway'])

#### 풍향 범주형 데이터 변환 작업 ####
wind = housing["wind_direction"]
wind_encoded, wind_categories = wind.factorize()
encoder = OneHotEncoder(categories='auto')

wind_onehot = encoder.fit_transform(wind_encoded.reshape(-1, 1))

y_data = train['sale_price']
x_data = train.drop(['sale_price'], axis=1)
print(test[0:3])
y_test = test['sale_price']
x_test = test.drop(['sale_price'], axis=1)

####################### 선형 회귀 분석 ##############################
linear_reg = LinearRegression()
linear_reg.fit(x_data, y_data)
result = linear_reg.predict(x_test)
print('\n\n---------------------------')
print("Linear Regressor Prediction : ", result[0:3])
# 테스트 셋에 따라 거의 오차가 없이 sale_price 값을 예측하는 것을 확인
####################################################################

####################### 결정 트리 회귀 분석 ##########################
decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(x_data, y_data)
result = decision_tree_reg.predict(x_test)
print('\n\n---------------------------')
print('Decision Tree Prediction : ', result[0:3])
# 실제 데이터와 추론 데이터 값이 아예 동일하기 때문에, 과적합이 우려됨
tree_mse = mean_squared_error(y_test, result)
tree_mse = np.sqrt(tree_mse)
print('Decision Tree mse : ', tree_mse)
####################################################################


## 결정 트리 k-fold 교차 검증
scores = cross_val_score(decision_tree_reg, x_data, y_data,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse = np.sqrt(-scores)
display_scores(scores)


## 랜덤 포레스트 및 k-fold 교차 검증
random_forest = RandomForestRegressor()
random_forest.fit(x_data, y_data)
result = random_forest.predict(x_test)
print('\n\n---------------------------')
print('Random Forest Prediction : ', result[0:3])
tree_mse = mean_squared_error(y_test, result)
tree_mse = np.sqrt(tree_mse)
display_scores(tree_mse)
# 결정 트리와 달리 예측 값이 같지 않아, 과적합의 우려가 적음

# Extremely Randomized Tree는 Random Forest의 극도의 랜덤 버젼으로
# RF의 노드 분할 기준은 'best'인 반면, ERT는 'random'임
# 최종적으로 ERT는 RF보다 데이터의 특성을 더 중요하게 여김
extra_tree = ExtraTreesRegressor()
extra_tree.fit(x_data, y_data)
result = extra_tree.predict(x_test)
print('\n\n---------------------------')
print('Extra Tree Prediction : ', result[0:3])
extra_tree_mse = mean_squared_error(y_test, result)
extra_tree_mse = np.sqrt(extra_tree_mse)
display_scores(extra_tree_mse)


print('decision tree feature importance')
for feat, importance in zip(housing.columns, decision_tree_reg.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))
print()

print('random forest feature importance')
for feat, importance in zip(housing.columns, random_forest.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))
print()

print('extra tree feature importance')
for feat, importance in zip(housing.columns, extra_tree.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))
print()
