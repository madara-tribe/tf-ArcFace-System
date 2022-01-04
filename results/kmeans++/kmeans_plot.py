import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
def kmaens_plot(X, y, marker_size=200):
    markers=['o', 'v', '<', '>', '8', 's', 'p', '*', 'h']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    km = KMeans(n_clusters=5,            # クラスターの個数
                init='k-means++',           # セントロイドの初期値をランダムに設定  default: 'k-means++'
                n_init=10,               # 異なるセントロイドの初期値を用いたk-meansの実行回数 default: '10' 実行したうちもっとSSE値が小さいモデルを最終モデルとして選択
                max_iter=10000,            # k-meansアルゴリズムの内部の最大イテレーション回数  default: '300'
                tol=1e-04,               # 収束と判定するための相対的な許容誤差 default: '1e-04'
                random_state=4)          # セントロイドの初期化に用いる乱数発生器の状態
    #X = MinMaxScaler().fit_transform(X)
    print(X.max(), X.min())
    y_km = km.fit_predict(X)
    
    print('plot when y max is 4')
    plt.scatter(X[y_km==0,0],         # y_km（クラスター番号）が0の時にXの0列目を抽出
                    X[y_km==0,1], # y_km（クラスター番号）が0の時にXの1列目を抽出
                    s=marker_size,
                    c='lightgreen',
                    marker='s',
                    label=ID_RACE_MAP[0])
    plt.scatter(X[y_km==1,0],
                        X[y_km==1,1],
                        s=marker_size,
                        c='orange',
                        marker='o',
                        label=ID_RACE_MAP[1])
    plt.scatter(X[y_km==2,0],
                       X[y_km==2,1],
                        s=marker_size,
                        c='lightblue',
                        marker='v',
                        label=ID_RACE_MAP[2])

    plt.scatter(X[y_km==3,0],
                       X[y_km==3,1],
                        s=marker_size,
                        c='black',
                        marker='o',
                        label=ID_RACE_MAP[3])

    plt.scatter(X[y_km==4,0],
                       X[y_km==4,1],
                        s=marker_size,
                        c='blue',
                        marker='o',
                        label=ID_RACE_MAP[4])


    plt.scatter(km.cluster_centers_[:,0],   # km.cluster_centers_には各クラスターのセントロイドの座標が入っている
                        km.cluster_centers_[:,1],
                        s=200,
                        marker='*',
                        c='red',
                        label='centroids')
    plt.legend()
    plt.grid()
    plt.savefig('kmeans++.png')
    plt.show()
    
    
    
if __name__=='__main__':
    X = np.load('X_embedding.npy')
    y = np.load('y_embedding.npy')
    print(y.shape, X.shape)
    kmaens_plot(X, y)
