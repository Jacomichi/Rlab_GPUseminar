# Rlab GPU seminar (2021 spring semester)
R研のGPUセミナー用のリポジトリ。

CUDAの基本的な書き方から初めて分子動力学計算を自力でようになるところまでの内容。

セミナーのスライドおよびノートは別途渡します。

## シラバス
## 1. Hello World
- CUDAの基本的な書き方。
- カーネル関数の書き方。
## 2. グローバルメモリーを使ったベクトルの足し算
- グローバルメモリーの確保
- スレッドインデックスを使った演算
## 3. Unifiedメモリーを使ったベクトルの足し算
- Unifiedメモリーの使い方
- Unifiedメモリーの高速化
- ex)行列の積
## 4. 行列の転置
- グローバルメモリーを使った行列の転置
- Sharedメモリーを使った行列の転置
- バンクコンフリクトとパディング
## 5. CuRand
- CuRandのホストAPIの使い方
- CuRandのデバイスAPIの使い方
- 一様分布の平均
- モンテカルロ法を用いた円周率の計算
## 6. Thrust
- device_vector
- transform
- reduce and sort
- Fancy Iterator
## 7. リダクション
- リダクションの基本的な書き方
- Sharedメモリーの活用
- Cooperative groupとwarp shuffle命令
## 8. CUDAストリーム
- CUDAストリームの使い方
- CUDAストリームを使ったベクトルの足し算

## 9. 自由な1次元ブラウン粒子の運動
相互作用のない1次元Langevin方程式の計算
- Kernel関数版
- Thrust版
## 10. 自由な3次元ブラウン粒子の運動
相互作用のない3次元Langevin方程式の計算
## 11. ヒストグラムの作成
バケットソートを用いたヒストグラムの作成
## 12. Verlet listの作成
- 全粒子検索バージョン($O(N^2)$)
- グリッド検索バージョン($O(N)$)
## 13. Many body Langevin equation
- without verlet list
- with verlet list
## 14. Molecular Dyanamics simulation (Velocity Verlet)
- Harmonic potential
- Softcore
- WCA

## TODO
- FIRE法を用いたエネルギー最小化
- 剪断変形
