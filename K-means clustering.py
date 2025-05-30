import pandas as pd
from sklearn.cluster import KMeans

# 파일 불러오기
df = pd.read_csv("Preprocessing_Insurance_Data.csv")

# Personal Score 기준 클러스터링
kmeans_personal = KMeans(n_clusters=3, random_state=0)
df['Cluster_Personal'] = kmeans_personal.fit_predict(df[['Personal_Score']])

# Health 관련 클러스터링
health_features = ['Smoking Status', 'Exercise Frequency', 'Health Score']
kmeans_health = KMeans(n_clusters=3, random_state=42)
df['Cluster_Health'] = kmeans_health.fit_predict(df[health_features])

# 기타요소 관련 클러스터링
other_features = ['Insurance Duration', 'Credit Score']
kmeans_other = KMeans(n_clusters=3, random_state=42)
df['Cluster_Credit'] = kmeans_other.fit_predict(df[other_features])

# 저장
df.to_csv("Preprocessing_Insurance_Data_with_clusters.csv", index=False)
print("✔ 클러스터링 완료 및 저장됨.")
