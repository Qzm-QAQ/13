# -*- coding: utf-8 -*-
"""
一元重复测量 ANOVA（通用版）+ K-means 聚类
行 = 被试/学生，列 = 条件/科目
自动计算 Treatment, Subject, Residual, Total 的 SS / df / MS / F / p
并自动对学生成绩做 K-means 聚类 (k=3)
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans   # ★ 新增：用于聚类

# ============================================================
# 数据在这里修改：每行一个学生，每列一门科目
# ============================================================
scores = np.array([

  [88, 69, 80],  # Student 1
    [83, 87, 77],  # Student 2
    [68, 73, 60],  # Student 3
    [89, 83, 49],  # Student 4
    [80, 81, 75],  # Student 5
    [66, 67, 94],  # Student 6
    [80, 66, 50],  # Student 7
    [60, 50, 28],  # Student 8
    [96, 62, 71],  # Student 9
    [61, 75, 90],  # Student 10
], dtype=float)

subject_names = ["Math", "Science", "History"]


def repeated_measures_anova(scores):
    n, k = scores.shape  # n 学生数，k 科目数

    # ====== 计算均值 ======
    grand_mean = scores.mean()           # 全体均值
    subject_means = scores.mean(axis=1)  # 每个学生的平均
    treat_means = scores.mean(axis=0)    # 每个科目的平均

    # ====== 均值汇总 ======
    print("===== MEANS SUMMARY =====")
    print(f"Grand Mean (全体平均) = {grand_mean:.4f}\n")

    print("Subject Means（每个学生的平均）:")
    for i, m in enumerate(subject_means):
        print(f"  Student {i+1}: {m:.4f}")
    print()

    print("Condition / Subject Means（每科平均）:")
    for j in range(k):
        name = subject_names[j] if len(subject_names) == k else f"Cond{j+1}"
        print(f"  {name}: {treat_means[j]:.4f}")
    print()

    # ====== SS（平方和） ======
    SS_treat = np.sum(n * (treat_means - grand_mean) ** 2)
    SS_subject = np.sum(k * (subject_means - grand_mean) ** 2)
    SS_total = np.sum((scores - grand_mean) ** 2)
    SS_res = SS_total - SS_treat - SS_subject

    # ====== df ======
    df_treat = k - 1
    df_subject = n - 1
    df_res = df_treat * df_subject
    df_total = n * k - 1

    # ====== MS ======
    MS_treat = SS_treat / df_treat
    MS_subject = SS_subject / df_subject
    MS_res = SS_res / df_res

    # ====== F & p ======
    F = MS_treat / MS_res
    p = 1 - stats.f.cdf(F, df_treat, df_res)

    # ====== 输出 ANOVA ======
    print("\n===== One-way Repeated-Measures ANOVA =====")
    print(f"被试 n = {n}, 条件数 k = {k}")
    print(f"F = MS_Treat / MS_Residual = {MS_treat:.4f} / {MS_res:.4f} = {F:.4f}")
    print(f"p-value = {p:.4f}\n")

    print("===== ANOVA Table (repeated-measures) =====")
    print(f"{'Source':<12} {'SS':>10} {'df':>6} {'MS':>12} {'F':>10} {'p':>10}")
    print(f"{'Treatment':<12} {SS_treat:10.4f} {df_treat:6} {MS_treat:12.4f} {F:10.4f} {p:10.4f}")
    print(f"{'Subject':<12} {SS_subject:10.4f} {df_subject:6} {MS_subject:12.4f}")
    print(f"{'Residual':<12} {SS_res:10.4f} {df_res:6} {MS_res:12.4f}")
    print(f"{'Total':<12} {SS_total:10.4f} {df_total:6}")

    # ============================================================
    # ★★★ 自动聚类部分（第二问） ★★★
    # ============================================================
    print("\n===== K-means Clustering (k = 3) =====")
    # 使用三科成绩向量做聚类
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(scores)   # labels: 0,1,2

    # 计算每个簇的平均总分，用来根据“大小”排序
    cluster_info = []
    for c in range(3):
        idx = np.where(labels == c)[0]
        cluster_mean = scores[idx].mean()   # 该簇整体平均成绩
        cluster_info.append((c, cluster_mean, idx))

    # 按平均成绩从低到高排序 => Group 1: 低, Group 2: 中, Group 3: 高
    cluster_info.sort(key=lambda x: x[1])

    for rank, (c, mean_val, idx) in enumerate(cluster_info, start=1):
        print(f"\nGroup {rank} (cluster {c}, average score = {mean_val:.2f}):")
        for i in idx:
            per_subj = ", ".join(f"{subject_names[j]}={scores[i, j]:.1f}" for j in range(k))
            print(f"  Student {i+1}: {per_subj}")

    # ============================================================
    # ★★★ 通解文字模板（方便抄到答案） ★★★
    # ============================================================
    print("\n\n===== ANSWER TEMPLATE (FOR REPORT) =====")
    print("(1) ANOVA:")
    print(f"  Means: {subject_names[0]} = {treat_means[0]:.2f}, "
          f"{subject_names[1]} = {treat_means[1]:.2f}, "
          f"{subject_names[2]} = {treat_means[2]:.2f}")
    print(f"  F({df_treat},{df_res}) = {F:.2f}, p = {p:.2f}")
    if p < 0.05:
        print("  → There IS a significant difference among the three subjects.")
    else:
        print("  → There is NO significant difference among the three subjects.")

    print("\n(2) Clustering:")
    print("  Each student is represented by a 3-dimensional vector "
          "(Math, Science, History).")
    print("  Using K-means clustering with k = 3, the students are divided into")
    print("  three groups according to their grade vectors, as listed above.")


if __name__ == "__main__":
    repeated_measures_anova(scores)
