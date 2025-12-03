# -*- coding: utf-8 -*-
"""
S-P 分析 + PROX 法（通用版）
(4) S-P analysis
(5) PROX method
行: 学习者 / learners
列: 题目 / items
"""

import numpy as np
import math

# --------------------------------------------------
# 1. 在这里填你的 0/1 数据（0 = × incorrect, 1 = ○ correct）
#    下面是 Table 3 的例子，可以直接改成别的数据
# --------------------------------------------------
answers = np.array([
    # Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10
    [0, 0, 1, 0, 1, 1, 1, 1, 0, 1],  # learner A
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],  # learner B
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],  # learner C
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # learner D
    [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],  # learner E
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # learner F
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0],  # learner G
    [1, 1, 0, 1, 1, 1, 1, 0, 0, 1],  # learner H
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],  # learner I
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],  # learner J
], dtype=int)

# 自动推导 N, K 以及学生名和题目名（通解关键）
N, K = answers.shape
students = [chr(ord('A') + i) for i in range(N)]        # A, B, C, ...
items = [f"Q{i+1}" for i in range(K)]                   # Q1, Q2, ...


def sp_and_prox_analysis(answers, students, items):
    N, K = answers.shape
    row_sum = answers.sum(axis=1)   # 每个学生总分 / total score of each learner
    col_sum = answers.sum(axis=0)   # 每题正答人数 / #correct of each item

    # 学生按总分从高到低；题目按正答人数从多到少
    stu_order = sorted(range(N), key=lambda i: (-row_sum[i], students[i]))
    item_order = sorted(range(K), key=lambda j: -col_sum[j])

    # --------------------------------------------------
    # (4) S-P 过程表 / S-P process table
    # --------------------------------------------------
    print("===== (4) S-P Analysis / S-P 分析：过程表 =====")
    header = "Student | " + " ".join(f"{items[j]:>3}" for j in item_order) + " | Sum"
    print(header)
    print("-" * len(header))
    for i in stu_order:
        row = answers[i, item_order]
        row_str = " ".join(f"{v:>3d}" for v in row)
        print(f"   {students[i]}    | {row_str} | {int(row_sum[i])}")
    print("-" * len(header))
    total_str = " ".join(f"{int(col_sum[j]):>3d}" for j in item_order)
    print(f" Total  | {total_str} |")
    print()

    # --------------------------------------------------
    # (4) 中文+英文答案模板
    # --------------------------------------------------
    print("===== (4) Answer Template / 解答模板 =====")

    # 1. 学习者总分
    sorted_learners = [(students[i], int(row_sum[i])) for i in stu_order]
    desc_learner_cn = "，".join(f"{s}：{sc}分" for s, sc in sorted_learners)
    desc_learner_en = ", ".join(f"{s}: {sc} points" for s, sc in sorted_learners)
    print("1) Learner total scores (high→low) / 学习者总分（从高到低）：")
    print("   " + desc_learner_en)
    print("   " + desc_learner_cn)
    print()

    # 2. 各题正答人数（原始顺序）
    correct_cn = "，".join(f"{items[j]}：{int(col_sum[j])}人正确"
                           for j in range(K))
    correct_en = ", ".join(f"{items[j]}: {int(col_sum[j])} correct"
                           for j in range(K))
    print("2) #Correct per item / 各题正答人数：")
    print("   " + correct_en)
    print("   " + correct_cn)
    print()

    # 3. 题目难度概括
    max_c = int(col_sum.max())
    min_c = int(col_sum.min())
    easiest_items = [items[j] for j, c in enumerate(col_sum) if c == max_c]
    hardest_items = [items[j] for j, c in enumerate(col_sum) if c == min_c]
    medium_items = [items[j] for j, c in enumerate(col_sum)
                    if c not in (max_c, min_c)]

    print("3) Item difficulty summary / 题目难度概括：")
    print(f"   EN: Easiest item(s): {', '.join(easiest_items)} "
          f"({max_c}/{N} correct).")
    print(f"       Hardest item(s): {', '.join(hardest_items)} "
          f"({min_c}/{N} correct).")
    if medium_items:
        print(f"       Other items have medium difficulty: "
              f"{', '.join(medium_items)}.")
    print(f"   CN: 最容易的题是 {', '.join(easiest_items)}，"
          f"{max_c} 人答对；最难的题是 {', '.join(hardest_items)}，"
          f"只有 {min_c} 人答对；其余题目为中等或中等偏难："
          f"{', '.join(medium_items)}。")
    print()

    # 4. 特征学习者 / 特征题目
    print("4) Characteristic learners & items / 特征学习者和特征题目：")
    highest = [students[i] for i in range(N) if row_sum[i] == row_sum.max()]
    lowest = [students[i] for i in range(N) if row_sum[i] == row_sum.min()]
    print(f"   EN: Highest-scoring learner(s): {', '.join(highest)} "
          f"({int(row_sum.max())}/{K}).")
    print(f"       Lowest-scoring learner(s): {', '.join(lowest)} "
          f"({int(row_sum.min())}/{K}).")
    print(f"   CN: 总分最高的学习者是 {', '.join(highest)}，"
          f"总分最低的是 {', '.join(lowest)}。")

    # 高分但在易题上错
    easiest_idx = [j for j, c in enumerate(col_sum) if c == max_c]
    high_idx = [i for i in range(N) if row_sum[i] >= row_sum.mean()]
    odd_high = [students[i] for i in high_idx
                if any(answers[i, j] == 0 for j in easiest_idx)]
    if odd_high:
        print(f"   EN: For example, learner(s) {', '.join(odd_high)} "
              f"have relatively high total scores but still miss very easy "
              f"items such as {', '.join(easiest_items)}, so they are "
              f'characteristic in the S-P table.')
        print(f"   CN: 例如学习者 {', '.join(odd_high)} 得分较高，"
              f"但在非常容易的题（如 {', '.join(easiest_items)}）上仍有错误，"
              "在 S-P 表中比较显眼。")
    print(f"   EN: The hardest item(s) {', '.join(hardest_items)} are useful "
          "to discriminate learner ability.")
    print(f"   CN: 最难的题 {', '.join(hardest_items)} 对区分学习者能力最有帮助。")
    print()

    # --------------------------------------------------
    # (5) PROX 分析 / PROX method
    # --------------------------------------------------
    print("===== (5) PROX Method / PROX 法分析 =====")
    r = row_sum
    s = col_sum
    eps = 0.5  # 防止 0 或满分导致 log(0)

    r_adj = np.clip(r, eps, K - eps)
    s_adj = np.clip(s, eps, N - eps)

    theta0 = np.log(r_adj / (K - r_adj))         # ability before centering
    b0 = np.log((N - s_adj) / s_adj)             # difficulty before centering

    theta = theta0 - theta0.mean()
    b = b0 - b0.mean()

    theta_r = [round(float(x), 2) for x in theta]
    b_r = [round(float(x), 2) for x in b]

    # 1. 学习者能力值
    ability_sorted = sorted(range(N), key=lambda i: -theta[i])
    print("1) Learner ability θ_i / 学习者能力值（从高到低）：")
    line_en = ", ".join(
        f"{students[i]}: θ={theta_r[i]:.2f} (score {int(r[i])}/{K})"
        for i in ability_sorted
    )
    line_cn = "，".join(
        f"{students[i]}：θ={theta_r[i]:.2f}（得分 {int(r[i])}/{K}）"
        for i in ability_sorted
    )
    print("   " + line_en)
    print("   " + line_cn)
    print()

    # 2. 题目难度值
    diff_sorted = sorted(range(K), key=lambda j: -b[j])
    print("2) Item difficulty b_j / 题目难度值（从难到易）：")
    line_en = ", ".join(
        f"{items[j]}: b={b_r[j]:.2f} ({int(s[j])}/{N} correct)"
        for j in diff_sorted
    )
    line_cn = "，".join(
        f"{items[j]}：b={b_r[j]:.2f}（{int(s[j])}/{N} 人正确）"
        for j in diff_sorted
    )
    print("   " + line_en)
    print("   " + line_cn)
    print()

    # 3. 总结
    top_learners = [students[i] for i in ability_sorted
                    if theta[i] == theta[ability_sorted[0]]]
    bottom_learners = [students[i] for i in ability_sorted
                       if theta[i] == theta[ability_sorted[-1]]]
    hardest_items2 = [items[j] for j in diff_sorted
                      if b[j] == b[diff_sorted[0]]]
    easiest_items2 = [items[j] for j in diff_sorted
                      if b[j] == b[diff_sorted[-1]]]

    print("3) Summary / 总结：")
    print(f"   EN: Highest ability learner(s): {', '.join(top_learners)}; "
          f"lowest ability learner(s): {', '.join(bottom_learners)}.")
    print(f"       Hardest item(s): {', '.join(hardest_items2)}; "
          f"easiest item(s): {', '.join(easiest_items2)}.")
    print("   CN: 能力值最高的学习者是 "
          f"{', '.join(top_learners)}，最低的是 "
          f"{', '.join(bottom_learners)}；最难的题是 "
          f"{', '.join(hardest_items2)}，最容易的题是 "
          f"{', '.join(easiest_items2)}。")
    print("   EN: The PROX results are consistent with the S-P analysis: "
          "high-ability learners answer most items correctly, and the most "
          "difficult item is only solved by a few learners.")
    print("   CN: PROX 结果与 S-P 分析一致：能力高的学习者大部分题目作答正确，"
          "而最难的题只被少数学习者答对。")


if __name__ == "__main__":
    sp_and_prox_analysis(answers, students, items)
