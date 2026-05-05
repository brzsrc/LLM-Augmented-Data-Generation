"""
cluster_analysis.py
===================
基于 cleaned_output.csv 的用户 location 聚类分析。

用每个用户在 weekday/weekend × 5 slots × 16 locations 的概率分布作为特征，
PCA 降维后 KMeans 聚类，输出 5 个行为模式不同的用户组。

输出:
  1. cluster_assignments.csv — 每个用户的 cluster 分配
  2. cluster_location_distributions.json — 每个 cluster × weekday/weekend × slot 的地点分布
  3. cluster_summary.md — 人类可读的聚类摘要
  4. 打印详细分析到控制台

用法:
  python cluster_analysis.py --cleaned /path/to/cleaned_output.csv --users /path/to/users.csv --output ./cluster_output
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def build_feature_matrix(c: pd.DataFrame) -> tuple:
    """
    构建特征矩阵：每用户 = 2(weekday/weekend) × 5(slots) × N_locations 维
    每个维度是该用户在该 (weekday, slot) 条件下出现在该 location 的概率
    """
    c = c.copy()
    c['datetime'] = pd.to_datetime(c['datetime'])
    c['weekday'] = c['datetime'].dt.dayofweek < 5  # Mon-Fri = True

    all_locs = sorted(c['location'].dropna().unique())
    print('all_locs ', all_locs)
    uids = sorted(c['uid'].unique())
    print('uids ', uids)

    feature_rows = []
    for uid in uids:
        ud = c[c['uid'] == uid]
        row = {'uid': uid}

        for wd_label, wd_val in [('wd', True), ('we', False)]:
            for slot in range(1, 6):
                subset = ud[(ud['weekday'] == wd_val) & (ud['day_slot'] == slot)]
                total = len(subset)
                for loc in all_locs:
                    key = f"{wd_label}_s{slot}_{loc}"
                    row[key] = (subset['location'] == loc).sum() / max(total, 1)

        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows)
    feat_cols = [col for col in feat_df.columns if col != 'uid']
    X = feat_df[feat_cols].values

    print(f"特征矩阵: {X.shape[0]} users × {X.shape[1]} features")
    print(f"  = 2 (weekday/weekend) × 5 (slots) × {len(all_locs)} (locations)")
    print(f"  Location 类别: {all_locs}")

    return feat_df, feat_cols, X, all_locs


def evaluate_k(X_scaled: np.ndarray, k_range=range(2, 8), pca_range=[3, 5, 7, 10, 15]):
    """对每个 K，遍历不同 n_pca，找该 K 下最优的 n_pca"""
    print(f"\n{'='*60}")
    print(f"KMeans 聚类评估 (K={k_range.start}..{k_range.stop-1}, n_pca={list(pca_range)})")
    print(f"{'='*60}")

    n_samples = X_scaled.shape[0]
    results = {}  # {k: (best_pca, best_sil)}

    for k in k_range:
        best_pca, best_sil = pca_range[0], -1
        for n_pca in pca_range:
            n_comp = min(n_pca, n_samples - 1)
            pca = PCA(n_components=n_comp)

            X_pca = pca.fit_transform(X_scaled)
            km = KMeans(n_clusters=k, random_state=42, n_init=20)

            labels = km.fit_predict(X_pca)
            sil = silhouette_score(X_pca, labels)
            if sil > best_sil:
                best_sil = sil
                best_pca = n_pca

        # 用最优 n_pca 跑一次拿 sizes
        pca = PCA(n_components=min(best_pca, n_samples - 1))
        X_pca = pca.fit_transform(X_scaled)
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_pca)
        sizes = dict(sorted(Counter(labels).items()))
        cum_var = np.cumsum(pca.explained_variance_ratio_)[-1]

        print(f"  K={k}: best_pca={best_pca}, var={cum_var:.0%}, sil={best_sil:.3f}, sizes={sizes}")
        results[k] = (best_pca, best_sil)

    best_k = max(results, key=lambda k: results[k][1])
    best_pca, best_sil = results[best_k]
    print(f"\n  Best overall: K={best_k}, n_pca={results[best_k][0]}, sil={results[best_k][1]:.3f}")
    return results, best_k, best_pca, best_sil


def run_clustering(feat_df, X, n_clusters=None, n_pca=None):
    """PCA 降维 + KMeans 聚类。n_pca=None 时自动选该 K 的最优 n_pca。"""
    X_scaled = StandardScaler().fit_transform(X)

    # 评估所有 K × n_pca 组合
    results, best_k, best_pca, best_sil = evaluate_k(X_scaled)

    print(results, best_k, best_pca, best_sil)

    pca = PCA(n_components=min(best_pca, X.shape[0] - 1))
    X_pca = pca.fit_transform(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA: {best_pca} components, cumulative variance: {cum_var[-1]:.1%}")

    # 用指定 K 聚类
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    feat_df = feat_df.copy()
    feat_df['cluster'] = labels

    sil = silhouette_score(X_pca, labels)
    print(f"\n最终: K={best_k}, n_pca={best_pca}, silhouette={sil:.3f}")

    return feat_df, labels


def analyze_clusters(c: pd.DataFrame, u: pd.DataFrame, feat_df: pd.DataFrame, output_dir: str):
    """对每个 cluster 做详细分析并输出文件"""
    c = c.copy()
    c['datetime'] = pd.to_datetime(c['datetime'])
    c['weekday'] = c['datetime'].dt.dayofweek < 5

    os.makedirs(output_dir, exist_ok=True)
    n_clusters = feat_df['cluster'].nunique()

    # ── 1. cluster_assignments.csv ──
    assignments = feat_df[['uid', 'cluster']].copy()
    # 合并用户信息
    if u is not None:
        user_info = u[['user.index', 'age', 'gender', 'selfeff.intake', 'consc']].rename(
            columns={'user.index': 'uid'})
        assignments = assignments.merge(user_info, on='uid', how='left')
    assignments.to_csv(os.path.join(output_dir, 'cluster_assignments.csv'), index=False)
    print(f"\n保存: cluster_assignments.csv")

    # ── 2. 详细分析 + cluster_location_distributions.json ──
    cluster_dists = {}
    md_lines = ["# Location Clustering Analysis\n"]

    for cl in range(n_clusters):
        cl_uids = feat_df[feat_df['cluster'] == cl]['uid'].values.tolist()
        cl_data = c[c['uid'].isin(cl_uids)]
        avail = cl_data[cl_data['avail'] == True].copy()
        avail['weekday'] = pd.to_datetime(avail['datetime']).dt.dayofweek < 5

        print(f"\n{'='*60}")
        print(f"Cluster {cl} ({len(cl_uids)} users): UIDs = {cl_uids}")
        print(f"{'='*60}")

        md_lines.append(f"\n## Cluster {cl} ({len(cl_uids)} users)\n")
        md_lines.append(f"UIDs: {cl_uids}\n")

        # 地点分布
        cluster_dists[cl] = {}
        for wd_label, wd_val, wd_name in [('weekday', True, 'Weekday'), ('weekend', False, 'Weekend')]:
            print(f"\n  {wd_name} location by slot:")
            md_lines.append(f"\n### {wd_name}\n")

            for slot in range(1, 6):
                slot_data = cl_data[(cl_data['weekday'] == wd_val) & (cl_data['day_slot'] == slot)]
                if len(slot_data) > 0:
                    dist = slot_data['location'].value_counts(normalize=True)
                    top3 = dist.head(3)
                    top_str = ", ".join(f"{loc}={pct:.0%}" for loc, pct in top3.items())
                    print(f"    Slot {slot}: {top_str}")
                    md_lines.append(f"- Slot {slot}: {top_str}\n")

                    # 存储完整分布
                    key = f"{wd_label}_slot{slot}"
                    cluster_dists[cl][key] = dist.to_dict()
                else:
                    print(f"    Slot {slot}: no data")

        # 步数统计
        print(f"\n  步数统计:")
        if len(avail) > 0:
            sent = avail[avail['send'] > 0]
            nosent = avail[avail['send'] == 0]

            mean_steps = avail['jbsteps30'].mean()
            median_steps = avail['jbsteps30'].median()
            zero_rate = (avail['jbsteps30'] == 0).mean()
            te = sent['jbsteps30'].mean() - nosent['jbsteps30'].mean() if len(sent) > 0 and len(nosent) > 0 else 0

            print(f"    mean={mean_steps:.0f}, median={median_steps:.0f}, "
                  f"zero_rate={zero_rate:.1%}, TE={te:+.0f}")

            md_lines.append(f"\n### Steps\n")
            md_lines.append(f"- Mean: {mean_steps:.0f}, Median: {median_steps:.0f}\n")
            md_lines.append(f"- Zero rate: {zero_rate:.1%}\n")
            md_lines.append(f"- Treatment effect: {te:+.0f}\n")

            # 按 slot 步数
            print(f"    按 slot:")
            for slot in range(1, 6):
                sd = avail[avail['day_slot'] == slot]
                if len(sd) > 0:
                    print(f"      Slot {slot}: mean={sd['jbsteps30'].mean():.0f}, "
                          f"zero={( sd['jbsteps30']==0).mean():.0%}")

            # weekday vs weekend
            wd_steps = avail[avail['weekday'] == True]['jbsteps30'].mean()
            we_steps = avail[avail['weekday'] == False]['jbsteps30'].mean()
            print(f"    Weekday: {wd_steps:.0f}, Weekend: {we_steps:.0f}")
            md_lines.append(f"- Weekday mean: {wd_steps:.0f}, Weekend mean: {we_steps:.0f}\n")

            # response 分析
            sent_data = cl_data[cl_data['send'] > 0]
            if len(sent_data) > 0:
                good_rate = (sent_data['response'] == 'good').mean()
                bad_rate = (sent_data['response'] == 'bad').mean()
                print(f"    Response: good={good_rate:.0%}, bad={bad_rate:.0%}")
                md_lines.append(f"- Response rate: good={good_rate:.0%}, bad={bad_rate:.0%}\n")

        # 每用户明细
        print(f"\n  每用户:")
        md_lines.append(f"\n### Per-user\n")
        for uid in cl_uids:
            ud = avail[avail['uid'] == uid] if len(avail) > 0 else pd.DataFrame()
            ui = u[u['user.index'] == uid].iloc[0] if u is not None and len(u[u['user.index'] == uid]) > 0 else None

            age_str = f"{int(ui['age'])}" if ui is not None else "?"
            gender_str = ui['gender'] if ui is not None else "?"
            mean_str = f"{ud['jbsteps30'].mean():.0f}" if len(ud) > 0 else "N/A"
            zero_str = f"{( ud['jbsteps30']==0).mean():.0%}" if len(ud) > 0 else "N/A"

            print(f"    User {uid:2d} (age={age_str}, {gender_str}): mean={mean_str}, zero_rate={zero_str}")
            md_lines.append(f"- User {uid} (age={age_str}, {gender_str}): mean={mean_str}, zero={zero_str}\n")

    # 保存 JSON
    json_path = os.path.join(output_dir, 'cluster_location_distributions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_dists, f, indent=2, ensure_ascii=False)
    print(f"\n保存: cluster_location_distributions.json")

    # 保存 Markdown
    md_path = os.path.join(output_dir, 'cluster_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"保存: cluster_summary.md")

    # ── 3. 打印总对比表 ──
    print(f"\n{'='*60}")
    print(f"总对比表")
    print(f"{'='*60}")
    print(f"{'Cluster':<10} {'N':>3} {'Mean':>6} {'Median':>7} {'Zero%':>6} {'TE':>6} {'WD':>6} {'WE':>6}")
    print("-" * 55)

    for cl in range(n_clusters):
        cl_uids = feat_df[feat_df['cluster'] == cl]['uid'].values
        cl_avail = c[(c['uid'].isin(cl_uids)) & (c['avail'] == True)].copy()
        cl_avail['weekday'] = pd.to_datetime(cl_avail['datetime']).dt.dayofweek < 5

        if len(cl_avail) == 0:
            continue

        sent = cl_avail[cl_avail['send'] > 0]
        nosent = cl_avail[cl_avail['send'] == 0]
        te = sent['jbsteps30'].mean() - nosent['jbsteps30'].mean() if len(sent) > 0 and len(nosent) > 0 else 0
        wd = cl_avail[cl_avail['weekday'] == True]['jbsteps30'].mean()
        we = cl_avail[cl_avail['weekday'] == False]['jbsteps30'].mean()

        print(f"  {cl:<8} {len(cl_uids):>3} {cl_avail['jbsteps30'].mean():>6.0f} "
              f"{cl_avail['jbsteps30'].median():>7.0f} "
              f"{( cl_avail['jbsteps30']==0).mean():>6.1%} {te:>+6.0f} {wd:>6.0f} {we:>6.0f}")

    # ── 4. 输出给 data_extractor.py 用的 cluster_uids 字典 ──
    print(f"\n{'='*60}")
    print("可直接粘贴到 data_extractor.py 的 cluster_uids:")
    print(f"{'='*60}")
    print("self.cluster_uids = {")
    for cl in range(n_clusters):
        cl_uids = sorted(feat_df[feat_df['cluster'] == cl]['uid'].values.tolist())
        print(f"    {cl}: {cl_uids},")
    print("}")

    return cluster_dists


def build_cluster_config(c: pd.DataFrame, feat_df: pd.DataFrame, output_path: str = 'cluster_config.py'):
    """
    直接生成 data_extractor.py 可以 import 的 cluster_config.py
    
    用法（在 data_extractor.py 中）:
        from cluster_config import CLUSTER_UIDS, CLUSTER_WEIGHTS, CLUSTER_NAMES, CLUSTER_LOCATION_DIST, CLUSTER_AVAIL_BY_SLOT
    """
    c = c.copy()
    c['datetime'] = pd.to_datetime(c['datetime'])
    c['is_weekday'] = c['datetime'].dt.dayofweek < 5
    
    n_clusters = feat_df['cluster'].nunique()
    
    # 1. cluster_uids
    cluster_uids = {}
    for cl in range(n_clusters):
        cluster_uids[cl] = sorted(feat_df[feat_df['cluster'] == cl]['uid'].values.tolist())
    
    # 2. cluster_weights（按人数）
    cluster_weights = {cl: len(uids) for cl, uids in cluster_uids.items()}
    
    # 3. cluster_names（自动生成描述性名称）
    cluster_names = {}
    for cl, cl_uids in cluster_uids.items():
        cl_data = c[c['uid'].isin(cl_uids)]
        # 用工作日 Slot 2 的 top location 命名
        wd_s2 = cl_data[(cl_data['is_weekday'] == True) & (cl_data['day_slot'] == 2)]
        if len(wd_s2) > 0:
            top_loc = wd_s2['location'].value_counts().index[0]
            cluster_names[cl] = f"cluster_{cl}_{top_loc.split()[0].lower()}"
        else:
            cluster_names[cl] = f"cluster_{cl}"
    
    # 4. cluster_location_dist: {cl: {(is_weekday, slot): {loc: prob}}}
    cluster_location_dist = {}
    for cl, cl_uids in cluster_uids.items():
        cl_data = c[c['uid'].isin(cl_uids)]
        cluster_location_dist[cl] = {}
        for is_wd in [True, False]:
            for slot in [1, 2, 3, 4, 5]:
                subset = cl_data[(cl_data['is_weekday'] == is_wd) & (cl_data['day_slot'] == slot)]
                if len(subset) > 0:
                    dist = subset['location'].value_counts(normalize=True).to_dict()
                else:
                    fallback = c[c['day_slot'] == slot]['location'].value_counts(normalize=True).to_dict()
                    dist = fallback
                cluster_location_dist[cl][(is_wd, slot)] = dist
    
    # 5. cluster_avail_by_slot
    cluster_avail_by_slot = {}
    for cl, cl_uids in cluster_uids.items():
        cl_data = c[c['uid'].isin(cl_uids)]
        cluster_avail_by_slot[cl] = {}
        for slot in [1, 2, 3, 4, 5]:
            subset = cl_data[cl_data['day_slot'] == slot]
            cluster_avail_by_slot[cl][slot] = subset['avail'].mean() if len(subset) > 0 else 0.87
    
    # 写成 .py 文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('cluster_config.py — 自动生成，勿手动修改\n')
        f.write('由 cluster_analysis.py 的 build_cluster_config() 生成\n')
        f.write('"""\n\n')
        
        # cluster_uids: {int: list}
        f.write('CLUSTER_UIDS = {\n')
        for cl in sorted(cluster_uids.keys()):
            f.write(f'    {cl}: {cluster_uids[cl]},\n')
        f.write('}\n\n')
        
        # cluster_weights: {int: int}
        f.write(f'CLUSTER_WEIGHTS = {repr(cluster_weights)}\n\n')
        
        # cluster_names: {int: str}
        f.write(f'CLUSTER_NAMES = {repr(cluster_names)}\n\n')
        
        # location_dist: {int: {(bool, int): {str: float}}}
        f.write('# key = (is_weekday: bool, slot: int)\n')
        f.write('CLUSTER_LOCATION_DIST = {\n')
        for cl in sorted(cluster_location_dist.keys()):
            f.write(f'    {cl}: {{\n')
            for (is_wd, slot) in sorted(cluster_location_dist[cl].keys()):
                dist = {k: float(round(v, 6)) for k, v in cluster_location_dist[cl][(is_wd, slot)].items()}
                f.write(f'        ({is_wd}, {slot}): {dist},\n')
            f.write(f'    }},\n')
        f.write('}\n\n')
        
        # avail_by_slot: {int: {int: float}}
        f.write('CLUSTER_AVAIL_BY_SLOT = {\n')
        for cl in sorted(cluster_avail_by_slot.keys()):
            slot_dict = {int(k): float(round(v, 4)) for k, v in cluster_avail_by_slot[cl].items()}
            f.write(f'    {cl}: {slot_dict},\n')
        f.write('}\n')
    
    print(f"\n生成: {output_path}")
    print(f"  CLUSTER_UIDS: {cluster_uids}")
    print(f"  CLUSTER_WEIGHTS: {cluster_weights}")
    print(f"  CLUSTER_NAMES: {cluster_names}")
    print(f"  CLUSTER_LOCATION_DIST: {n_clusters} clusters × 10 conditions × locations")
    print(f"  CLUSTER_AVAIL_BY_SLOT: {n_clusters} clusters × 5 slots")
    
    return cluster_uids, cluster_weights, cluster_names, cluster_location_dist, cluster_avail_by_slot


def main():
    c = pd.read_csv('./data/cleaned_output.csv')
    u = pd.read_csv('./data/users.csv')
    print(f"数据: {len(c)} decisions, {c['uid'].nunique()} users, {c['location'].nunique()} locations")

    # Step 1: 构建特征矩阵
    feat_df, feat_cols, X, all_locs = build_feature_matrix(c)

    # Step 2: PCA + 聚类（n_pca=None 自动选该 K 的最优值）
    feat_df, labels = run_clustering(feat_df, X, n_clusters=3, n_pca=None)


    # Step 3: 分析
    cluster_dists = analyze_clusters(c, u, feat_df,'./data/clusters')

    # Step 4: 生成 data_extractor 可直接 import 的 config
    build_cluster_config(c, feat_df, 'data/cluster_config.py')



if __name__ == "__main__":
    main()
