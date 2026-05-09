import numpy as np
import pandas as pd
from collections import Counter
import warnings

# 【修改4+14】从 cluster_analysis 生成的 cluster_config.py 导入聚类结果
from data.cluster_config import (CLUSTER_UIDS, CLUSTER_WEIGHTS, CLUSTER_NAMES,
                                 CLUSTER_LOCATION_DIST, CLUSTER_AVAIL_BY_SLOT)


warnings.filterwarnings('ignore')
"""
synthetic_user_generator_v3_cleaned.py
======================================
基于 cleaned_output.csv 重写的合成用户生成器。

与 v3 原版相比的所有修改：
  【修改1】输入数据：suggestions.csv + users.csv → cleaned_output.csv + users.csv
  【修改2】列名映射：user.index→uid, sugg.select.slot→day_slot, 
           dec.temperature→temperature, recognized.activity→activity,
           dec.location.category→location, dec.weather.condition→weather
  【修改3】send 编码：True/False → 0/1/2 (0=no_send, 1=active, 2=sedentary)
  【修改4】location 分类：3类(home/work/other) → 16类(直接用cleaned的分类)
  【修改5】avail 率：硬编码(70%/83%) → 从数据按slot提取实际值
  【修改6】步数统计：global_mean=276 → 从cleaned计算=246.5
  【修改7】零值率：0.27 → 从cleaned计算=0.371
  【修改8】activity 名称：WALKING → ON_FOOT
  【修改9】发送概率：固定π_b=0.3 → 基于is_randomized的三值概率
  【修改10】TE 计算：send==True vs False → send>0 vs send==0
  【修改11】response 列：新增 good/bad/no_response/snoozed 信息用于观察记录
  【修改12】dosage 计算：适配 send=1 和 send=2 都算发送
  【修改13】create_observation：区分 active suggestion 和 sedentary suggestion
"""


# ================================================================
# 第一部分：从 cleaned_output.csv + users.csv 提取参数
# ================================================================

class V1DataExtractor:
    """
    【修改1】输入从 suggestions.csv 改为 cleaned_output.csv
    【修改2】所有列名适配 cleaned 格式
    """

    def __init__(self, users_path: str, cleaned_path: str,
                 train_uids_path: str = 'data/train_uids.json',
                 test_uids_path: str = 'data/test_uids.json'):
        import json

        users_full = pd.read_csv(users_path)
        sugg_full  = pd.read_csv(cleaned_path)

        # ── Train/test split：所有种子统计只在 train 上算 ──
        with open(train_uids_path) as f:
            train_uids = set(json.load(f))
        with open(test_uids_path) as f:
            test_uids = set(json.load(f))
        self.train_uids = train_uids
        self.test_uids  = test_uids

        # users.csv 里 uid 列名是 user.index（来自原始 mHealth 数据）
        uid_col = 'user.index' if 'user.index' in users_full.columns else 'uid'

        # train split：用来 fit 所有种子统计
        self.sugg  = sugg_full[sugg_full['uid'].isin(train_uids)].reset_index(drop=True)
        self.users = users_full[users_full[uid_col].isin(train_uids)].reset_index(drop=True)

        # test split：留给后面 fidelity / FQE 评估时用，
        # generator 的训练阶段绝不能碰
        self.sugg_test  = sugg_full[sugg_full['uid'].isin(test_uids)].reset_index(drop=True)
        self.users_test = users_full[users_full[uid_col].isin(test_uids)].reset_index(drop=True)

        print(f"[split] train: {len(train_uids)} users / {len(self.sugg)} rows  |  "
              f"test: {len(test_uids)} users / {len(self.sugg_test)} rows")

        self._compute_user_profiles()
        self._compute_context_params()
        self._compute_step_params()

    def _compute_user_profiles(self):
        """generate_baseline_vectors 消费：Cholesky基线 + 步数/TE经验分布 + 性别 + 聚类"""
        s, u = self.sugg, self.users

        # Cholesky 基线参数（生成 selfeff, consc, age 给 persona 文本用，不预测步数）
        self.baseline_cols = ['selfeff.intake', 'consc', 'age']
        data = u[self.baseline_cols].dropna()
        self.baseline_mu = data.mean().values
        self.baseline_sigma = data.cov().values
        eig = np.min(np.linalg.eigvalsh(self.baseline_sigma))
        if eig < 1e-6:
            self.baseline_sigma += (abs(eig) + 1e-5) * np.eye(len(self.baseline_cols))

        # 每用户的真实平均步数和 TE（直接采样用，不做回归）
        user_stats = s.groupby('uid').agg(mean_steps=('jbsteps30', 'mean')).reset_index()
        self.real_mean_steps = user_stats['mean_steps'].values

        avail = s[s['avail'] == True]
        te_list = []
        for uid in avail['uid'].unique():
            ud = avail[avail['uid'] == uid]
            sent_mean = ud[ud['send'] > 0]['jbsteps30'].mean()
            nosent_mean = ud[ud['send'] == 0]['jbsteps30'].mean()
            te = sent_mean - nosent_mean if not np.isnan(sent_mean - nosent_mean) else 0
            te_list.append(te)
        self.real_te = np.array(te_list)

        # 性别分布
        self.gender_probs = u['gender'].value_counts(normalize=True).to_dict()

        # 聚类结果
        self.cluster_uids = CLUSTER_UIDS
        self.cluster_weights = CLUSTER_WEIGHTS
        self.cluster_names = CLUSTER_NAMES
        self.cluster_location_dist = CLUSTER_LOCATION_DIST
        self.cluster_avail_by_slot = CLUSTER_AVAIL_BY_SLOT

        print(f"[user_profiles] {len(u)} users, "
              f"mean_steps range=[{self.real_mean_steps.min():.0f}, {self.real_mean_steps.max():.0f}], "
              f"TE range=[{self.real_te.min():.0f}, {self.real_te.max():+.0f}], "
              f"{len(self.cluster_uids)} clusters")

    def _compute_context_params(self):
        """generate_context 消费：avail率 + location分布 + 发送概率 + response分布"""
        s = self.sugg

        # avail 率（全局，按 slot）
        self.avail_by_slot = {}
        for slot in [1, 2, 3, 4, 5]:
            self.avail_by_slot[slot] = s[s['day_slot'] == slot]['avail'].mean()

        # 全局 location 分布（fallback）
        self.location_categories = sorted(s['location'].dropna().unique().tolist())
        self.location_by_slot = {}
        for slot in [1, 2, 3, 4, 5]:
            slot_data = s[s['day_slot'] == slot]
            self.location_by_slot[slot] = slot_data['location'].value_counts(normalize=True).to_dict()

        # 发送概率
        rand_data = s[s['is_randomized'] == True]
        self.send_probs = rand_data['send'].value_counts(normalize=True).to_dict()
        self.randomization_rate = s['is_randomized'].mean()

        # response 分布
        sent_data = s[s['send'] > 0]
        self.response_probs = sent_data['response'].value_counts(normalize=True).to_dict()

        print(f"[context_params] send_probs={self.send_probs}, "
              f"randomization_rate={self.randomization_rate:.1%}, "
              f"{len(self.location_categories)} location types")

    def _compute_step_params(self):
        """generate_base_steps 消费：slot统计 + 全局统计 + location步数"""
        s = self.sugg

        # 每 slot 统计
        self.slot_stats = {
            slot: {
                'mean': s[s['day_slot'] == slot]['jbsteps30'].mean(),
                'zero_rate': (s[s['day_slot'] == slot]['jbsteps30'] == 0).mean(),
            }
            for slot in [1, 2, 3, 4, 5]
        }

        # 全局统计
        self.global_mean_steps = s['jbsteps30'].mean()
        self.global_zero_rate = (s['jbsteps30'] == 0).mean()

        # 每 location 的平均步数
        self.steps_by_location = s.groupby('location')['jbsteps30'].mean().to_dict()

        slot_means = [f"{self.slot_stats[i]['mean']:.0f}" for i in range(1, 6)]
        print(f"[step_params] global_mean={self.global_mean_steps:.1f}, "
              f"zero_rate={self.global_zero_rate:.1%}, "
              f"slot_means={slot_means}")


def generate_baseline_vectors(ext: V1DataExtractor, n: int) -> pd.DataFrame:

    """Cholesky生成心理属性（给persona用）+ 从V1经验分布直接采样步数和TE"""
    L = np.linalg.cholesky(ext.baseline_sigma)

    z = np.random.standard_normal((n, len(ext.baseline_cols)))
    bl = ext.baseline_mu + z @ L.T

    alpha = np.random.uniform(-2, 2, len(ext.baseline_cols))
    delta = alpha / np.sqrt(1 + alpha @ alpha)

    for i in range(n):
        u, v = np.random.standard_normal(len(ext.baseline_cols)), np.random.standard_normal(len(ext.baseline_cols))
        bl[i] += (u > 0).astype(float) * (delta * np.abs(v))

    df = pd.DataFrame(bl, columns=ext.baseline_cols)

    df['selfeff.intake'] = df['selfeff.intake'].clip(5, 25)
    df['consc'] = df['consc'].clip(12, 30)
    df['age'] = df['age'].clip(19, 65).round().astype(int)

    gv, gp = list(ext.gender_probs.keys()), list(ext.gender_probs.values())
    df['gender'] = np.random.choice(gv, n, p=gp)
    df['user_id'] = range(1, n + 1)

    # 直接从 V1 的 37 人经验分布采样（不用回归）
    df['predicted_mean_steps'] = np.random.choice(ext.real_mean_steps, n, replace=True)
    df['predicted_te'] = np.random.choice(ext.real_te, n, replace=True)

    # 按聚类比例分配 cluster
    cluster_ids = list(ext.cluster_weights.keys())
    cluster_counts = list(ext.cluster_weights.values())
    cluster_probs = [c / sum(cluster_counts) for c in cluster_counts]
    df['cluster'] = np.random.choice(cluster_ids, n, p=cluster_probs)
    df['cluster_name'] = df['cluster'].map(ext.cluster_names)

    print(f"  Cluster分配: {dict(Counter(df['cluster'].values))}")
    print(f"  mean_steps: mean={df['predicted_mean_steps'].mean():.0f}, "
          f"range=[{df['predicted_mean_steps'].min():.0f}, {df['predicted_mean_steps'].max():.0f}]")
    print(f"  TE: mean={df['predicted_te'].mean():+.0f}, "
          f"range=[{df['predicted_te'].min():+.0f}, {df['predicted_te'].max():+.0f}]")

    return df


# ================================================================
# 第五部分：上下文和步数生成
# ================================================================

def generate_context(ext: V1DataExtractor, params: dict, day: int, slot: int, traj: list) -> dict:
    """
    【修改14】location 从该用户所属 cluster 的 weekday/weekend × slot 分布中采样
    """
    is_weekday = (day - 1) % 7 < 5
    cluster_id = int(params.get('cluster', 0))

    # 【修改14】从 cluster 的地点分布采样
    loc_dist = ext.cluster_location_dist.get(cluster_id, {}).get((is_weekday, slot), None)
    if loc_dist is None:
        # fallback 到全局分布
        loc_dist = ext.location_by_slot.get(slot, {'Home': 1.0})

    locs = list(loc_dist.keys())
    probs = list(loc_dist.values())
    probs = [p / sum(probs) for p in probs]  # 归一化
    location = np.random.choice(locs, p=probs)

    # avail 率也用 cluster 的
    cl_avail = ext.cluster_avail_by_slot.get(cluster_id, {})
    avail_prob = cl_avail.get(slot, ext.avail_by_slot.get(slot, 0.87))
    avail = int(np.random.random() < avail_prob)

    # 天气
    weather_options = ['Clear', 'Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Rain']
    weather = np.random.choice(weather_options, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    temperature = round(np.random.normal(22, 8), 1)

    # 前30分钟步数
    user_scale = params['predicted_mean_steps'] / max(ext.global_mean_steps, 1)
    ss = ext.slot_stats.get(slot, {'mean': 200})
    prior = max(0, int(np.random.exponential(ss['mean'] * user_scale * 0.8)))

    yesterday = sum(t['jbsteps30'] for t in traj[-5:]) if len(traj) >= 5 else int(params['predicted_mean_steps'] * 5)

    # activity
    if prior > 200:
        activity = 'ON_FOOT'
    elif np.random.random() < 0.08:
        activity = 'IN_VEHICLE'
    else:
        activity = 'STILL'

    # 【修改9】send: 三值概率
    # 先决定是否 randomized（~59%），然后按三值概率分配
    if avail and np.random.random() < ext.randomization_rate:
        # is_randomized=True: 按 send_probs 采样
        send_vals = list(ext.send_probs.keys())
        send_ps = list(ext.send_probs.values())
        action = int(np.random.choice(send_vals, p=send_ps))
    else:
        action = 0  # not randomized → no send

    if not avail:
        action = 0

    # 【修改11】生成 response
    response = 'no_send'
    if action > 0:
        resp_vals = list(ext.response_probs.keys())
        resp_ps = list(ext.response_probs.values())
        response = np.random.choice(resp_vals, p=resp_ps)

    return dict(study_day=day, slot=slot, location=location, avail=avail,
                temperature=temperature, weather=weather,
                prior_30min_steps=prior, yesterday_steps=yesterday,
                activity=activity, weekday=(day - 1) % 7 < 5, action=action,
                response=response)


def generate_base_steps(params: dict, ctx: dict, action: int, dosage: float, ext: V1DataExtractor) -> int:
    """
    【修改3】action=1 和 action=2 都有治疗效果，但可能不同
    【修改6】用 cleaned 的 global_mean=246.5
    【修改7】零值率用 cleaned 的 0.371
    """
    # slot 比例（从 cleaned 数据计算：141/247, 307/247, 260/247, 268/247, 256/247）
    slot_ratios = {1: 0.57, 2: 1.24, 3: 1.06, 4: 1.09, 5: 1.04}
    baseline = params['predicted_mean_steps'] * slot_ratios.get(ctx['slot'], 1.0)

    # 地点效应（从 cleaned 的 location 平均步数）
    loc_mean = ext.steps_by_location.get(ctx['location'], ext.global_mean_steps)
    loc_ratio = loc_mean / max(ext.global_mean_steps, 1)
    baseline *= (0.5 + 0.5 * loc_ratio)  # 地点效应混合

    if not ctx['weekday']:
        baseline *= 0.9

    # 【修改3】治疗效果：send=1 和 send=2 都有效
    week = ctx['study_day'] // 7
    decay = max(0, 1.0 - 0.22 * week)
    # 【修改12】send=1 和 send=2 都算发送
    if action > 0:
        te = params['predicted_te'] * decay * max(0, 1.0 - 0.05 * dosage)
        if action == 2:
            te *= 0.8  # sedentary suggestion 效果略弱
    else:
        te = 0

    raw = max(0, (baseline + te) * np.random.lognormal(0, 0.5))

    # 【修改7】零值率从 cleaned 数据
    zero_prob = ext.global_zero_rate + 0.01 * max(0, ctx['study_day'] - 20)
    if ctx['activity'] == 'IN_VEHICLE':
        zero_prob = 0.7

    return 0 if np.random.random() < zero_prob else int(raw)


# ================================================================
# 第七部分：主入口
# ================================================================

def main():
    users_csv = 'data/users.csv'
    cleaned_csv = 'data/cleaned_output.csv'


    # 【修改1】用 cleaned_output.csv
    ext = V1DataExtractor(users_csv, cleaned_csv)

    # 健全性检查：种子统计绝不能见过 test 用户
    assert ext.sugg['uid'].isin(ext.test_uids).sum() == 0, \
        "LEAKAGE: test uids found in training sugg!"
    assert len(ext.train_uids & ext.test_uids) == 0, \
        "LEAKAGE: train and test uids overlap!"
    # 同时检查 cluster_config 没有 test uids（防止忘了重跑 cluster_analysis.py）
    all_cluster_uids = set(uid for lst in CLUSTER_UIDS.values() for uid in lst)
    leaked = ext.test_uids & all_cluster_uids
    assert not leaked, \
        f"LEAKAGE: test uids {leaked} appear in CLUSTER_UIDS — " \
        f"rerun cluster_analysis.py after split_users.py!"
    print(f"[sanity] no leakage detected ✓")


    baseline_df = generate_baseline_vectors(ext, 10)
    print('baseline_df', baseline_df.to_string())





if __name__ == "__main__":
    main()
