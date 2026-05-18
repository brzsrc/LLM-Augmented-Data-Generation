import numpy as np
import pandas as pd
from collections import Counter
import warnings

# 【修改4+14】从 cluster_analysis 生成的 cluster_config.py 导入聚类结果
from data.cluster_config import (CLUSTER_UIDS, CLUSTER_WEIGHTS, CLUSTER_NAMES,
                                 CLUSTER_LOCATION_DIST, CLUSTER_AVAIL_BY_SLOT)


warnings.filterwarnings('ignore')

# ================================================================
# 第一部分：从 cleaned_output.csv + users.csv 提取参数
# ================================================================

class V1DataExtractor:
    """
    【修改1】输入从 suggestions.csv 改为 cleaned_output.csv
    【修改2】所有列名适配 cleaned 格式
    """

    def __init__(self, cleaned_path: str,
                 train_uids_path: str = 'data/train_uids.json',
                 test_uids_path: str = 'data/test_uids.json'):
        import json

        sugg_full  = pd.read_csv(cleaned_path)

        # ── Train/test split：所有种子统计只在 train 上算 ──
        with open(train_uids_path) as f:
            train_uids = set(json.load(f))
        with open(test_uids_path) as f:
            test_uids = set(json.load(f))
        self.train_uids = train_uids
        self.test_uids  = test_uids


        # train split：用来 fit 所有种子统计
        self.sugg  = sugg_full[sugg_full['uid'].isin(train_uids)].reset_index(drop=True)

        # test split：留给后面 fidelity / FQE 评估时用，
        # generator 的训练阶段绝不能碰
        self.sugg_test  = sugg_full[sugg_full['uid'].isin(test_uids)].reset_index(drop=True)

        print(f"[split] train: {len(train_uids)} users / {len(self.sugg)} rows  |  "
              f"test: {len(test_uids)} users / {len(self.sugg_test)} rows")

        self._compute_user_profiles()
        self._compute_context_params()
        self._compute_step_params()
        self._compute_user_context_stats()

    def _compute_user_profiles(self):
        """generate_baseline_vectors 消费：Cholesky基线 + 步数/TE经验分布 + 性别 + 聚类"""
        s = self.sugg

        # 每用户的真实平均步数和 TE（直接采样用，不做回归）
        user_stats = s.groupby('uid').agg(mean_steps=('jbsteps30', 'mean')).reset_index()
        self.real_mean_steps = user_stats['mean_steps'].values

        user_stats = s.groupby('uid').agg(mean_steps=('jbsteps30pre', 'mean')).reset_index()
        self.real_mean_steps_pre = user_stats['mean_steps'].values

        avail = s[s['avail'] == True]
        te_list = []
        for uid in avail['uid'].unique():
            ud = avail[avail['uid'] == uid]
            sent_mean = ud[ud['send'] > 0]['jbsteps30'].mean()
            nosent_mean = ud[ud['send'] == 0]['jbsteps30'].mean()
            te = sent_mean - nosent_mean if not np.isnan(sent_mean - nosent_mean) else 0
            te_list.append(te)
        self.real_te = np.array(te_list)


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
        """generate_context / generate_base_steps 消费:
        slot 统计 + 全局统计 + location 步数 + 每 (slot, loc_group) 桶的零膨胀对数正态参数

        【新增】per-bin (slot, loc_group) 的 ZI-lognormal 参数:
            zero_prob, log_mu, log_sd
        prior 直接从该桶采样, baseline 在 prior 基础上做修正。
        """
        s = self.sugg

        # 每 slot 统计
        self.slot_stats = {
            slot: {
                'mean': s[s['day_slot'] == slot]['jbsteps30'].mean(),
                'zero_rate': (s[s['day_slot'] == slot]['jbsteps30'] == 0).mean(),
                'mean_pre': s[s['day_slot'] == slot]['jbsteps30pre'].mean(),
                'zero_rate_pre': (s[s['day_slot'] == slot]['jbsteps30pre'] == 0).mean(),
            }
            for slot in [1, 2, 3, 4, 5]
        }

        # 全局统计
        self.global_mean_steps = s['jbsteps30'].mean()
        self.global_zero_rate = (s['jbsteps30'] == 0).mean()
        self.global_log_steps = float(np.log(s.loc[s['jbsteps30'] > 0, 'jbsteps30']).mean())

        self.global_mean_steps_pre = s['jbsteps30pre'].mean()
        self.global_zero_rate_pre = (s['jbsteps30pre'] == 0).mean()
        self.global_log_steps_pre = float(np.log(s.loc[s['jbsteps30pre'] > 0, 'jbsteps30pre']).mean())

        # 每 location 的平均步数
        self.steps_by_location = s.groupby('location')['jbsteps30'].mean().to_dict()

        # ─── 【新增】per-bin ZI-lognormal 参数 ──────────────────────────
        # 把 location 折成 top-8 + 'Other' 防止稀疏桶
        top_locs = s['location'].value_counts().head(8).index.tolist()
        self.top_locations = set(top_locs)
        s_loc_g = s['location'].where(s['location'].isin(top_locs), 'Other')

        # 全局 fallback
        nz_all = s.loc[s['jbsteps30'] > 0, 'jbsteps30']
        self.global_log_mu = float(np.log(nz_all).mean())
        self.global_log_sd = float(min(np.log(nz_all).std(), 1.2))

        # 桶内拟合
        self.step_bin_params = {}
        df = s.copy()
        df['_loc_g'] = s_loc_g
        for (slot, loc_g), grp in df.groupby(['day_slot', '_loc_g']):
            n = len(grp)
            zp = float((grp['jbsteps30'] == 0).mean())
            nz = grp.loc[grp['jbsteps30'] > 0, 'jbsteps30']
            if len(nz) >= 5:
                log_mu = float(np.log(nz).mean())
                log_sd = float(min(np.log(nz).std(), 1.2))  # 截断防长尾爆炸
            else:
                log_mu, log_sd = self.global_log_mu, self.global_log_sd
            self.step_bin_params[(int(slot), str(loc_g))] = (zp, log_mu, log_sd, n)

        slot_means = [f"{self.slot_stats[i]['mean']:.0f}" for i in range(1, 6)]
        print(f"[step_params] global_mean={self.global_mean_steps:.1f}, "
              f"zero_rate={self.global_zero_rate:.1%}, "
              f"slot_means={slot_means}, "
              f"step_bins={len(self.step_bin_params)} fitted")

    def get_step_bin(self, slot: int, location: str):
        """取 (slot, location) 桶的 (zero_prob, log_mu, log_sd) 参数。
        location 不在 top-8 时映射到 'Other'; 桶找不到时回退到全局参数。"""
        loc_g = location if location in self.top_locations else 'Other'
        params = self.step_bin_params.get((int(slot), loc_g))
        if params is None:
            return self.global_zero_rate, self.global_log_mu, self.global_log_sd
        return params[0], params[1], params[2]

    # =========================================================================
    # 【新增】per-user 个体级统计 + rich persona generator
    # 给 Plan A 的三个 prompt 增强方法服务:
    #   方法 1: population/user reference points  → get_user_bin_mean, step_bin_params
    #   方法 2: slot-matched history              → get_user_slot_history
    #   方法 3: rich persona                       → get_rich_persona
    # =========================================================================
    def _compute_user_context_stats(self):
        """对 train 上的每个用户, 算:
          - user_slot_loc_mean[uid][(slot, loc_g)]: 该用户在 (slot, loc_g) 桶的真实步数均值
          - user_slot_mean[uid][slot]: 该用户在 slot 的均值
          - user_slot_zero[uid][slot]: 该用户在 slot 的零率
          - user_slot_history[uid][slot]: chronological list of (date, jbsteps30) for that slot
          - user_persona_text[uid]: rich persona (从前 5 个 study_day derive)

        注: 用 train 上该用户的全部行 derive (跟 generator 在 production 的口径一致)
            persona 单独用前 5 个 study_day 的数据 derive, 模拟 "study 跑了 5 天后我们知道这个人是怎样的"
        """
        s = self.sugg
        # location 折叠到 top-8 + Other (跟 step_bin_params 一致)
        loc_g_series = s['location'].where(s['location'].isin(self.top_locations), 'Other')
        s_loc_g = s.assign(_loc_g=loc_g_series)

        self.user_slot_loc_mean = {}    # uid -> {(slot, loc_g): mean_steps}
        self.user_slot_mean = {}        # uid -> {slot: mean_steps}
        self.user_slot_zero = {}        # uid -> {slot: zero_rate}
        self.user_slot_history = {}     # uid -> {slot: [(date, steps), ...]}
        self.user_overall_mean = {}     # uid -> overall jbsteps30 mean
        self.user_te = {}               # uid -> send vs no-send mean diff

        s_sorted = s_loc_g.sort_values(['uid', 'date', 'day_slot'])
        for uid, ud in s_sorted.groupby('uid'):
            # per (slot, loc_g)
            bin_means = {}
            for (slot, loc_g), grp in ud.groupby(['day_slot', '_loc_g']):
                bin_means[(int(slot), str(loc_g))] = float(grp['jbsteps30'].mean())
            self.user_slot_loc_mean[uid] = bin_means

            # per slot
            slot_means, slot_zeros, slot_hist = {}, {}, {}
            for slot, grp in ud.groupby('day_slot'):
                slot_means[int(slot)] = float(grp['jbsteps30'].mean())
                slot_zeros[int(slot)] = float((grp['jbsteps30'] == 0).mean())
                slot_hist[int(slot)] = [
                    (str(r['date']), int(r['jbsteps30']))
                    for _, r in grp[['date', 'jbsteps30']].iterrows()
                ]
            self.user_slot_mean[uid] = slot_means
            self.user_slot_zero[uid] = slot_zeros
            self.user_slot_history[uid] = slot_hist

            # overall
            self.user_overall_mean[uid] = float(ud['jbsteps30'].mean())
            avail = ud[ud['avail'] == True] if 'avail' in ud.columns else ud
            if len(avail) > 0:
                sent = avail[avail['send'] > 0]['jbsteps30']
                nosent = avail[avail['send'] == 0]['jbsteps30']
                if len(sent) > 0 and len(nosent) > 0:
                    self.user_te[uid] = float(sent.mean() - nosent.mean())
                else:
                    self.user_te[uid] = 0.0
            else:
                self.user_te[uid] = 0.0

        # ── 构造 rich persona (方法 3) ──
        self.user_persona_text = {}
        for uid, ud in s_sorted.groupby('uid'):
            uniq_dates = sorted(ud['date'].unique())
            self.user_persona_text[uid] = self._build_rich_persona(uid, ud)

        print(f"[user_context_stats] {len(self.user_slot_loc_mean)} users, "
              f"avg bins/user={np.mean([len(v) for v in self.user_slot_loc_mean.values()]):.1f}")

    def _build_rich_persona(self, uid, all_df):
        """
        从all_df数据 derive 一段自然语言 persona, 用于 LLM prompt.
        """
        ref = all_df

        # 基本面: 真实步数均值 (over all rows, including unavail)
        overall_mean = float(ref['jbsteps30pre'].mean()) if len(ref) > 0 else 0.0

        # 各 slot 的均值, 找最活跃 / 最不活跃
        slot_means = {}
        for s in [1, 2, 3, 4, 5]:
            sub = ref[ref['day_slot'] == s]
            if len(sub) > 0:
                slot_means[s] = float(sub['jbsteps30pre'].mean())
        if slot_means:
            most_active_slot = max(slot_means, key=slot_means.get)
            least_active_slot = min(slot_means, key=slot_means.get)
            slot_desc_map = {1: "early morning", 2: "morning", 3: "midday",
                             4: "afternoon", 5: "evening"}
            most_desc = slot_desc_map.get(most_active_slot, f"slot {most_active_slot}")
            least_desc = slot_desc_map.get(least_active_slot, f"slot {least_active_slot}")
            most_steps = int(slot_means[most_active_slot])
            least_steps = int(slot_means[least_active_slot])
        else:
            most_desc, least_desc, most_steps, least_steps = "midday", "early morning", 0, 0

        # 整体零率
        zero_rate = float((ref['jbsteps30pre'] == 0).mean()) if len(ref) > 0 else 0.0

        # location 分布
        loc_counts = ref['location'].value_counts(normalize=True)
        top3_locs = loc_counts.head(5)
        loc_strs = [f"{l} ({p*100:.0f}%)" for l, p in top3_locs.items()]
        loc_desc = ", ".join(loc_strs) if loc_strs else "various"

        text = (
            f"Behavioral profile (user general walking behavior without intervention):\n"
            f"- Walks ~{int(overall_mean)} steps per 30-min slot on average; "
            f"{int(zero_rate*100)}% of slots have zero steps.\n"
            f"- Most active in the {most_desc} (slot {most_active_slot if slot_means else '?'}, "
            f"~{most_steps} steps); least active in the {least_desc} "
            f"(slot {least_active_slot if slot_means else '?'}, ~{least_steps} steps).\n"
            f"- Top5 most frequent locations visited: {loc_desc}.\n"
        )
        return text

    def get_rich_persona(self, uid):
        """返回该 user 的 rich persona 文本 (从前 5 个 study_day derive).
        若用户没 derive 过 (e.g. test user), 返回简短的通用 fallback."""
        if uid in self.user_persona_text:
            return self.user_persona_text[uid]
        return (
            f"Behavioral profile (no warmup data available):\n"
            f"- Walks ~{int(self.global_mean_steps)} steps per 30-min slot on average."
        )


# ================================================================
# 第五部分：上下文和步数生成
# ================================================================

# def generate_context(ext: V1DataExtractor, params: dict, day: int, slot: int, traj: list) -> dict:
#     """
#     【修改14】location 从该用户所属 cluster 的 weekday/weekend × slot 分布中采样
#     【重构】prior_30min_steps 改为按 (slot, location) 桶的零膨胀对数正态采样,
#             与 generate_base_steps 中 baseline 的形状统一。
#     """
#     is_weekday = (day - 1) % 7 < 5
#     cluster_id = int(params.get('cluster', 0))
#
#     # 【修改14】从 cluster 的地点分布采样
#     loc_dist = ext.cluster_location_dist.get(cluster_id, {}).get((is_weekday, slot), None)
#     if loc_dist is None:
#         # fallback 到全局分布
#         loc_dist = ext.location_by_slot.get(slot, {'Home': 1.0})
#
#     locs = list(loc_dist.keys())
#     probs = list(loc_dist.values())
#     probs = [p / sum(probs) for p in probs]  # 归一化
#     location = np.random.choice(locs, p=probs)
#
#     # avail 率也用 cluster 的
#     cl_avail = ext.cluster_avail_by_slot.get(cluster_id, {})
#     avail_prob = cl_avail.get(slot, ext.avail_by_slot.get(slot, 0.87))
#     avail = int(np.random.random() < avail_prob)
#
#     # 天气
#     weather_options = ['Clear', 'Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Rain']
#     weather = np.random.choice(weather_options, p=[0.3, 0.3, 0.2, 0.1, 0.1])
#     temperature = round(np.random.normal(22, 8), 1)
#
#     # ── 前 30 分钟步数: 零膨胀对数正态, 桶参数来自真实数据 ───────────
#     zp_p, log_mu_p, log_sd_p = ext.get_step_bin(slot, location)
#     user_offset = np.log(max(params['predicted_mean_steps_pre'], 1.0)
#                          / max(ext.global_mean_steps_pre, 1.0))
#     weekend_offset = 0.0 if is_weekday else np.log(0.9)
#     if np.random.random() < zp_p:
#         prior = 0
#     else:
#         prior = max(0, min(6000,
#                     int(np.exp(np.random.normal(
#                         log_mu_p + user_offset + weekend_offset, log_sd_p)))))
#
#     yesterday = (sum(t['jbsteps30'] for t in traj[-5:]) if len(traj) >= 5
#                  else int(params['predicted_mean_steps'] * 5))
#
#     # activity
#     if prior > 200:
#         activity = 'ON_FOOT'
#     elif np.random.random() < 0.08:
#         activity = 'IN_VEHICLE'
#     else:
#         activity = 'STILL'
#
#     # 【修改9】send: 三值概率
#     # 先决定是否 randomized（~59%），然后按三值概率分配
#     if avail and np.random.random() < ext.randomization_rate:
#         # is_randomized=True: 按 send_probs 采样
#         send_vals = list(ext.send_probs.keys())
#         send_ps = list(ext.send_probs.values())
#         action = int(np.random.choice(send_vals, p=send_ps))
#     else:
#         action = 0  # not randomized → no send
#
#     if not avail:
#         action = 0
#
#     # 【修改11】生成 response
#     response = 'no_send'
#     if action > 0:
#         resp_vals = list(ext.response_probs.keys())
#         resp_ps = list(ext.response_probs.values())
#         response = np.random.choice(resp_vals, p=resp_ps)
#
#     return dict(study_day=day, slot=slot, location=location, avail=avail,
#                 temperature=temperature, weather=weather,
#                 prior_30min_steps=prior, yesterday_steps=yesterday,
#                 activity=activity, weekday=(day - 1) % 7 < 5, action=action,
#                 response=response)


def generate_base_steps(params: dict, ctx: dict, action: int, dosage: float, ext: V1DataExtractor) -> int:
    """
    【重构】baseline 直接基于 ctx['prior_30min_steps'] 计算, 不再独立采样。

    思路:
      prior 已经从真实桶采过样, 同时携带了"前 30 分钟人在做什么"的信息。
      baseline 要预测紧接着的 30 分钟, 应当与 prior 强相关 (真实数据 corr ≈ 0.4)。
      于是把 prior 映射到 log 空间作为锚, 上面叠加:
          (a) slot 切换偏移   (从 prior_slot → 当前 slot 的桶 log_mu 之差)
          (b) location 切换偏移 (同上)
          (c) action / dosage 治疗效应 (log-加性)
          (d) IN_VEHICLE 强制零率
      再叠加一个 log_sd 的随机扰动表示 30 分钟内的自然波动。

    保留: 治疗效应、weekend 微调、IN_VEHICLE 零率上调。
    去掉: slot_ratios 硬编码、(0.5+0.5×loc_ratio) 平滑、独立 lognormal(0, 0.5)。
    """
    # IN_VEHICLE 时强制零率
    if ctx.get('activity') == 'IN_VEHICLE' and np.random.random() < 0.7:
        return 0

    # 当前桶参数
    zp_b, log_mu_b, log_sd_b = ext.get_step_bin(ctx['slot'], ctx['location'])

    # 当前 slot 自身的零膨胀: 但 prior=0 的人继续 0 的概率应该更高
    # P(curr=0 | prior=0) ≈ 0.57 真实数据, 这里取 max(zp_b, 0.55) 当 prior=0 时
    prior = ctx.get('prior_30min_steps', 0)
    if prior == 0:
        zero_prob_eff = max(zp_b, 0.55)  # prior=0 时 zero 倾向上调
    else:
        zero_prob_eff = zp_b * 0.6        # prior>0 时 zero 倾向下调
    if np.random.random() < zero_prob_eff:
        return 0

    # ── log-空间 anchoring ────────────────────────────────────────
    # prior 提供"当前活动水平"的锚: 把 prior 转 log, 与桶内 log_mu 做混合,
    # 混合系数 ρ 近似真实 corr(log prior, log curr) ≈ 0.4。
    # 具体:  log_curr = log_mu_b + ρ * (log_prior - log_mu_p) + offsets + noise
    rho = 0.4
    if prior > 0:
        log_prior = np.log(prior)
        # 用 prior 当时所处桶的 log_mu 做差 (这一桶在 generate_context 里已经被采过)
        # 这里用当前桶 log_mu_b 当代理也行——目的只是把 prior 的偏离量加权传进来
        zp_p, log_mu_p, _ = ext.get_step_bin(ctx['slot'], ctx['location'])
        prior_dev = log_prior - log_mu_p
    else:
        prior_dev = 0.0  # prior=0 但通过零率筛选, 视为没有偏离信息

    # 用户尺度偏移 (log)
    user_offset = np.log(max(params['predicted_mean_steps'], 1.0)
                         / max(ext.global_mean_steps, 1.0))

    # 周末微调
    weekend_offset = 0.0 if ctx['weekday'] else np.log(0.9)

    # 治疗效应 (log 加性, dosage 衰减)
    te_offset = 0.0
    if action > 0:
        te_pct = params.get('predicted_te', 0.0) / max(ext.global_mean_steps, 1.0)
        te_decayed = te_pct * max(0.0, 1.0 - 0.05 * dosage)
        te_offset = np.log1p(max(te_decayed, -0.99))
        if action == 2:
            te_offset *= 0.8  # sedentary 略弱

    mu = log_mu_b + user_offset + weekend_offset + te_offset + rho * prior_dev

    # 30 分钟内的随机扰动: 用桶的 log_sd, 再缩一点 (一部分方差已被 prior_dev 锁定)
    sd = log_sd_b * np.sqrt(1 - rho ** 2)

    return max(0, min(6000, int(np.exp(np.random.normal(mu, sd)))))


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






if __name__ == "__main__":
    main()
