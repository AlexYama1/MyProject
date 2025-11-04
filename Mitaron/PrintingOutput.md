print(f"\nデータ形状: {df_DID.shape}")
print("=" * 50)
print(f"処置群: {treated_patients}人 ({treated_rows:,}行)")
print(f"対照群: {control_patients}人 ({control_rows:,}行)")
print("=" * 50)




print("\n" + "="*80)
print("STEP 2: Propensity Score推定（患者レベル、政策前期間）")
print("="*80)

# 政策前の患者レベルデータを作成
df_pre_patient = df_DID[df_DID["period"] == 0].groupby("patient_id").agg({
    "D": "first",
    **{covar: "mean" for covar in matching_covariates if covar in df_DID.columns}
}).reset_index()

print(f"政策前の患者数: {len(df_pre_patient):,}")

# 欠損値処理
df_ps = df_pre_patient[["patient_id", "D"] + matching_covariates].dropna()
print(f"欠損値除外後: {len(df_ps):,}人")

# Propensity Score推定
X = df_ps[matching_covariates].values
y_treat = df_ps["D"].values

ps_model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
ps_model.fit(X, y_treat)


df_ps["propensity_score"] = ps_model.predict_proba(X)[:, 1]

#標準偏差
ps_std = np.std(df_ps["propensity_score"])
print(f"Propensity Score の標準偏差: {ps_std:.6f}")

print(f"\nPropensity Score統計:")
print(f"  処置群 平均: {df_ps[df_ps['D']==1]['propensity_score'].mean():.4f}")
print(f"  対照群 平均: {df_ps[df_ps['D']==0]['propensity_score'].mean():.4f}")

# PS分布の可視化
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_ps[df_ps["D"]==0]["propensity_score"], bins=50, alpha=0.6, label="対照群", color="blue")
ax.hist(df_ps[df_ps["D"]==1]["propensity_score"], bins=50, alpha=0.6, label="処置群", color="orange")
ax.set_xlabel("Propensity Score", fontsize=12)
ax.set_ylabel("度数", fontsize=12)
ax.set_title("Propensity Scoreの分布", fontsize=14, fontweight='bold')
ax.legend()
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# バランス改善の可視化
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = range(len(balance_comparison))
width = 0.35

ax.barh([i - width/2 for i in y_pos], balance_comparison["SMD_before"],
        width, label="マッチング前", alpha=0.7, color="red")
ax.barh([i + width/2 for i in y_pos], balance_comparison["SMD_after"],
        width, label="マッチング後", alpha=0.7, color="green")

ax.set_yticks(y_pos)
ax.set_yticklabels(balance_comparison["variable"])
ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2)
ax.axvline(x=-0.1, color='red', linestyle='--', linewidth=2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("標準化平均差 (SMD)", fontsize=12)
ax.set_title("PSマッチング前後の共変量バランス", fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
pzt.show()



### Comments for the Code

1. Do the per visit, per visit price of the med
2. 二次関数for the RDD
3. Poisson (TWFE) - if the dummy variables?  for the RDD 



warnings.filterwarnings('ignore')

# SET GLOBAL SEED FOR REPRODUCIBILITY
np.random.seed(4)

# ============================================================================
# 0. データ準備
# ============================================================================

print("="*80)
print("共通トレンド仮定が満たされない場合の包括的分析")
print("="*80)

df_DID = final_df.to_pandas()

df_DID["time_numeric"] = (
    df_DID["medtreat_yymm"].astype(str).str[:4].astype(int)
    + (df_DID["medtreat_yymm"].astype(str).str[4:6].astype(int) - 1) / 12
)

policy_time = 2021 + 3/12
df_DID["period"] = (df_DID["time_numeric"] >= policy_time).astype(int)
df_DID["did"] = df_DID["D"] * df_DID["period"]
df_DID["y"] = df_DID["ika_out_req_amt"]
df_DID["month"] = df_DID["medtreat_yymm"]

treated_patients = df_DID.loc[df_DID["D"] == 1, "patient_id"].nunique()
treated_rows = df_DID["D"].sum()
control_patients = df_DID.loc[df_DID["D"] == 0, "patient_id"].nunique()
control_rows = ((1 - df_DID["D"])).sum()

# ============================================================================
#  1 共通トレンド（Parallel Trends）の確認プロット
# ============================================================================
# Compute average outcome by month and treatment status
trend_df = (
    df_DID.groupby(["month", "D"])
    .agg(mean_y=("y", "mean"))
    .reset_index()
    .sort_values("month")
)

# Convert YYYYMM to numeric time axis
trend_df["year"] = trend_df["month"].astype(str).str[:4].astype(int)
trend_df["mon"] = trend_df["month"].astype(str).str[4:6].astype(int)
trend_df["time"] = trend_df["year"] + (trend_df["mon"] - 1) / 12

# Define policy change timing
policy_time = 2021 + 3/12

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=trend_df,
    x="time",
    y="mean_y",
    hue="D",
    palette={0: "gray", 1: "steelblue"},
    linewidth=2.2
)

# Add vertical line for policy introduction
fig1, ax1 = plt.subplots()
plt.axvline(x=policy_time, color="red", linestyle="--", linewidth=1.5)
plt.text(policy_time + 0.02, trend_df["mean_y"].max()*0.95,
         "Policy starts (2021-04)", color="red", fontsize=11)

# Labels and formatting
plt.title("Treatment vs. Control Trends Over Time", fontsize=14, weight="bold")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Outcome (y)", fontsize=12)
plt.legend(title="Group", labels=["Control (D=0)", "Treatment (D=1)"])
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Saving the plot
buf1 = BytesIO()
fig1.savefig(buf1, format = "png", dpi = 100, bbox_inches = 'tight')
buf1.seek(0)
ps_plot_base64 = base64.b64encode(buf1.read()).decode('utf-8')
plt.close(fig1)

# ============================================================================
# 1. 共変量の選択
# ============================================================================

print("\n" + "="*80)
print("STEP 1: 共変量の選択")
print("="*80)


potential_covariates = ["sex_type_nm", "business_type_num", "annual_salary_rank_num"]

print(f"共変量候補数: {len(potential_covariates)}")

correlation_results = []
for covar in potential_covariates:
    try:
        valid_data = df_DID[[covar, "y"]].dropna()
        if len(valid_data) > 0:
            corr = valid_data[covar].corr(valid_data["y"])
            correlation_results.append({
                "variable": covar,
                "correlation": corr,
                "abs_correlation": abs(corr)
            })
    except:
        continue

corr_df = pd.DataFrame(correlation_results).sort_values("abs_correlation", ascending=False)
matching_covariates = corr_df[abs(corr_df["correlation"]) != 0]["variable"].tolist()[:15]

print(f"\nマッチング用共変量数: {len(matching_covariates)}")
print(f"マッチング用共変量: {matching_covariates}")

# ============================================================================
# 2. Propensity Score推定
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Propensity Score推定（患者レベル、政策前期間）")
print("="*80)

df_pre_patient = df_DID[df_DID["period"] == 0].groupby("patient_id").agg({
    "D": "first",
    **{covar: "mean" for covar in matching_covariates if covar in df_DID.columns}
}).reset_index()

print(f"政策前の患者数: {len(df_pre_patient):,}")

df_ps = df_pre_patient[["patient_id", "D"] + matching_covariates].dropna()
print(f"欠損値除外後: {len(df_ps):,}人")

X = df_ps[matching_covariates].values
y_treat = df_ps["D"].values

ps_model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
ps_model.fit(X, y_treat)

df_ps["propensity_score"] = ps_model.predict_proba(X)[:, 1]

ps_std = np.std(df_ps["propensity_score"])
print(f"Propensity Score の標準偏差: {ps_std:.6f}")
print(f"  処置群 平均: {df_ps[df_ps['D']==1]['propensity_score'].mean():.4f}")
print(f"  対照群 平均: {df_ps[df_ps['D']==0]['propensity_score'].mean():.4f}")

# ============================================================================
# 3. PSM with STABLE matching
# ============================================================================

print("\n" + "="*80)
print(f"STEP 3: Propensity Score Matching（1:1, Caliper={ps_std*2:.6f}）")
print("="*80)

treated = df_ps[df_ps["D"] == 1].copy().reset_index(drop=True)  # Reset index for stability
control = df_ps[df_ps["D"] == 0].copy().reset_index(drop=True)

caliper = ps_std * 2

matched_pairs = []
used_control_ids = set()

for idx, treated_row in treated.iterrows():
    treated_ps = treated_row["propensity_score"]
    
    candidates = control[
        (~control["patient_id"].isin(used_control_ids)) &
        (abs(control["propensity_score"] - treated_ps) <= caliper)
    ].copy()
    
    if len(candidates) > 0:
        # STABLE TIE-BREAKING: add random jitter
        candidates["distance"] = abs(candidates["propensity_score"] - treated_ps)
        candidates["jitter"] = np.random.uniform(0, 1e-10, len(candidates))  # Tiny random noise
        candidates["final_distance"] = candidates["distance"] + candidates["jitter"]
        
        best_match = candidates.loc[candidates["final_distance"].idxmin()]
        
        matched_pairs.append({
            "treated_id": treated_row["patient_id"],
            "control_id": best_match["patient_id"],
            "treated_ps": treated_ps,
            "control_ps": best_match["propensity_score"],
            "ps_diff": abs(treated_ps - best_match["propensity_score"])
        })
        
        used_control_ids.add(best_match["patient_id"])

matched_df = pd.DataFrame(matched_pairs)
print(f"\nマッチング成功:")
print(f"  マッチされたペア数: {len(matched_df):,} / {len(treated):,} ({len(matched_df)/len(treated)*100:.1f}%)")
print(f"  PS差の平均: {matched_df['ps_diff'].mean():.4f}")
print(f"  PS差の最大: {matched_df['ps_diff'].max():.4f}")

matched_patient_ids = set(matched_df["treated_id"]) | set(matched_df["control_id"])
df_matched = df_DID[df_DID["patient_id"].isin(matched_patient_ids)].copy()

print(f"\nマッチング後のデータ:")
print(f"  観測数: {len(df_matched):,}")
print(f"  処置群: {df_matched['D'].sum():,}行")
print(f"  対照群: {(1-df_matched['D']).sum():,}行")

# ============================================================================
# 4. バランスチェック
# ============================================================================

print("\n" + "="*80)
print("STEP 4: マッチング後の共変量バランス")
print("="*80)

df_matched_pre = df_matched[df_matched["period"] == 0].copy()

balance_before = []
balance_after = []

for covar in matching_covariates[:15]:
    if covar not in df_matched_pre.columns:
        continue
    
    df_pre_all = df_DID[df_DID["period"] == 0]
    control_all = df_pre_all[df_pre_all["D"] == 0][covar].dropna()
    treated_all = df_pre_all[df_pre_all["D"] == 1][covar].dropna()
    
    if len(control_all) > 0 and len(treated_all) > 0:
        pooled_std_before = np.sqrt((control_all.std()**2 + treated_all.std()**2) / 2)
        smd_before = (treated_all.mean() - control_all.mean()) / pooled_std_before if pooled_std_before > 0 else 0
    else:
        smd_before = np.nan
    
    control_matched = df_matched_pre[df_matched_pre["D"] == 0][covar].dropna()
    treated_matched = df_matched_pre[df_matched_pre["D"] == 1][covar].dropna()
    
    if len(control_matched) > 0 and len(treated_matched) > 0:
        pooled_std_after = np.sqrt((control_matched.std()**2 + treated_matched.std()**2) / 2)
        smd_after = (treated_matched.mean() - control_matched.mean()) / pooled_std_after if pooled_std_after > 0 else 0
    else:
        smd_after = np.nan
    
    balance_before.append({"variable": covar, "smd": smd_before})
    balance_after.append({"variable": covar, "smd": smd_after})

balance_comparison = pd.DataFrame({
    "variable": [b["variable"] for b in balance_before],
    "SMD_before": [b["smd"] for b in balance_before],
    "SMD_after": [b["smd"] for b in balance_after]
})

print("\n共変量バランス比較:")
print(balance_comparison.to_string(index=False))

# ============================================================================
# 5. IPW重みの計算 (WITH CHECKS)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: IPW重みの計算")
print("="*80)

# Check for missing PS before calculating IPW
df_ipw = df_DID.merge(
    df_ps[["patient_id", "propensity_score"]],
    on="patient_id",
    how="left"
)

missing_ps = df_ipw["propensity_score"].isna().sum()
if missing_ps > 0:
    print(f"⚠️ WARNING: {missing_ps} observations missing propensity scores - using mean imputation")
    df_ipw["propensity_score"].fillna(df_ipw["propensity_score"].mean(), inplace=True)

# Clip extreme PS to avoid division by near-zero
df_ipw["propensity_score"] = df_ipw["propensity_score"].clip(0.01, 0.99)

df_ipw["ipw_weight"] = np.where(
    df_ipw["D"] == 1,
    1 / df_ipw["propensity_score"],
    1 / (1 - df_ipw["propensity_score"])
)

lower = df_ipw["ipw_weight"].quantile(0.01)
upper = df_ipw["ipw_weight"].quantile(0.99)
df_ipw["ipw_weight_trimmed"] = df_ipw["ipw_weight"].clip(lower, upper)

print(f"IPW重み統計（トリム後）:")
print(f"  平均: {df_ipw['ipw_weight_trimmed'].mean():.4f}")
print(f"  中央値: {df_ipw['ipw_weight_trimmed'].median():.4f}")
print(f"  標準偏差: {df_ipw['ipw_weight_trimmed'].std():.4f}")

# PSM + IPW
df_psm_ipw = df_matched.merge(
    df_ps[["patient_id", "propensity_score"]],
    on="patient_id",
    how="left"
)

df_psm_ipw["propensity_score"] = df_psm_ipw["propensity_score"].clip(0.01, 0.99)

df_psm_ipw["ipw_weight"] = np.where(
    df_psm_ipw["D"] == 1,
    1 / df_psm_ipw["propensity_score"],
    1 / (1 - df_psm_ipw["propensity_score"])
)

lower = df_psm_ipw["ipw_weight"].quantile(0.01)
upper = df_psm_ipw["ipw_weight"].quantile(0.99)
df_psm_ipw["ipw_weight_trimmed"] = df_psm_ipw["ipw_weight"].clip(lower, upper)

print(f"\nPSM + IPW重み統計（トリム後）:")
print(f"  平均: {df_psm_ipw['ipw_weight_trimmed'].mean():.4f}")
print(f"  中央値: {df_psm_ipw['ipw_weight_trimmed'].median():.4f}")

# ============================================================================
# 6. DID推定
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: Simple DID")
print("="*80)
m1 = smf.ols("y ~ D + period + did", data=df_DID).fit(
    cov_type="cluster", cov_kwds={"groups": df_DID["area_id"]}
)
print(f"DID係数: {m1.params['did']:.4f} (SE: {m1.bse['did']:.4f}, P: {m1.pvalues['did']:.4f})")

print("\n" + "="*80)
print("MODEL 2: DID + TWFE")
print("="*80)
df_panel = df_DID.set_index(["patient_id", "medtreat_yymm"])
m2 = PanelOLS.from_formula("y ~ 1 + did + EntityEffects + TimeEffects", data=df_panel)
result_m2 = m2.fit(cov_type="clustered", clusters=df_panel["area_id"])
print(result_m2.summary)

print("\n" + "="*80)
print("MODEL 3: DID + TWFE + PSM")
print("="*80)
df_panel = df_matched.set_index(["patient_id", "medtreat_yymm"])
m3 = PanelOLS.from_formula("y ~ 1 + did + EntityEffects + TimeEffects", data=df_panel)
result_m3 = m3.fit(cov_type="clustered", clusters=df_panel["area_id"])
print(result_m3.summary)

print("\n" + "="*80)
print("MODEL 4: DID + TWFE + IPW")
print("="*80)
df_panel = df_ipw.set_index(["patient_id", "medtreat_yymm"]).sort_index()
weights_ipw = df_panel["ipw_weight_trimmed"].astype(float)
m4 = PanelOLS.from_formula("y ~ 1 + did + EntityEffects + TimeEffects", data=df_panel, weights=weights_ipw)
result_m4 = m4.fit(cov_type="clustered", clusters=df_panel["area_id"])
print(result_m4.summary)

print("\n" + "="*80)
print("MODEL 5: DID + TWFE + PSM + IPW")
print("="*80)
df_panel = df_psm_ipw.set_index(["patient_id", "medtreat_yymm"]).sort_index()
weights_psm_ipw = df_panel["ipw_weight_trimmed"].astype(float)
m5 = PanelOLS.from_formula("y ~ 1 + did + EntityEffects + TimeEffects", data=df_panel, weights=weights_psm_ipw)
result_m5 = m5.fit(cov_type="clustered", clusters=df_panel["area_id"])
print(result_m5.summary)


# ============================================================================
# Generate HTML Report
# ============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
file_path = f"DID_comprehensive_analysis_{timestamp}.html"

# Build summary statistics HTML
summary_stats = f"""
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;">
    <h3>データサマリー</h3>
    <table style="width: auto; margin: 0;">
        <tr><td><b>データ形状:</b></td><td>{df_DID.shape[0]:,} × {df_DID.shape[1]}</td></tr>
        <tr><td><b>処置群:</b></td><td>{treated_patients:,}人 ({treated_rows:,}行)</td></tr>
        <tr><td><b>対照群:</b></td><td>{control_patients:,}人 ({control_rows:,}行)</td></tr>
        <tr><td><b>政策実施時期:</b></td><td>2021年4月</td></tr>
    </table>
</div>

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;">
    <h3>共変量選択</h3>
    <table style="width: auto; margin: 0;">
        <tr><td><b>共変量候補数:</b></td><td>{len(potential_covariates)}</td></tr>
        <tr><td><b>マッチング用共変量数:</b></td><td>{len(matching_covariates)}</td></tr>
        <tr><td><b>マッチング用共変量:</b></td><td>{', '.join(matching_covariates)}</td></tr>
    </table>
</div>

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;">
    <h3>Propensity Score推定結果（患者レベル、政策前期間）</h3>
    <table style="width: auto; margin: 0;">
        <tr><td><b>政策前の患者数:</b></td><td>{len(df_pre_patient):,}</td></tr>
        <tr><td><b>欠損値除外後:</b></td><td>{len(df_ps):,}人</td></tr>
        <tr><td><b>PS標準偏差:</b></td><td>{ps_std:.6f}</td></tr>
        <tr><td><b>処置群 PS平均:</b></td><td>{ps_treat_mean:.4f}</td></tr>
        <tr><td><b>対照群 PS平均:</b></td><td>{ps_control_mean:.4f}</td></tr>
    </table>
    <br>
    <img src="data:image/png;base64,{ps_plot_base64}" style="width:100%; max-width:800px;">
</div>

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;">
    <h3>Propensity Score Matching（1:1, Caliper={caliper:.6f}）</h3>
    <table style="width: auto; margin: 0;">
        <tr><td><b>マッチング前 処置群:</b></td><td>{treated_patients:,}人</td></tr>
        <tr><td><b>マッチング前 対照群:</b></td><td>{control_patients:,}人</td></tr>
        <tr><td><b>マッチされたペア数:</b></td><td>{len(matched_df):,}</td></tr>
        <tr><td><b>PS差の平均:</b></td><td>{matched_df['ps_diff'].mean():.4f}</td></tr>
        <tr><td><b>PS差の最大:</b></td><td>{matched_df['ps_diff'].max():.4f}</td></tr>
        <tr><td><b>マッチング後 観測数:</b></td><td>{len(df_matched):,}</td></tr>
        <tr><td><b>マッチング後 処置群:</b></td><td>{df_matched['D'].sum():,}</td></tr>
        <tr><td><b>マッチング後 対照群:</b></td><td>{(1-df_matched['D']).sum():,}</td></tr>
    </table>
</div>

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;">
    <h3>マッチング後の共変量バランス</h3>
    {balance_comparison.to_html(index=False, border=0)}
    <br>
    <img src="data:image/png;base64,{balance_plot_base64}" style="width:100%; max-width:900px;">
</div>

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;">
    <h3>IPW重みの計算（マッチング後）</h3>
    <h4>IPW重み統計（トリム後）</h4>
    <table style="width: auto; margin: 0;">
        <tr><td><b>平均:</b></td><td>{ipw_mean:.4f}</td></tr>
        <tr><td><b>標準偏差:</b></td><td>{ipw_std:.4f}</td></tr>
        <tr><td><b>最小値:</b></td><td>{ipw_min:.4f}</td></tr>
        <tr><td><b>最大値:</b></td><td>{ipw_max:.4f}</td></tr>
    </table>
    <br>
    <h4>PSM + IPW重み統計（トリム後）</h4>
    <table style="width: auto; margin: 0;">
        <tr><td><b>平均:</b></td><td>{psm_ipw_mean:.4f}</td></tr>
        <tr><td><b>標準偏差:</b></td><td>{psm_ipw_std:.4f}</td></tr>
        <tr><td><b>最小値:</b></td><td>{psm_ipw_min:.4f}</td></tr>
        <tr><td><b>最大値:</b></td><td>{psm_ipw_max:.4f}</td></tr>
    </table>
</div>
"""

# Combine model results
html_sections = [
    "<h2>MODEL 1: Simple DID (no FE, cluster by area)</h2>" + m1_simple.summary().as_html(),
    "<h2>MODEL 2: DID + TWFE</h2>" + result_m2.summary.as_html(),
    "<h2>MODEL 3: DID + TWFE + PSM</h2>" + result_m3.summary.as_html(),
    "<h2>MODEL 4: DID + TWFE + IPW</h2>" + result_m4.summary.as_html(),
    "<h2>MODEL 5: DID + TWFE + PSM + IPW</h2>" + result_m5.summary.as_html()
]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"""
    <html><head><meta charset='utf-8'>
    <title>DID 包括的分析レポート</title>
    <style>
      body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #fff; }}
      h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
      h2 {{ color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }}
      h3 {{ color: #34495e; }}
      h4 {{ color: #5a6c7d; }}
      hr {{ margin: 40px 0; border: 0; border-top: 2px solid #ecf0f1; }}
      table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
      th {{ background-color: #3498db; color: white; text-align: center; }}
      .timestamp {{ text-align: center; color: #7f8c8d; font-size: 0.9em; margin-top: 10px; }}
    </style></head><body>
    <h1>共通トレンド仮定が満たされない場合の包括的分析</h1>
    <div class="timestamp">作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</div>
    {summary_stats}
    <hr>
    {"<hr>".join(html_sections)}
    </body></html>
    """)

print(f"✅ Analysis complete! File ready: {file_path}")



# print("\n" + "="*80)
# print("SUMMARY OF RESULTS")
# print("="*80)
# print(f"M1 Simple DID: {m1.params['did']:.2f}")
# print(f"M2 + TWFE: {result_m2.params['did']:.2f}")
# print(f"M3 + PSM: {result_m3.params['did']:.2f}")
# print(f"M4 + IPW: {result_m4.params['did']:.2f}")
# print(f"M5 + PSM+IPW: {result_m5.params['did']:.2f}")