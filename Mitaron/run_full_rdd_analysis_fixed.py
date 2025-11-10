import warnings
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
from datetime import datetime
from rdrobust import rdrobust, rdplot

warnings.filterwarnings("ignore")

def run_full_rdd_analysis(final_df, age_base=19, date_base=202201):
    """
    Run full RDD + Placebo (age±1) analysis and save an HTML report.
    """

    # === 1️⃣ Ask for HTML title ===
    html_title = input("Enter HTML report title (e.g. RDD_Age19_Analysis): ").strip()
    if not html_title:
        html_title = f"RDD_Analysis_Age{age_base}"

    # === 2️⃣ Data prep ===
    np.random.seed(4)
    df_rdd = final_df.clone() if isinstance(final_df, pl.DataFrame) else pl.from_pandas(final_df)

    df_rdd = df_rdd.with_columns([
        (pl.col(f"age_at_{date_base}") - age_base).alias("running_var"),
        (pl.col(f"age_at_{date_base}") < age_base).cast(pl.Int8).alias("treated")
    ])

    # Local window ±3 years
    df_local = df_rdd.filter(
        (pl.col(f"age_at_{date_base}") >= age_base - 3.0) &
        (pl.col(f"age_at_{date_base}") <= age_base + 3.0)
    )
    pdf = df_local.select(["running_var", "treated", "ika_out_req_amt"]).to_pandas()
    pdf = pdf.dropna().reset_index(drop=True)  # 🔧 FIX: Remove NaN values
    Y = "ika_out_req_amt"

    # === 3️⃣ Winsorize + log transform ===
    q_low, q_high = pdf[Y].quantile([0.01, 0.99])
    pdf["Y_winsor"] = pdf[Y].clip(q_low, q_high)
    pdf["Y_log_win"] = np.log1p(pdf["Y_winsor"])

    # === 4️⃣ Save distribution comparison ===
    fig_dist, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(pdf[Y], bins=100, color="gray", alpha=0.7)
    axes[0].set_title("Original Y (ika_out_req_amt)")
    axes[0].set_xlabel("¥ (yen)")
    axes[1].hist(pdf["Y_winsor"], bins=100, color="orange", alpha=0.7)
    axes[1].set_title("Winsorized (top 1%)")
    axes[1].set_xlabel("¥ (yen)")
    axes[2].hist(pdf["Y_log_win"], bins=100, color="steelblue", alpha=0.7)
    axes[2].set_title("log(1 + Winsorized Y)")
    axes[2].set_xlabel("log(1 + ¥)")
    plt.tight_layout()
    buf_dist = BytesIO()
    fig_dist.savefig(buf_dist, format="png", dpi=100, bbox_inches="tight")
    buf_dist.seek(0)
    dist_base64 = base64.b64encode(buf_dist.read()).decode("utf-8")
    plt.close(fig_dist)

    # === 5️⃣ Continuity plot ===
    pdf["bin"] = (pdf["running_var"] / 0.1).round() * 0.1
    bin_counts = pdf.groupby("bin").size().reset_index(name="count")
    fig_cont, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(bin_counts["bin"], bin_counts["count"], s=40, color="tomato", marker="D", alpha=0.7)
    sns.regplot(data=bin_counts, x="bin", y="count", scatter=False, order=2, color="black", ci=None)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set(title=f"Continuity in Running Variable around Age {age_base} Cutoff",
           xlabel=f"Age − {age_base} (Running Variable)", ylabel="Number of Observations")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    buf_cont = BytesIO()
    fig_cont.savefig(buf_cont, format="png", dpi=100, bbox_inches="tight")
    buf_cont.seek(0)
    cont_base64 = base64.b64encode(buf_cont.read()).decode("utf-8")
    plt.close(fig_cont)

    # === 6️⃣ RDD Mean Plot ===
    pdf["bin"] = (pdf["running_var"] * 10).round() / 10
    binned = pdf.groupby(["bin", "treated"])["Y_log_win"].mean().reset_index()
    fig_rdd, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=binned, x="bin", y="Y_log_win", hue="treated",
                    palette=["tab:green", "royalblue"], s=60, alpha=0.9, ax=ax)
    sns.regplot(data=pdf[pdf["running_var"] < 0], x="running_var", y="Y_log_win",
                scatter=False, color="tab:green", order=1, ax=ax)
    sns.regplot(data=pdf[pdf["running_var"] >= 0], x="running_var", y="Y_log_win",
                scatter=False, color="royalblue", order=1, ax=ax)
    ax.axvline(0, color="tomato", linestyle="--", linewidth=1.5)
    ax.set(title=f"RDD Mean Plot: log(Medical Expenditure) around Age {age_base}",
           xlabel=f"Age − {age_base} (Running Variable)", ylabel="log(1 + Medical Expenditure)")
    ax.legend(title="Treated (Eligible)", labels=["Control (≥19)", "Treated (<19)"])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    buf_rdd = BytesIO()
    fig_rdd.savefig(buf_rdd, format="png", dpi=100, bbox_inches="tight")
    buf_rdd.seek(0)
    rdd_base64 = base64.b64encode(buf_rdd.read()).decode("utf-8")
    plt.close(fig_rdd)

    # === 7️⃣ Nonparametric RDD and Placebos ===
    figs_base64 = []
    cutoffs = [0, -1, +1]

    print("\n" + "="*70)
    print("Running RDD Analysis at Multiple Cutoffs")
    print("="*70)

    for c in cutoffs:
        cutoff_label = f"Age {age_base + c}"
        print(f"Processing cutoff: {cutoff_label}")

        # 🔧 FIX: rdplot returns a single object, not a tuple
        result = rdplot(
            y=pdf['Y_log_win'].values,
            x=pdf['running_var'].values,
            c=c,
            title=f"RDD Plot: Medical Expenditure at {cutoff_label} Cutoff",
            x_label=f"Age − {age_base} (Running Variable)",
            y_label="log(1 + Medical Expenditure)",
            binselect="es"
        )
        buf = BytesIO()
        result.rdplot.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        figs_base64.append((c, base64.b64encode(buf.read()).decode("utf-8")))
        plt.close(result.rdplot)

    # === 8️⃣ Run rdrobust for statistics ===
    print("\n" + "="*70)
    print("RDD Estimates")
    print("="*70)

    rdd_stats = []
    for c in cutoffs:
        result = rdrobust(
            y=pdf['Y_log_win'].values,
            x=pdf['running_var'].values,
            c=c
        )
        coef = result.Estimate[0]
        se = result.se[0]
        pval = result.pv[0]
        ci_lower = result.ci[0, 0]
        ci_upper = result.ci[0, 1]

        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        cutoff_label = f"Age {age_base + c}"

        rdd_stats.append({
            'Cutoff': cutoff_label,
            'Coefficient': coef,
            'Std_Error': se,
            'P_value': pval,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Significance': sig
        })

        print(f"{cutoff_label:15s}: Coef={coef:+.4f}{sig:3s}, SE={se:.4f}, p={pval:.4f}")

    rdd_stats_df = pd.DataFrame(rdd_stats)

    # === 9️⃣ Build HTML ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = "Mitaron/RDD_Results"
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/{html_title}_{timestamp}.html"

    # Create statistics table HTML
    stats_html = rdd_stats_df.to_html(index=False, float_format=lambda x: f'{x:.4f}')

    html = f"""
    <!DOCTYPE html>
    <html><head><meta charset='utf-8'><title>{html_title}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
      h1 {{ text-align:center; border-bottom:3px solid #2c7be5; color: #2c3e50; }}
      h2 {{ border-left:5px solid #2c7be5; padding-left:10px; color: #34495e; margin-top: 30px; }}
      h3 {{ color: #7f8c8d; margin-top: 20px; }}
      img {{ display:block; margin:auto; border: 1px solid #ddd; padding: 10px; background: white; }}
      table {{ margin: 20px auto; border-collapse: collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
      th, td {{ padding: 10px 15px; text-align: left; border: 1px solid #ddd; }}
      th {{ background-color: #2c7be5; color: white; }}
      tr:nth-child(even) {{ background-color: #f2f2f2; }}
      .info-box {{ background: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 20px 0; }}
      .timestamp {{ text-align: center; color: #7f8c8d; font-size: 0.9em; }}
    </style></head><body>
    <h1>📊 {html_title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="info-box">
    <strong>Analysis Summary:</strong><br>
    • Sample Size: {len(pdf):,}<br>
    • Cutoff Age: {age_base}<br>
    • Running Variable Range: [{pdf['running_var'].min():.2f}, {pdf['running_var'].max():.2f}]<br>
    • Outcome: log(1 + Medical Expenditure)
    </div>

    <h2>1. Data Transformation</h2>
    <p>To handle outliers and skewness: (1) Winsorization at 1st and 99th percentiles, (2) Log transformation.</p>
    <img src="data:image/png;base64,{dist_base64}" style="width:95%;max-width:900px;">

    <h2>2. Continuity Check (McCrary Test)</h2>
    <p>Testing for manipulation around the cutoff. A smooth distribution suggests no manipulation.</p>
    <img src="data:image/png;base64,{cont_base64}" style="width:90%;max-width:800px;">

    <h2>3. RDD Mean Plot (Local Linear)</h2>
    <p>Visual evidence of discontinuity at the cutoff using binned means and local linear regression.</p>
    <img src="data:image/png;base64,{rdd_base64}" style="width:90%;max-width:800px;">

    <h2>4. RDD Estimates</h2>
    <p>Nonparametric estimates using rdrobust with optimal bandwidth selection.</p>
    {stats_html}

    <h2>5. Nonparametric RDD Plots (rdrobust.rdplot)</h2>

    <h3>Main Analysis: Age {age_base} Cutoff</h3>
    <p><strong>This is the main treatment effect.</strong> A visible discontinuity indicates a causal effect.</p>
    <img src="data:image/png;base64,{figs_base64[0][1]}" style="width:90%;max-width:800px;">

    <h3>Placebo Test: Age {age_base - 1} Cutoff</h3>
    <p><strong>Placebo test below the true cutoff.</strong> Should show NO discontinuity if RDD is valid.</p>
    <img src="data:image/png;base64,{figs_base64[1][1]}" style="width:90%;max-width:800px;">

    <h3>Placebo Test: Age {age_base + 1} Cutoff</h3>
    <p><strong>Placebo test above the true cutoff.</strong> Should show NO discontinuity if RDD is valid.</p>
    <img src="data:image/png;base64,{figs_base64[2][1]}" style="width:90%;max-width:800px;">

    <h2>6. Interpretation</h2>
    <div class="info-box">
    <strong>Key Findings:</strong><br>
    • Treatment Effect at Age {age_base}: <strong>{rdd_stats[0]['Coefficient']:.4f}</strong> (p={rdd_stats[0]['P_value']:.4f})<br>
    • Placebo at Age {age_base-1}: {rdd_stats[1]['Coefficient']:.4f} (p={rdd_stats[1]['P_value']:.4f})<br>
    • Placebo at Age {age_base+1}: {rdd_stats[2]['Coefficient']:.4f} (p={rdd_stats[2]['P_value']:.4f})<br>
    <br>
    <strong>✓ Valid RDD if:</strong> Main effect is significant, placebos are not.
    </div>

    <p style="text-align:center;margin-top:40px;color:#7f8c8d;">✅ Report saved at: <code>{file_path}</code></p>
    </body></html>
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("="*70)
    print(f"✅ RDD Report generated: {file_path}")
    print("="*70)

    return file_path
