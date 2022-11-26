import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataframe
path = 't4_res_ort_gpt2_greedy.pkl'
df_greedy = pd.read_pickle(path)
path = 't4_res_ort_gpt2_beam5.pkl'
df_beam = pd.read_pickle(path)
df = pd.concat([df_greedy, df_beam], ignore_index=True)
df["seq_len_str"] = df['seq_len'].astype(str)
# df["Framework"] = df.apply(lambda row: "PyTorch" if row.framework=="pt" else "ORT", axis=1)
df["Framework"] = df.apply(lambda row: "PyTorch" if row.framework=="PyTorch" else "ORT", axis=1)
df["Generation"] = df.apply(lambda row: "Beam(5)" if row.num_beam>1 else "Greedy", axis=1)

# Draw

# hue = "(" + df['Framework'].astype(str) + ', ' + df['Generation'].astype(str) + ")"
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
ax_s = sns.scatterplot(
    x='seq_len_str', y='time_ms_per_token',
    data=df, hue="Framework", style="Generation"
)
ax_s.get_legend().remove()
ax = sns.lineplot(
    x='seq_len_str', y='time_p95_ms_per_token',
    data=df, hue="Framework", style="Generation", legend="full"
) #, orient="v", palette="deep", dodge=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title(
    "GPT2 inference: PyTorch V.S. ONNX Runtime"
)
plt.xlabel("Sequence Length")
# plt.ylim(0, 10)
plt.ylabel("Per Token Latency (ms)")
plt.tight_layout()
plt.savefig("t4_res_ort_gpt2.png", dpi=900)