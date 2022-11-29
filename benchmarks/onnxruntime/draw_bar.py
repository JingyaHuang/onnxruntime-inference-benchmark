import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

models = ["gpt2", "t5_s", "m2m100_418m"]
greedy_paths = [f"t4_res_ort_{model}_greedy.pkl" for model in models]
beam_paths = [f"t4_res_ort_{model}_beam5.pkl" for model in models]
greedy_dfs = [pd.read_pickle(path) for path in greedy_paths]
beam_dfs = [pd.read_pickle(path) for path in beam_paths]
greedy_stat = pd.DataFrame({
    "Model":["GPT2", "T5-small", "M2M100(418M)"],
    "ORT": [
        round(list(df[(df['seq_len']==8) & (df["framework"]=="ONNX")]["time_p95_ms_per_token"])[0], 2)
        for df in greedy_dfs
    ],
    "PyTorch": [
        round(list(df[(df['seq_len']==8) & (df["framework"]=="PyTorch")]["time_p95_ms_per_token"])[0], 2)
        for df in greedy_dfs
    ],
})
greedy_improv = []
for per in greedy_stat.apply(lambda row: round(-100. * (row.PyTorch - row.ORT) / row.PyTorch, 2), axis=1):
    greedy_improv.append(str(per) + "%")
greedy_improv = greedy_improv + [""]*3
greedy = greedy_stat.melt(id_vars='Model').rename(columns=str.title)
greedy["Framework"] = greedy["Variable"]
beam_stat =  pd.DataFrame({
    "Model":["GPT2", "T5-small", "M2M100(418M)"],
    "ORT": [
        round(list(df[(df['seq_len']==512) & (df["framework"]=="ONNX")]["time_p95_ms_per_token"])[0], 2)
        for df in beam_dfs
    ],
    "PyTorch": [
        round(list(df[(df['seq_len']==512) & (df["framework"]=="PyTorch")]["time_p95_ms_per_token"])[0], 2)
        for df in beam_dfs
    ],
})
beam_improv = []
for per in beam_stat.apply(lambda row: round(-100. * (row.PyTorch - row.ORT) / row.PyTorch, 2), axis=1):
    beam_improv.append(str(per) + "%")
beam_improv = beam_improv + [""]*3
beam = beam_stat.melt(id_vars='Model').rename(columns=str.title)
beam["Framework"] = beam["Variable"]

# Draw
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.barplot(x='Value', y='Model', hue="Framework", data=greedy, ax=ax1)
sns.barplot(x='Value', y='Model', hue="Framework", data=beam, ax=ax2)
for container, i in  zip(ax1.containers, range(2)):
    ax1.bar_label(container, labels=greedy_improv[3*i:3*i+3])
for container, i in  zip(ax2.containers, range(2)):
    ax2.bar_label(container, labels=beam_improv[3*i:3*i+3])
sns.despine(fig)

ax1.set_ylabel('')
ax1.set_xlabel('Greedy search - Per token latency(p95/ms)')
ax2.set_ylabel('')
ax2.set_xlabel('Beam search(5) - Per token latency(p95/ms)')

fig.suptitle("Inference benchmark: PyTorch V.S. ONNX Runtime")
fig.tight_layout()

plt.annotate(
    "seq_len:8, Tesla T4",
    (-1.6, -0.13),
    (0, -20),
    xycoords="axes fraction",
    textcoords="offset points",
    va="top",
)

plt.savefig("inference_models_8.png", bbox_inches="tight")