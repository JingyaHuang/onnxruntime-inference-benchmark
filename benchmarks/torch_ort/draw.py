import matplotlib.pyplot as plt
import numpy as np

N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.2  # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

onnx_vals = [40.23] * 3
rects1 = ax.bar(ind, onnx_vals, width)  # , color='r')
pt_vals = [23.66] * 3
rects2 = ax.bar(ind + width, pt_vals, width)  # , color='g')
optimized_vals = [19.93, 12.24, 16.78]
rects3 = ax.bar(ind + width * 2, optimized_vals, width)  # , color='b')

ax.set_ylabel("Latency")
ax.set_xticks(ind + width)
ax.set_xticklabels(
    ("graph-optimized", "Optimized-quantized(d)", "ORT-infer-module(ov)")
)
ax.legend(
    (rects1[0], rects2[0], rects3[0]),
    ("Vanilla ONNX", "Vanilla PyTorch", "Accelerated"),
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
ax.set_ylim([0, 45])


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.05 * h,
            "%d" % int(h),
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.title(
    "Inference benchamrk: ORT optimization V.S. ORTInferenceModule(OpenVINO) - P95"
)
plt.annotate(
    "seq_len:128, optimization_level:99, quantization:dynamic",
    (0, 0),
    (0, -20),
    xycoords="axes fraction",
    textcoords="offset points",
    va="top",
)
# plt.show()

plt.savefig("infer-bench.png", bbox_inches="tight")
