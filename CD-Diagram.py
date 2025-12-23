import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# تابع محاسبه CD برای تست Bonferroni-Dunn
# =====================================================
def compute_CD_bd(n_methods, n_datasets, alpha="0.05"):
    """
    محاسبه Critical Difference مخصوص تست Bonferroni-Dunn
    """
    # مقادیر بحرانی q_alpha برای تست Bonferroni-Dunn (مقایسه با کنترل)
    qalpha_bd = {
        "0.05": {
            2: 1.960, 3: 2.241, 4: 2.394, 5: 2.498,
            6: 2.576, 7: 2.638, 8: 2.690, 9: 2.724,
            10: 2.773
        },
        "0.1": {
            2: 1.645, 3: 1.960, 4: 2.128, 5: 2.241,
            6: 2.326, 7: 2.394, 8: 2.450, 9: 2.498,
            10: 2.539
        }
    }

    k = n_methods
    if k > 10:
        # برای بیش از 10 متد، مقدار تخمینی یا نزدیکترین استفاده می‌شود
        q = 2.773
    else:
        q = qalpha_bd[alpha][k]

    # فرمول CD برای Bonferroni-Dunn
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    return cd

# =====================================================
# تابع اصلی رسم نمودار (نسخه اصلاح شده برای تصویر درخواستی)
# =====================================================
def graph_ranks(avranks, names, cd=None, width=10, textspace=1.5, reverse=True):
    lowv = 1
    highv = int(np.ceil(max(avranks)))
    if highv < len(names): highv = len(names)

    k = len(avranks)
    cline = 0.4
    distanceh = 0.25
    cline += distanceh

    # مرتب‌سازی داده‌ها
    ssums_sorted = sorted(zip(avranks, names), key=lambda x: x[0])
    ssums = [x[0] for x in ssums_sorted]
    nnames = [x[1] for x in ssums_sorted]

    height = cline + ((k + 1) / 2) * 0.2 + 0.5
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height
    wf = 1. / width

    # رسم خط مقیاس اصلی
    ax.plot([textspace*wf, (width-textspace)*wf], [1-cline*hf, 1-cline*hf], "k-", linewidth=1)

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + (width - 2*textspace) / (highv - lowv) * a

    # رسم تیک‌ها و اعداد
    for a in range(lowv, highv + 1):
        x = rankpos(a) * wf
        ax.plot([x, x], [1-cline*hf, 1-(cline-0.1)*hf], "k-", linewidth=1)
        ax.text(x, 1-(cline-0.2)*hf, str(a), ha="center", va="bottom", size=12)

    # رسم پاره‌خط CD در بالا
    if cd:
        cd_x_start = textspace * wf
        cd_x_end = rankpos(lowv + cd) * wf if not reverse else rankpos(highv - cd) * wf
        # رسم پاره‌خط کوچک برای نمایش طول CD
        ax.plot([cd_x_start, cd_x_start + (abs(rankpos(lowv+cd)-rankpos(lowv))*wf)],
                [1-(cline-0.4)*hf, 1-(cline-0.4)*hf], "k-", linewidth=2)
        ax.text(cd_x_start + (abs(rankpos(lowv+cd)-rankpos(lowv))*wf)/2,
                1-(cline-0.5)*hf, "CD", ha="center", size=11, fontweight='bold')

    # تقسیم متدها به چپ و راست
    half = int((k + 1) / 2)
    for i, (rank, name) in enumerate(zip(ssums, nnames)):
        y = 1 - (cline + ( (i % half) + 1) * 0.3) * hf
        x_rank = rankpos(rank) * wf

        if i < half: # سمت چپ
            x_text = (textspace - 0.1) * wf
            ax.plot([x_text, x_rank], [y, y], "k-", linewidth=0.8)
            ax.text(x_text - 0.02, y, name, ha="right", va="center", size=11)
        else: # سمت راست
            x_text = (width - textspace + 0.1) * wf
            ax.plot([x_rank, x_text], [y, y], "k-", linewidth=0.8)
            ax.text(x_text + 0.02, y, name, ha="left", va="center", size=11)

        ax.plot([x_rank, x_rank], [y, 1-cline*hf], "k-", linewidth=0.8)
        ax.plot([x_rank], [y], "ko", markersize=4)

    # رسم خطوط ضخیم (نشان‌دهنده عدم تفاوت معنادار)
# رسم خطوط ضخیم: تمام روش‌هایی که با روش کنترل تفاوت معنادار ندارند
    if cd:
        best_rank = ssums[0]

        # پیدا کردن دورترین الگوریتمی که هنوز داخل CD است
        max_rank_in_cd = best_rank
        for r in ssums[1:]:
            if abs(r - best_rank) <= cd:
                max_rank_in_cd = r

        # رسم خط ضخیم از بهترین روش تا آخرین روش داخل CD
        line_y = 1 - (cline + 0.05) * hf
        ax.plot(
            [rankpos(best_rank) * wf, rankpos(max_rank_in_cd) * wf],
            [line_y, line_y],
            "k-",
            linewidth=4
        )


    return fig

# ======================================
# داده‌ها و اجرای برنامه
# ======================================
methods = ['DES-bADE', 'DESKNN', 'KNORA-U', 'KNOP', 'KNORAE', 'META-DES', 'OLA', 'Rank', 'MCB', 'MLA', 'LCA']
avg_ranks = [2.85,      3.48,       4.65,     4.83,   4.85,       5.03,   6.27,   7.25,   7.72,   9.48, 9.58]

num_datasets = 30
alpha = "0.05"

# محاسبه CD مخصوص Bonferroni-Dunn
cd_val = compute_CD_bd(len(methods), num_datasets, alpha=alpha)
print(f"Critical Difference (CD) calculated: {cd_val:.4f}")

# رسم نمودار
graph_ranks(avg_ranks, methods, cd=cd_val, reverse=False)
plt.suptitle(f"Critical Difference Diagram (Bonferroni-Dunn, α = {alpha})",
             fontsize=13, fontweight='bold', y=1.2)

plt.show()