import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 👇 手动输入你的数据
# ═══════════════════════════════════════════════════════════════
accs_arr = [66.52, 58.47, 52.89, 39.30, 13.14, 10.0]  # 7-bit 到 2-bit 的准确率 (%)
baseline_acc = 60.5  # 64-bit Computer 基线准确率 (%)
# ═══════════════════════════════════════════════════════════════

bits_arr = [7, 6, 5, 4, 3, 2]
COLOR_REFERENCE = '#9E9E9E'

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(bits_arr, accs_arr, 'bo-', ms=8, lw=2, label='PNN Accuracy')
ax.axhline(y=baseline_acc, color=COLOR_REFERENCE, ls='--', lw=1.5,
           label=f'64-bit Computer ({baseline_acc:.2f}%)')
ax.set_xlabel('Bit Width', fontsize=8)
ax.set_ylabel('Test Accuracy (%)', fontsize=8)
ax.set_title('Quantization Bit-Width Sensitivity (CIFAR-10 Grayscale)', fontsize=9)
ax.set_xticks(bits_arr)
ax.set_xticklabels([f'{b}-bit\n({2**b} levels)' for b in bits_arr])
ax.legend(fontsize=7)
ax.grid(False)

for b, a in zip(bits_arr, accs_arr):
    ax.annotate(f'{a:.1f}%', (b, a), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=7, fontweight='bold')

fig.tight_layout()
fig.savefig('bit_sensitivity.png', dpi=150)
plt.close(fig)
print("✓ 图表已保存: bit_sensitivity.png")