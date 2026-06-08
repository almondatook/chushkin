"""
Рисует кривую обучения из loss_curve.csv -> loss_curve.png
Запускать ПОСЛЕ (или во время) обучения: python plot_loss.py

На графике две линии:
  train loss — насколько модель угадывает следующий символ на ОБУЧАЮЩЕМ тексте
  val  loss — то же на ОТЛОЖЕННОМ тексте, который модель не видела

Пока обе падают вместе — модель учит ФОРМУ языка (хорошо).
Когда train продолжает падать, а val разворачивается вверх — началось
ПЕРЕОБУЧЕНИЕ: модель зубрит корпус наизусть вместо обобщения. Это и есть
тот самый слайд "вот почему нужен отложенный набор".
"""
import csv

CURVE_FILE = 'loss_curve.csv'
OUT = 'loss_curve.png'
BASELINE = None  # ln(vocab_size); подставим из данных, если хотим линию "наугад"

steps, train, val = [], [], []
with open(CURVE_FILE, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        steps.append(int(row['step']))
        train.append(float(row['train_loss']))
        val.append(float(row['val_loss']))

print(f"Точек в кривой: {len(steps)}")
if steps:
    print(f"Старт: train {train[0]:.3f} / val {val[0]:.3f}")
    print(f"Финал: train {train[-1]:.3f} / val {val[-1]:.3f}")
    gap = val[-1] - train[-1]
    print(f"Разрыв val-train на финале: {gap:+.3f} "
          f"({'есть переобучение' if gap > 0.15 else 'переобучения почти нет'})")

try:
    import matplotlib
    matplotlib.use('Agg')  # без окна, просто файл
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib не установлен — графика не будет, но цифры выше есть.")
    print("Поставить: python -m pip install matplotlib")
    raise SystemExit(0)

import math
plt.figure(figsize=(9, 5.5))
plt.plot(steps, train, label='train loss', linewidth=2)
plt.plot(steps, val, label='val loss', linewidth=2)
# Линия "наугад" = ln(76) ~= 4.33, если знаем размер словаря
try:
    vocab = len(set(open('onegin.txt', encoding='utf-8').read()))
    plt.axhline(math.log(vocab), color='grey', linestyle='--',
                label=f'наугад = ln({vocab}) = {math.log(vocab):.2f}')
except OSError:
    pass
plt.xlabel('шаг обучения')
plt.ylabel('cross-entropy loss')
plt.title('Tiny-LLM: как падает loss (char-level, «Онегин»)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print(f"\nГрафик сохранен: {OUT}")
