"""
=====================================================================
  TINY-LLM  ·  "модель модели" для воркшопа
=====================================================================
Один файл, который проходит ВЕСЬ конвейер обучения языковой модели
в миниатюре — на одном CPU, за минуты, на одном текстовом файле.

Что мы честно воспроизводим из "взрослого" пайплайна:
  1. данные            -> читаем текстовый файл
  2. токенизация       -> character-level (1 символ = 1 токен)
  3. train/val split   -> чтобы видеть переобучение
  4. архитектура       -> настоящий трансформер-декодер (self-attention)
  5. цель обучения     -> предсказать следующий токен (cross-entropy)
  6. цикл обучения      -> forward -> loss -> backward -> step
  7. генерация         -> сэмплируем по одному токену

Что мы СОЗНАТЕЛЬНО выкинули (и честно скажем студентам):
  - BPE-токенизацию (у нас буквы вместо субслов)
  - масштаб (у нас ~0.8M параметров против сотен миллиардов)
  - распределенное обучение на тысячах GPU
  - alignment: ни SFT, ни RLHF — у нас голая base model
Идейный родитель: nanoGPT / makemore Андрея Карпатого.

ГЛАВНЫЙ НОМЕР ВОРКШОПА: по ходу обучения скрипт складывает в samples.txt
снимки генерации на разных шагах. Прокрутите файл сверху вниз — и увидите,
как из чистого шума постепенно проступает ФОРМА языка (не смысл!):
шум -> буквы с правильной частотой -> псевдослова с чередованием
гласных/согласных -> псевдострочки с пунктуацией на местах.
=====================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------------------------
#  ГИПЕРПАРАМЕТРЫ  (все маленькое — чтобы крутилось на CPU за минуты)
# --------------------------------------------------------------------
batch_size   = 32      # сколько кусков текста жуем за один шаг
block_size   = 128     # длина контекста: на сколько символов назад смотрим
max_iters    = 5000    # сколько шагов обучения (у нас нет таймаута — добиваем до сходимости)
eval_every   = 250     # как часто меряем loss на валидации (и пишем точку в график)
eval_iters   = 50      # по скольки батчам усредняем оценку
n_embd       = 128     # размерность эмбеддинга (вектор на каждый токен)
n_head       = 4       # число голов внимания
n_layer      = 4       # число трансформер-блоков
dropout      = 0.1     # регуляризация: глушим часть нейронов
learning_rate = 3e-4
seed         = 1337

# Вехи, на которых сохраняем СНИМОК генерации (главный номер воркшопа).
# Подобраны так, чтобы поймать качественные переходы: шум -> буквы ->
# псевдослова -> строчки. Финальный шаг (max_iters) добавляем автоматически.
sample_steps  = [0, 100, 300, 700, 1500, 3000, max_iters]
sample_tokens = 400    # длина одного демо-сэмпла в символах
SAMPLES_FILE  = 'samples.txt'    # сюда копится эволюция генерации
CURVE_FILE    = 'loss_curve.csv' # сюда копится кривая train/val loss

torch.manual_seed(seed)
device = 'cpu'

# --------------------------------------------------------------------
#  ШАГ 1-2: ДАННЫЕ + ТОКЕНИЗАЦИЯ
# --------------------------------------------------------------------
with open('onegin.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Весь "токенизатор" — это отсортированный набор уникальных символов.
# Вот он, словарь, целиком, глазами. Никакого черного ящика.
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # символ -> число
itos = {i: ch for i, ch in enumerate(chars)}   # число  -> символ
encode = lambda s: [stoi[c] for c in s]        # строка -> список чисел
decode = lambda l: ''.join(itos[i] for i in l) # список чисел -> строка

print(f"Символов в корпусе: {len(text):,}")
print(f"Размер словаря: {vocab_size}")
print(f"Словарь: {''.join(chars)}")

# Весь текст -> один длинный тензор чисел
data = torch.tensor(encode(text), dtype=torch.long)

# ШАГ 3: режем на train / val (90/10), чтобы ловить переобучение
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Берем batch_size случайных кусков длины block_size.
    x — вход, y — тот же кусок, сдвинутый на 1 вправо (это и есть
    'следующий токен', который модель должна угадать в каждой позиции)."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# --------------------------------------------------------------------
#  ШАГ 4: АРХИТЕКТУРА — ТРАНСФОРМЕР-ДЕКОДЕР
# --------------------------------------------------------------------
class Head(nn.Module):
    """Одна голова self-attention. Каждый токен смотрит на предыдущие
    и решает, у кого что взять. Маска не дает подглядывать в будущее."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # насколько каждый токен "интересен" каждому (внимание)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # маска: будущее запрещено — иначе модель сжульничает
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    """Несколько голов параллельно — разные 'точки зрения' на контекст."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """Полносвязный блок: даем модели 'подумать' над собранной информацией."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Трансформер-блок: внимание + feed-forward, оба с residual-связями
    (x + ...) и нормализацией. Вот этот блок и штампуется n_layer раз."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # внимание + остаточная связь
        x = x + self.ffwd(self.ln2(x))  # подумать + остаточная связь
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # эмбеддинг токена + эмбеддинг позиции (модель должна знать ПОРЯДОК)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # проекция в словарь

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # для каждой позиции — баллы по всем символам

        if targets is None:
            loss = None
        else:
            # ШАГ 5: ЦЕЛЬ — угадать следующий токен. Меряем cross-entropy.
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """ШАГ 7: генерация. Берем предсказание для последней позиции,
        превращаем в вероятности, тянем один символ, дописываем, повторяем."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]      # обрезаем по контексту
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --------------------------------------------------------------------
#  ВСПОМОГАТЕЛЬНОЕ: снимок генерации на текущем шаге обучения
# --------------------------------------------------------------------
def snapshot(model, step, losses, temp=0.8, n=sample_tokens):
    """Генерим кусок текста ТЕКУЩЕЙ моделью и дописываем в samples.txt
    с пометкой шага и loss. Прокрутив файл сверху вниз, студент видит,
    как модель проходит путь: шум -> буквы с правильной частотой ->
    псевдослова с чередованием гласных/согласных -> псевдострочки."""
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    txt = decode(model.generate(ctx, n, temperature=temp)[0].tolist())
    model.train()
    header = (f"\n{'='*64}\n"
              f"ШАГ {step:5d}  |  train {losses['train']:.3f}  "
              f"val {losses['val']:.3f}  |  temp {temp}\n"
              f"{'='*64}\n")
    with open(SAMPLES_FILE, 'a', encoding='utf-8') as f:
        f.write(header + txt + "\n")
    print(header + txt)
    return txt

# --------------------------------------------------------------------
#  ШАГ 6: ЦИКЛ ОБУЧЕНИЯ
# --------------------------------------------------------------------
model = TinyGPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nПараметров в модели: {n_params:,}  (~{n_params/1e6:.2f}M)")

# Для контраста: чем "тупит" необученная модель — случайный шум.
# Теоретический loss наобум = ln(vocab_size).
import math
print(f"Ожидаемый loss у необученной модели (наугад): {math.log(vocab_size):.3f}\n")

# Заводим файлы заново (чистим прошлый прогон)
open(SAMPLES_FILE, 'w', encoding='utf-8').close()
with open(CURVE_FILE, 'w', encoding='utf-8') as f:
    f.write("step,train_loss,val_loss\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

import time
t0 = time.time()
for it in range(max_iters + 1):
    # Замер loss + точка в график. На вехах — еще и снимок генерации.
    if it % eval_every == 0 or it in sample_steps:
        losses = estimate_loss(model)
        with open(CURVE_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f}\n")
        dt = time.time() - t0
        print(f"шаг {it:5d} | train loss {losses['train']:.3f} | "
              f"val loss {losses['val']:.3f} | {dt:5.0f}s")
        if it in sample_steps:
            snapshot(model, it, losses)

    if it == max_iters:
        break  # на финале только мерили/сэмплировали, лишний шаг не делаем

    # сам шаг обучения: forward -> loss -> backward -> step
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("=" * 64)
print("=== ЭФФЕКТ ТЕМПЕРАТУРЫ (одна модель, один и тот же пустой старт) ===")
print("0.5 — связнее и зануднее, повторы; 1.0 — рискованнее и бредовее.\n")
final_losses = estimate_loss(model)
for temp in [0.5, 0.8, 1.0]:
    snapshot(model, max_iters, final_losses, temp=temp, n=500)

torch.save(model.state_dict(), 'tiny_gpt.pt')
print("\n" + "=" * 64)
print("Веса сохранены в tiny_gpt.pt")
print(f"Эволюция генерации:  {SAMPLES_FILE}")
print(f"Кривая обучения:     {CURVE_FILE}")
