"""
=====================================================================
  GENERATE  ·  заставляем ОБУЧЕННУЮ модель говорить
=====================================================================
Здесь НЕТ обучения. Мы грузим готовые веса (tiny_gpt.pt) и сэмплируем
текст по одному символу. Поэтому работает мгновенно — в отличие от
tiny_llm.py, который учит модель с нуля.

КАК ЗАПУСКАТЬ (примеры для воркшопа):
  python generate.py                          # 500 символов, температура 0.8
  python generate.py --len 1000 --temp 0.5    # длиннее и СВЯЗНЕЕ (зануднее)
  python generate.py --temp 1.1               # рискованнее (ярче, но больше бреда)
  python generate.py --prompt "Татьяна"       # с ЗАТРАВКОЙ — продолжит ее
  python generate.py --seed 42                # повторяемый результат

Когда переучим на своем корпусе — тот же скрипт, просто укажи файлы:
  python generate.py --corpus mycorpus.txt --ckpt tiny_gpt_luba.pt
=====================================================================
"""
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- гиперпараметры: ТЕ ЖЕ, что в tiny_llm.py (конфиг A) ---
# Если разойдутся с обучением — веса не загрузятся (ошибка размерностей).
block_size = 128
n_embd     = 128
n_head     = 4
n_layer    = 4
dropout    = 0.0       # при генерации регуляризация не нужна (и eval() ее гасит)
device     = 'cpu'

# --------------------------------------------------------------------
#  Параметры запуска (то, чем студенты будут играть)
# --------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Генерация текста обученной Tiny-LLM")
ap.add_argument('--corpus', default='onegin.txt', help='корпус для словаря (тот же, на чем учили)')
ap.add_argument('--ckpt',   default='tiny_gpt.pt', help='файл с весами')
ap.add_argument('--len',    type=int,   default=500, help='сколько символов сгенерить')
ap.add_argument('--temp',   type=float, default=0.8, help='температура: <1 связнее, >1 рискованнее')
ap.add_argument('--prompt', default='',  help='затравка — модель ее продолжит')
ap.add_argument('--seed',   type=int,   default=None, help='зерно ГСЧ для повторяемости')
args = ap.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

# --------------------------------------------------------------------
#  Токенизатор — строим из того же корпуса, на котором учили.
#  Словарь обязан совпасть с обучением, иначе числа поедут.
# --------------------------------------------------------------------
with open(args.corpus, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: ''.join(itos[i] for i in l)

# --------------------------------------------------------------------
#  Архитектура — точная копия из tiny_llm.py (иначе веса не лягут)
# --------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = self.blocks(tok + pos)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --------------------------------------------------------------------
#  Грузим веса и генерируем
# --------------------------------------------------------------------
model = TinyGPT().to(device)
try:
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
except FileNotFoundError:
    raise SystemExit(f"Не найден файл весов: {args.ckpt}. Сначала обучи модель (python tiny_llm.py).")
except RuntimeError as e:
    raise SystemExit(f"Веса не подошли к архитектуре ({args.ckpt}).\n"
                     f"Скорее всего, гиперпараметры в generate.py разошлись с обучением.\n{e}")
model.eval()

# Затравка: кодируем символы, которых нет в словаре — выкидываем с предупреждением.
if args.prompt:
    unknown = sorted({c for c in args.prompt if c not in stoi})
    if unknown:
        print(f"(символы не из словаря выкинуты: {unknown})")
    ids = [stoi[c] for c in args.prompt if c in stoi]
    if not ids:
        ids = [0]
    ctx = torch.tensor([ids], dtype=torch.long, device=device)
else:
    # пустой старт — "новорожденный" контекст из одного нулевого токена
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)

print(f"\n--- генерация | corpus={args.corpus} | ckpt={args.ckpt} | "
      f"temp={args.temp} | {args.len} символов ---\n")
out = model.generate(ctx, args.len, temperature=args.temp)
print(decode(out[0].tolist()))
