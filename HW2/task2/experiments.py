from train import *


print("Python:", platform.python_version())
for package in ("torch", "transformers", "numpy", "tqdm"):
    print(f"{package}: {version(package)}")
print("CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    for device_index in range(torch.cuda.device_count()):
        print(f"GPU {device_index}: {torch.cuda.get_device_name(device_index)}")
else:
    print("CUDA GPU недоступен. Определения можно редактировать; для обучения включите GPU.")


KINDS = ["fp32", "fp16", "static", "dynamic"]
RESULTS = {}

if RUN_TRAINING:
    if not Path(CONFIG.path).is_file():
        raise FileNotFoundError(f"Укажите путь к файлу задания в CONFIG.path: {CONFIG.path}")
    for kind in KINDS:
        print(f"=== kind: {kind} ===")
        RESULTS[kind] = train(SimpleNamespace(**{**vars(CONFIG), "kind": kind}))
        gc.collect()
        torch.cuda.empty_cache()
else:
    print("Обучение выключено. Установите RUN_TRAINING = True.")


from statistics import mean, stdev

TAIL = 20  # final quality = mean over the last TAIL iterations

print(f"{'kind':8} {'iters':>5} {'nan loss':>8} {'loss':>8} {'acc, %':>7} {'grad time, s':>19} {'final scale':>12}")
for kind, stats in RESULTS.items():
    losses = np.array(stats['loss'], dtype=np.float64)
    accuracies = np.array(stats['accuracy'], dtype=np.float64)
    mean_time = mean(stats['grad_time'])
    std_time = stdev(stats['grad_time'])
    print(
        f"{kind:8} {len(losses):5d} {int((~np.isfinite(losses)).sum()):8d} "
        f"{losses[-TAIL:].mean():8.4f} {accuracies[-TAIL:].mean() * 100:7.2f} "
        f"{mean_time:9.5f}+-{std_time:.5f} {stats['scale'][-1]:12.1f}"
    )


import matplotlib.pyplot as plt

COLORS = {"fp32": "#2a78d6", "fp16": "#eb6834", "static": "#1baf7a", "dynamic": "#eda100"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#fcfcfb")
for ax, key, title in zip(axes, ["loss", "accuracy"], ["Loss по итерациям", "Accuracy по итерациям"]):
    for kind, stats in RESULTS.items():
        values = np.array(stats[key], dtype=np.float64)
        finite = int(np.isfinite(np.array(stats['loss'], dtype=np.float64)).sum())
        label = kind if finite == len(values) else f"{kind} (конечный loss: {finite} из {len(values)} итераций)"
        ax.plot(np.arange(1, len(values) + 1), values, color=COLORS[kind], linewidth=2, label=label)
    ax.set_title(title, loc="left", color="#0b0b0b")
    ax.set_xlabel("итерация", color="#52514e")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.tick_params(colors="#898781")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
axes[0].legend(frameon=False, labelcolor="#0b0b0b")
fig.tight_layout()
fig.savefig(WORK_DIR / "amp_comparison.png", dpi=150)
plt.show()


BAD_SCALE = 2.0**24
RESULTS_SCALE = {}
for kind in ["static", "dynamic"]:
    RESULTS_SCALE[kind] = train(SimpleNamespace(**{**vars(CONFIG), "kind": kind, "scale": BAD_SCALE}))
    gc.collect()
    torch.cuda.empty_cache()

for kind, stats in RESULTS_SCALE.items():
    losses = np.array(stats['loss'], dtype=np.float64)
    print(f"{kind:8} loss {losses[-TAIL:].mean():.4f}  scale {stats['scale'][0]:.0f} -> {stats['scale'][-1]:.0f}")


LOADERS = ["base", "standard", "balanced", "sequenced"]
RESULTS_DATA = {}
for loader in LOADERS:
    print(f"=== dataloader: {loader} ===")
    RESULTS_DATA[loader] = train(SimpleNamespace(**{**vars(CONFIG), "kind": "dynamic", "dataloader": loader}))
    gc.collect()
    torch.cuda.empty_cache()

print(f"{'dataloader':10} {'iters':>5} {'prep, s':>8} {'train, s':>9} {'peak mem, MB':>13} {'loss':>8}")
for loader, stats in RESULTS_DATA.items():
    losses = np.array(stats['loss'], dtype=np.float64)
    print(f"{loader:10} {len(losses):5d} {stats['prep_time']:8.2f} {stats['train_time']:9.2f} "
          f"{stats['peak_memory_mb']:13.0f} {losses[-TAIL:].mean():8.4f}")


from torch.profiler import profile, schedule, tensorboard_trace_handler

OPTIMAL_SETUP = dict(kind="dynamic", dataloader="sequenced")
TRACE_DIR = WORK_DIR / "traces"


def profile_training(name, wait=2, warmup=2, active=5):
    """Profile `active` training steps in OPTIMAL_SETUP, as in bench_attention.py from seminar 02."""
    trace_dir = TRACE_DIR / name
    with profile(
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        train(
            SimpleNamespace(**{**vars(CONFIG), **OPTIMAL_SETUP}),
            profiler=prof,
            max_steps=wait + warmup + active,
        )
    gc.collect()
    torch.cuda.empty_cache()
    return prof, trace_dir


PROF_BEFORE, TRACE_BEFORE = profile_training("before")
print("Трасса для Perfetto:", *TRACE_BEFORE.glob("*.pt.trace.json"), sep="\n")


import json

SUSPECTS = {
    "aten::item": ".item(); долгий вызов = CPU ждёт GPU",
    "aten::_local_scalar_dense": "то же, внутренняя операция .item()",
    "cudaStreamSynchronize": "ожидание GPU (item, nonzero, synchronize)",
    "cudaDeviceSynchronize": "torch.cuda.synchronize()",
    "aten::nonzero": "индексация булевой маской",
    "aten::zero_": "запись нулей в тензоры (zero_grad без set_to_none)",
    "aten::isfinite": "проверка градиентов в скейлере",
    "aten::mul_": "деление градиентов на scale",
    "aten::linear": "линейные слои",
    "aten::to": "приведение типа / перенос на устройство",
    "aten::copy_": "копирование данных",
    "aten::masked_fill_": "построение маски внимания",
    "aten::argmax": "accuracy",
    "Optimizer.step#Adam.step": "шаг оптимизатора",
}


def load_trace(trace_dir):
    path = max(Path(trace_dir).glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
    with open(path) as trace_file:
        trace = json.load(trace_file)
    return path, [event for event in trace["traceEvents"] if event.get("ph") == "X"]


def summarize_trace(trace_dir, top=12):
    path, events = load_trace(trace_dir)
    steps = [e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep#")]
    num_steps = len(steps)
    step_ms = sum(e["dur"] for e in steps) / num_steps / 1000
    print(f"{path.name}: {path.stat().st_size / 2**20:.1f} МБ, шагов в трассе: {num_steps}, средний шаг: {step_ms:.1f} мс\n")

    stats = {}
    for event in events:
        category, name = event.get("cat"), event["name"]
        if category in ("user_annotation", "gpu_user_annotation", "Trace", "python_function"):
            continue
        entry = stats.setdefault(category, {}).setdefault(name, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += event["dur"]
        entry[2] = max(entry[2], event["dur"])

    print(f"{'категория':18} {'событий/шаг':>12} {'мс/шаг':>10}")
    for category, names in sorted(stats.items(), key=lambda item: -sum(v[1] for v in item[1].values())):
        calls = sum(v[0] for v in names.values()) / num_steps
        total_ms = sum(v[1] for v in names.values()) / num_steps / 1000
        print(f"{category:18} {calls:12.1f} {total_ms:10.2f}")

    kernel_ms = sum(v[1] for v in stats.get("kernel", {}).values()) / num_steps / 1000
    if kernel_ms:
        print(f"\nGPU занят kernel-ами {kernel_ms:.1f} мс из {step_ms:.1f} мс шага ({kernel_ms / step_ms:.0%})")

    for category in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cpu_op"):
        names = stats.get(category)
        if not names:
            continue
        print(f"\nТоп по времени, категория {category}:")
        print(f"  {'имя':58} {'вызовов/шаг':>12} {'мс/шаг':>9} {'% шага':>7}")
        for name, (calls, duration, _) in sorted(names.items(), key=lambda item: -item[1][1])[:top]:
            ms = duration / num_steps / 1000
            print(f"  {name[:58]:58} {calls / num_steps:12.1f} {ms:9.3f} {ms / step_ms:7.1%}")

    print("\nНа что смотреть:")
    print(f"  {'имя':28} {'вызовов/шаг':>12} {'мс/шаг':>9} {'макс, мс':>9}  что это")
    merged = {}
    for names in stats.values():
        for name, (calls, duration, longest) in names.items():
            entry = merged.setdefault(name, [0, 0.0, 0.0])
            entry[0] += calls
            entry[1] += duration
            entry[2] = max(entry[2], longest)
    for name, meaning in SUSPECTS.items():
        if name in merged:
            calls, duration, longest = merged[name]
            print(f"  {name:28} {calls / num_steps:12.1f} {duration / num_steps / 1000:9.3f} {longest / 1000:9.3f}  {meaning}")


sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
print(PROF_BEFORE.key_averages().table(sort_by=sort_by, row_limit=15))
summarize_trace(TRACE_BEFORE)


FIXES_OFF = dict(
    fused_adam=False,            
    zero_grad_to_none=False,     
    fast_finite_check=False,     
    dense_loss=False,            
    log_every=1,                 
    skip_dead_projection=False,  
    cache_causal_mask=False,     
    hidden_dim=1020,             
)
FIXES_ON = dict(
    fused_adam=True, zero_grad_to_none=True, fast_finite_check=True, dense_loss=True, log_every=10,
    skip_dead_projection=True, cache_causal_mask=True,
)


class DenseLMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        logits = outputs[:, :-1]
        targets = tokens[:, 1:]
        if loss_mask is not None:
            mask = loss_mask.to(device=tokens.device, dtype=torch.bool)
        else:
            tokens_lens = torch.as_tensor(tokens_lens, device=tokens.device)
            positions = torch.arange(tokens.shape[1] - 1, device=tokens.device)
            mask = positions[None, :] < (tokens_lens[:, None] - 1)
        targets = targets.masked_fill(mask.logical_not(), self.ignore_index)
        return super().forward(logits.flatten(0, 1), targets.flatten())


def collect_grads(optimizer):
    return [param.grad for group in optimizer.param_groups for param in group["params"] if param.grad is not None]


def all_finite_one_pass(grads):
    return torch.stack([grad.sum(dtype=torch.float32) for grad in grads]).isfinite().all().item()


class FastStaticGradScaler(StaticGradScaler):
    def step(self, optimizer):
        with torch.no_grad():
            grads = collect_grads(optimizer)
            if grads and not all_finite_one_pass(grads):
                optimizer.zero_grad()
                return
            for grad in grads:
                grad.mul_(self.inv_scale)
        optimizer.step()


class FastDynamicGradScaler(DynamicGradScaler):
    def step(self, optimizer):
        with torch.no_grad():
            grads = collect_grads(optimizer)
            if grads and not all_finite_one_pass(grads):
                optimizer.zero_grad()
                self._scale = max(self._scale / self.factor, self.min_scale)
                self.counter = 0
                return
            for grad in grads:
                grad.mul_(1 / self._scale)
        optimizer.step()
        self.counter += 1


class FixedGPT2LikeModel(GPT2LikeModel):
    def __init__(self, vocab_size, hidden_dim=1020, skip_dead_projection=True, cache_causal_mask=True):
        super().__init__(vocab_size, hidden_dim=hidden_dim)
        self.skip_dead_projection = skip_dead_projection
        max_length = self.positional_encoding.pe.shape[0]
        causal_mask = torch.full((max_length, max_length), float("-inf")).triu_(diagonal=1)
        self.register_buffer("causal_mask", causal_mask if cache_causal_mask else None, persistent=False)

    def forward(self, x, attention_mask):
        x = x.transpose(0, 1) 
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.hidden_projector(x)
        if not self.skip_dead_projection:
            y = self.hidden_projector2(x)
        if attention_mask is None and self.causal_mask is not None:
            attention_mask = self.causal_mask[:x.size(0), :x.size(0)]
        elif attention_mask is None:
            attention_mask = torch.tril(
                torch.ones((x.size(0), x.size(0)), dtype=torch.bool, device=x.device)
            )
        else:
            attention_mask = attention_mask.to(x.device)
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(
                attention_mask.logical_not(), float("-inf")
            )
        attention_mask = attention_mask.to(x.dtype)

        out = self.decoder(tgt=x, memory=x, tgt_mask=attention_mask, memory_mask=attention_mask)
        out = self.output_linear(out)
        return out.transpose(0, 1)


def train_epoch_fixed(train_loader, model, criterion, metric, optimizer, device, kind, scaler, args,
                      profiler=None, max_steps=None):
    model.train()
    autocast = Autocast(enabled=kind in ('static', 'dynamic'))
    losses, accuracies, scales, events = [], [], [], []
    pbar = tqdm(enumerate(train_loader))
    for i, data in pbar:
        tokens, tokens_lens, attention_mask = data['tokens'].to(device), data['lengthes'], data['attention_mask']
        loss_mask = data.get('loss_mask')
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        optimizer.zero_grad(set_to_none=args.zero_grad_to_none)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with autocast:
            outputs = model(tokens, attention_mask)
            loss = criterion(outputs, tokens, tokens_lens, loss_mask=loss_mask)
        if kind == 'fp16' or kind == 'fp32':
            loss.backward()
        else:
            scaler.scale(loss).backward()
        end.record()
        events.append((start, end))
        if args.log_every == 1:
            torch.cuda.synchronize()  # the original code waits for the GPU on every step
        scales.append(1.0 if scaler is None else scaler._scale)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.detach())  
        if i % args.log_every == 0:
            accuracy = metric(outputs, tokens, tokens_lens, loss_mask=loss_mask)
            accuracies.append(accuracy)
            pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")
        if profiler is not None:
            profiler.step()
        if max_steps is not None and i + 1 >= max_steps:
            break
    torch.cuda.synchronize()  
    return {
        'loss': torch.stack(losses).float().tolist(),
        'accuracy': torch.stack(accuracies).float().tolist(),  # one value per log_every steps
        'grad_time': [start.elapsed_time(end) / 1000 for start, end in events],
        'scale': scales,
    }


def train_fixed(args, profiler=None, max_steps=None, model_hook=None):
    set_global_seed(42)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    prep_start = time.perf_counter()
    dataloader = get_dataloader(args.dataloader, args.batch_size, args.path, args.k)
    prep_time = time.perf_counter() - prep_start
    model = FixedGPT2LikeModel(
        dataloader.dataset.tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        skip_dead_projection=args.skip_dead_projection,
        cache_causal_mask=args.cache_causal_mask,
    ).to(device)
    if args.kind == 'fp16':
        model = model.half()
    if model_hook is not None:
        model = model_hook(model)
    criterion = DenseLMCrossEntropyLoss() if args.dense_loss else LMCrossEntropyLoss()
    metric = LMAccuracy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True if args.fused_adam else None)
    scaler = None
    if args.kind == 'static':
        scaler = (FastStaticGradScaler if args.fast_finite_check else StaticGradScaler)(args.scale)
    elif args.kind == 'dynamic':
        scaler = (FastDynamicGradScaler if args.fast_finite_check else DynamicGradScaler)(
            scale=args.scale,
            factor=args.factor,
            patience=args.patience,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
    stats = {'loss': [], 'accuracy': [], 'grad_time': [], 'scale': []}
    epoch_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_epochs + 1)]
    epoch_events[0].record()
    for epoch in range(0, args.num_epochs):
        epoch_stats = train_epoch_fixed(
            train_loader=dataloader,
            model=model,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer,
            device=device,
            kind=args.kind,
            scaler=scaler,
            args=args,
            profiler=profiler,
            max_steps=max_steps,
        )
        epoch_events[epoch + 1].record()
        for key, values in epoch_stats.items():
            stats[key].extend(values)
    torch.cuda.synchronize()
    stats['prep_time'] = prep_time
    stats['epoch_time'] = [start.elapsed_time(end) / 1000 for start, end in zip(epoch_events, epoch_events[1:])]
    stats['train_time'] = epoch_events[0].elapsed_time(epoch_events[-1]) / 1000
    stats['peak_memory_mb'] = torch.cuda.max_memory_allocated(device) / 2**20
    return stats


def run_fixed(profiler=None, max_steps=None, model_hook=None, **fixes):
    args = SimpleNamespace(**{**vars(CONFIG), **OPTIMAL_SETUP, **FIXES_OFF, **fixes})
    stats = train_fixed(args, profiler=profiler, max_steps=max_steps, model_hook=model_hook)
    gc.collect()
    torch.cuda.empty_cache()
    return stats


ABLATION_STEPS = [
    ("исходный код", {}),
    ("+ fused Adam", dict(fused_adam=True)),
    ("+ zero_grad(set_to_none=True)", dict(zero_grad_to_none=True)),
    ("+ проверка градиентов за 1 проход", dict(fast_finite_check=True)),
    ("+ loss без булевой индексации", dict(dense_loss=True)),
    ("+ .item() и accuracy раз в 10 шагов", dict(log_every=10)),
    ("+ без hidden_projector2", dict(skip_dead_projection=True)),
    ("+ hidden_dim 1020 -> 1024", dict(hidden_dim=1024)),
]
RESULTS_ABLATION = {}

fixes = {}
for name, change in ABLATION_STEPS:
    fixes.update(change)
    print(f"=== {name} ===")
    RESULTS_ABLATION[name] = run_fixed(**fixes)

baseline_time = previous_time = RESULTS_ABLATION["исходный код"]['train_time']
print(f"{'шаг':36} {'iters':>5} {'train, s':>9} {'мс/шаг':>7} {'к пред.':>8} {'к исх.':>7} {'peak mem, MB':>13} {'loss':>8}")
for name, stats in RESULTS_ABLATION.items():
    losses = np.array(stats['loss'], dtype=np.float64)
    print(
        f"{name:36} {len(losses):5d} {stats['train_time']:9.2f} {stats['train_time'] / len(losses) * 1000:7.1f} "
        f"{previous_time / stats['train_time']:7.2f}x {baseline_time / stats['train_time']:6.2f}x "
        f"{stats['peak_memory_mb']:13.0f} {losses[-TAIL:].mean():8.4f}"
    )
    previous_time = stats['train_time']


from torch.autograd import DeviceType


def profile_fixed(name, wait=2, warmup=2, active=10, **fixes):
    trace_dir = TRACE_DIR / name
    with profile(
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        run_fixed(profiler=prof, max_steps=wait + warmup + active, **fixes)
    return prof, trace_dir


def device_time(evt):
    return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)


def gpu_ms_per_step(prof):
    averages = prof.key_averages()
    num_steps = max(evt.count for evt in averages if evt.key.startswith("ProfilerStep"))
    on_gpu = torch.cuda.is_available()
    operations, optimizer_calls = {}, {}
    for evt in averages:
        if evt.key.startswith("ProfilerStep"):
            continue
        if evt.device_type != DeviceType.CPU:
            # GPU-side rows: kernels (already counted inside the operations) and annotations such as Optimizer.step
            if evt.key.startswith("Optimizer."):
                optimizer_calls[evt.key] = (evt.count / num_steps, device_time(evt) / num_steps / 1000)
            continue
        total = device_time(evt) if on_gpu else evt.self_cpu_time_total
        if total > 0:
            operations[evt.key] = (evt.count / num_steps, total / num_steps / 1000)
    return operations, optimizer_calls


def print_rows(before, after, names):
    for name in names:
        calls_before, ms_before = before.get(name, (0, 0))
        calls_after, ms_after = after.get(name, (0, 0))
        print(f"{name[:44]:44} {calls_before:10.1f}{calls_after:10.1f} {ms_before:10.3f}{ms_after:10.3f}")


def compare_profiles(before, after, top=25):
    (before, before_optimizer), (after, after_optimizer) = gpu_ms_per_step(before), gpu_ms_per_step(after)
    print(f"{'операция':44} {'вызовов/шаг':>20} {'GPU, мс/шаг':>20}")
    print(f"{'':44} {'до':>10}{'после':>10} {'до':>10}{'после':>10}")
    largest = lambda name: -max(before.get(name, (0, 0))[1], after.get(name, (0, 0))[1])
    print_rows(before, after, sorted(set(before) | set(after), key=largest)[:top])
    total_before, total_after = sum(v[1] for v in before.values()), sum(v[1] for v in after.values())
    print(f"{'всего по всем операциям':44} {'':20} {total_before:10.3f}{total_after:10.3f}")
    if before_optimizer or after_optimizer:
        print("\nИз них внутри вызовов оптимизатора:")
        print_rows(before_optimizer, after_optimizer, sorted(set(before_optimizer) | set(after_optimizer)))


PROF_AFTER, TRACE_AFTER = profile_fixed("after", **FIXES_ON)
print("Трасса для Perfetto:", *TRACE_AFTER.glob("*.pt.trace.json"), sep="\n")
compare_profiles(PROF_BEFORE, PROF_AFTER)
summarize_trace(TRACE_AFTER)


BONUS_EPOCHS = 3
RESULTS_BONUS = {}


def run_bonus(name, model_hook=None):
    print(f"=== {name} ===")
    torch._dynamo.reset()
    RESULTS_BONUS[name] = run_fixed(model_hook=model_hook, num_epochs=BONUS_EPOCHS, **FIXES_ON)


def print_bonus():
    epochs = " ".join(f"{'эпоха ' + str(i + 1) + ', с':>11}" for i in range(BONUS_EPOCHS))
    print(f"{'вариант':26} {epochs} {'peak mem, MB':>13} {'loss':>8}")
    for name, stats in RESULTS_BONUS.items():
        times = " ".join(f"{t:11.2f}" for t in stats['epoch_time'])
        print(f"{name:26} {times} {stats['peak_memory_mb']:13.0f} {np.mean(stats['loss'][-TAIL:]):8.4f}")


run_bonus("все исправления")
run_bonus("+ torch.compile", lambda model: torch.compile(model, dynamic=True))
print_bonus()


import subprocess
import sys
import torch.utils.cpp_extension as cpp_extension

APEX_DIR = WORK_DIR / "apex"
print("CUDA_HOME:", cpp_extension.CUDA_HOME, "| CUDA в torch:", torch.version.cuda)
if not APEX_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/apex", str(APEX_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=True)
sys.path.insert(0, str(APEX_DIR))
sys.modules["fused_layer_norm_cuda"] = cpp_extension.load(
    name="fused_layer_norm_cuda",
    sources=[str(APEX_DIR / "csrc" / "layer_norm_cuda.cpp"), str(APEX_DIR / "csrc" / "layer_norm_cuda_kernel.cu")],
    extra_include_paths=[str(APEX_DIR / "csrc")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-maxrregcount=50", "-O3", "--use_fast_math"],
)
from apex.normalization import FusedLayerNorm
print("FusedLayerNorm собран:", FusedLayerNorm(8))


class Fp32FusedLayerNorm(FusedLayerNorm):
    def forward(self, input):
        return super().forward(input.to(self.weight.dtype))


def use_fused_layer_norm(model):
    for name in ("norm1", "norm2", "norm3"):
        old = getattr(model.decoder, name)
        new = Fp32FusedLayerNorm(old.normalized_shape, eps=old.eps).to(old.weight.device)
        new.load_state_dict(old.state_dict())
        setattr(model.decoder, name, new)
    return model


run_bonus("+ FusedLayerNorm (Apex)", use_fused_layer_norm)
print_bonus()

PROF_APEX, TRACE_APEX = profile_fixed("apex", model_hook=use_fused_layer_norm, **FIXES_ON)
compare_profiles(PROF_AFTER, PROF_APEX, top=15)
