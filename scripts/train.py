import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from cmcr.model import build_model
from cmcr.data import read_rows, subset, Collator, device_batch, metrics

def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def emit(out, name, obj):
    print(json.dumps(obj), flush=True)
    with (out / name).open('a') as f:
        f.write(json.dumps(obj) + '\n')

@torch.inference_mode()
def evaluate(model, loader, device, path=None):
    model.eval()
    ys, ps, records = ([], [], [])
    for batch in loader:
        b = device_batch(batch, device)
        r = model(text_mask=b['text_mask'], visual_mask=b['visual_mask'], **b['inputs'])
        probs = r['logits'].softmax(-1).cpu().tolist()
        labels = b['labels'].cpu().tolist()
        pred = r['logits'].argmax(-1).cpu().tolist()
        for i, y, p, prob in zip(b['ids'], labels, pred, probs):
            records.append(dict(id=i, label=y, prediction=p, probabilities=prob))
        ys.extend(labels)
        ps.extend(pred)
    result = metrics(ys, ps)
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        (path / 'predictions.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in records)))
        write(path / 'metrics.json', result)
    return result

def save(model, config, path, step, epoch, metric):
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / 'adapter')
    torch.save(model.custom_state(), path / 'module.pt')
    write(path / 'config.json', dict(config, selected_step=step, selected_epoch=epoch, validation=metric))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completed.json').exists() and (not args.smoke):
        print('completed; reuse existing result', flush=True)
        return
    if (out / 'train.jsonl').exists() and (not args.smoke):
        raise RuntimeError('Unfinished output exists: inspect manually; refusing blind overwrite/restart')
    seed = c['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    processor = AutoProcessor.from_pretrained(c['base_model'], local_files_only=True)
    collator = Collator(processor)
    splits = {s: read_rows(c[s + '_data']) for s in ('train', 'val', 'test')}
    ids = {s: {str(r['id']) for r in rows} for s, rows in splits.items()}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    tr = subset(splits['train'], c['train_samples'])
    va = subset(splits['val'], c['val_samples'])
    manifest = dict(counts={k: len(v) for k, v in splits.items()}, train_ids=[r['id'] for r in tr], val_ids=[r['id'] for r in va], input_hashes={s: hashlib.sha256(Path(c[s + '_data']).read_bytes()).hexdigest() for s in splits}, no_split_id_overlap=True, selection_rule='validation macro_f1, once per epoch; never test', config=c, train_counts={str(k): sum((x['label'] == k for x in tr)) for k in [0, 1]})
    write(out / 'manifest.json', manifest)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr, batch_size=c['batch_size'], shuffle=True, generator=generator, num_workers=2, collate_fn=collator)
    val = DataLoader(va, batch_size=2, num_workers=2, collate_fn=collator)
    model = build_model(c).to(device)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    custom = [p for n, p in model.named_parameters() if p.requires_grad and (not n.startswith('backbone.'))]
    parameters = dict(trainable=sum((p.numel() for p in lora + custom)), adapter=sum((p.numel() for p in custom)), total=sum((p.numel() for p in model.parameters())), added_vocabulary_tokens=0)
    write(out / 'parameters.json', parameters)
    print('PARAMETERS', parameters, flush=True)
    opt = torch.optim.AdamW([{'params': lora, 'lr': c['lora_lr']}, {'params': custom, 'lr': c['module_lr']}], weight_decay=0.01)
    accumulation = c['accumulation']
    steps_per = math.ceil(len(loader) / accumulation)
    total = steps_per * c['epochs']
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * 0.05)), total)
    step = 0
    best = -1.0
    start = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(c['epochs']):
        model.train()
        running = []
        for bi, batch in enumerate(loader):
            b = device_batch(batch, device)
            r = model(text_mask=b['text_mask'], visual_mask=b['visual_mask'], **b['inputs'])
            cls = torch.nn.functional.cross_entropy(r['logits'], b['labels'])
            loss = cls + c['energy_weight'] * r['energy']
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            group_size = min(accumulation, len(loader) - bi // accumulation * accumulation)
            (loss / group_size).backward()
            running.append([float(cls.detach()), float(r['energy'].detach()), float(r['energy'].detach())])
            if args.smoke:
                norms = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and ('rotator' in n or 'head' in n)}
                assert any((v > 0 for n, v in norms.items() if 'bases' in n))
                assert any((v > 0 for n, v in norms.items() if 'down' in n))
                opt.step()
                emit(out, 'smoke.jsonl', dict(event='forward_backward_optimizer_pass', variant=c['variant'], loss=float(loss), active_layers=list(model.layers), gradient_norms=norms, mechanism_stats={k: float(v) for k, v in r['stats'].items()}, peak_gpu_mib=torch.cuda.max_memory_allocated() / 2 ** 20))
                return
            if (bi + 1) % accumulation and bi + 1 < len(loader):
                continue
            torch.nn.utils.clip_grad_norm_(lora + custom, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or bi + 1 == len(loader):
                emit(out, 'train.jsonl', dict(step=step, total_steps=total, epoch=epoch + (bi + 1) / len(loader), cls=float(np.mean(running, axis=0)[0]), rotation_energy=float(np.mean(running, axis=0)[1]), energy=float(np.mean(running, axis=0)[2]), elapsed_sec=time.time() - start, peak_gpu_mib=torch.cuda.max_memory_allocated() / 2 ** 20, lr=sched.get_last_lr(), mechanism_stats={k: float(v) for k, v in r['stats'].items()}))
                running = []
        vm = evaluate(model, val, device)
        emit(out, 'validation.jsonl', dict(step=step, epoch=epoch + 1, **vm))
        if vm['macro_f1'] > best:
            best = vm['macro_f1']
            save(model, c, out / 'checkpoint-best', step, epoch + 1, vm)
    write(out / 'training_done.json', dict(step=step, total_steps=total, best_val_macro_f1=best, elapsed_sec=time.time() - start))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model.backbone, load_file(str(out / 'checkpoint-best/adapter/adapter_model.safetensors')))
    model.load_state_dict(torch.load(out / 'checkpoint-best/module.pt', map_location='cpu', weights_only=True), strict=False)
    test = DataLoader(splits['test'], batch_size=2, num_workers=2, collate_fn=collator)
    tm = evaluate(model, test, device, out / 'test')
    interventions = {}
    if c['variant'] == 'cmcr':
        diagnostic_rows = subset(splits['test'], c.get('diagnostic_samples', 128), seed=3907)
        dl = DataLoader(diagnostic_rows, batch_size=2, num_workers=2, collate_fn=collator)
        for mode in ('none', 'bypass', 'no_comm', 'reverse_rotation', 'common_frame'):
            model.rotator.intervention = mode
            interventions[mode] = evaluate(model, dl, device, out / 'diagnostics' / mode)
        model.rotator.intervention = 'none'
    write(out / 'completed.json', dict(test=tm, interventions=interventions, elapsed_sec=time.time() - start))
    print('COMPLETED', json.dumps(tm), flush=True)
if __name__ == '__main__':
    main()
