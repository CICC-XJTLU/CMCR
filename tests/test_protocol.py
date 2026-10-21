"""Fast checks for configuration and training/evaluation separation."""

import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = ast.parse((ROOT / "scripts/train.py").read_text(encoding="utf-8"))
        cls.main = next(n for n in cls.train.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        cls.eval = next(n for n in cls.train.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")

    def test_python_syntax(self):
        for folder in ["cmcr", "scripts", "tests"]:
            for path in (ROOT / folder).glob("*.py"):
                with self.subTest(path=path.name):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_configurations(self):
        paths = list((ROOT / "configs").glob("*/*.json"))
        self.assertEqual(len(paths), 6)
        expected = {"cmcr", "static", "no_comm"}
        for folder in ["qwen3vl8b", "qwen3vl4b"]:
            configs = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "configs" / folder).glob("*.json")]
            self.assertEqual({c["variant"] for c in configs}, expected)
            for c in configs:
                with self.subTest(backbone=folder, variant=c["variant"]):
                    self.assertEqual(c["layers"], [11, 23, 31])
                    self.assertEqual((c["rank"], c["seed"], c["epochs"]), (16, 42, 2))
                    self.assertEqual((c["train_samples"], c["val_samples"]), (19344, 2393))
                    self.assertEqual((c["batch_size"], c["accumulation"]), (2, 8))
                    self.assertEqual((c["lora_lr"], c["module_lr"], c["energy_weight"]), (5e-5, 2e-4, 1e-3))
                    for split in ["train", "val", "test"]:
                        self.assertEqual(c[split + "_data"], "data/clean/" + split + ".jsonl")
                    self.assertTrue(c["base_model"].startswith("models/Qwen3-VL-"))
                    self.assertEqual(c["output_dir"], "outputs/" + folder + "/" + c["variant"])

    def test_train_loader_uses_train_only(self):
        assignments = {n.targets[0].id: n.value for n in ast.walk(self.main)
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.unparse(assignments["tr"]), "subset(splits['train'], c['train_samples'])")
        self.assertEqual(ast.unparse(assignments["loader"].args[0]), "tr")
        backwards = [n for n in ast.walk(self.main) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "backward"]
        self.assertEqual(len(backwards), 1)
        batch_loop = next(n for n in ast.walk(self.main) if isinstance(n, ast.For)
                          and ast.unparse(n.iter) == "enumerate(loader)")
        self.assertIn(backwards[0], list(ast.walk(batch_loop)))

    def test_validation_checkpoint_selection(self):
        selections = [n for n in ast.walk(self.main) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "vm['macro_f1'] > best"]
        self.assertEqual(len(selections), 1)
        self.assertIn("save(model, c, out / 'checkpoint-best', step, epoch + 1, vm)",
                      ast.unparse(selections[0]))
        self.assertIn("vm = evaluate(model, val, device)", ast.unparse(self.main))

    def test_evaluation_disables_gradients(self):
        self.assertIn("torch.inference_mode()", [ast.unparse(d) for d in self.eval.decorator_list])
        self.assertIn("model.eval()", ast.unparse(self.eval))
        self.assertNotIn("backward", ast.unparse(self.eval))
        self.assertNotIn("opt.step", ast.unparse(self.eval))

    def test_test_evaluation_follows_training(self):
        statements = self.main.body
        epoch_loop = next(i for i, n in enumerate(statements) if isinstance(n, ast.For)
                          and ast.unparse(n.iter) == "range(c['epochs'])")
        test_loader = next(i for i, n in enumerate(statements) if isinstance(n, ast.Assign)
                           and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "test")
        self.assertGreater(test_loader, epoch_loop)
        self.assertIn("DataLoader(splits['test']", ast.unparse(statements[test_loader]))

    def test_labels_are_not_model_inputs(self):
        data = ast.parse((ROOT / "cmcr/data.py").read_text(encoding="utf-8"))
        collate = next(n for n in ast.walk(data) if isinstance(n, ast.FunctionDef) and n.name == "__call__")
        prompt = next(n.value for n in ast.walk(collate) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "messages" for t in n.targets))
        self.assertNotIn("label", ast.unparse(prompt))
        for call in ast.walk(self.train):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "model":
                self.assertNotIn("labels", [k.arg for k in call.keywords])


if __name__ == "__main__":
    unittest.main()
