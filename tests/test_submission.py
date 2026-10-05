"""Static checks for the completed student submission package."""
import ast
import json
import unittest
import zipfile
from xml.etree import ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submissions" / "2A202603012_BuiQuangVinh"
CODE = SUBMISSION / "code"


class SubmissionPackageTests(unittest.TestCase):
    def test_completed_modules_parse_and_have_no_stubs(self):
        for path in sorted(CODE.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            self.assertFalse(any(isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                                 and getattr(node.exc.func, "id", "") == "NotImplementedError"
                                 for node in ast.walk(tree)), path.name)

    def test_notebook_is_clean_and_code_cells_parse(self):
        path = CODE / "lab_day2_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 40)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                                     if not line.startswith("%"))
                ast.parse(source)
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        for required in ("FINAL CONFIG LOCK", "save_test_predictions", "eval.py", "images.zip",
                         "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"):
            project_text = text + "\n" + (CODE / "workflow.py").read_text(encoding="utf-8")
            self.assertIn(required, project_text)

    def test_results_workbook_has_required_sheets_and_pending_status(self):
        with zipfile.ZipFile(SUBMISSION / "results.xlsx") as archive:
            workbook_xml = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheetnames = {sheet.attrib["name"] for sheet in workbook_xml.findall("main:sheets/main:sheet", namespace)}
        self.assertEqual(sheetnames,
                         {"Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"})
        self.assertIn("PENDING", (SUBMISSION / "report.md").read_text(encoding="utf-8"))

    def test_eval_is_identical_to_official_file(self):
        self.assertEqual((ROOT / "eval.py").read_bytes(), (CODE / "eval.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
