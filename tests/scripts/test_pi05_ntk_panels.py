"""Preserve measured coordinates and native PPT diagram objects during panel replacement."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from lerobot.scripts.replot_pi05_ntk_panels import NS, REL_NS, common_limits, fill_pptx


class TestNtkPanelExport(unittest.TestCase):
    def test_shared_limits_include_all_points_and_keep_log_energy(self):
        values = {
            "vlm": np.array([[[2.0, 1e-7]], [[5.0, 2e-7]]]),
            "action": np.array([[[3.0, 1e-5]], [[7.0, 2e-5]]]),
        }
        xlim, ylim, logarithmic = common_limits(values)
        self.assertLess(xlim[0], 2)
        self.assertGreater(xlim[1], 7)
        self.assertLess(ylim[0], 1e-7)
        self.assertGreater(ylim[1], 2e-5)
        self.assertTrue(logarithmic)

    def test_zero_energy_uses_linear_axis(self):
        _, ylim, logarithmic = common_limits({"vlm": np.array([[[0.0, 0.0]]])})
        self.assertFalse(logarithmic)
        self.assertEqual(ylim[0], 0)
        self.assertGreater(ylim[1], 0)

    def test_template_cannot_relabel_a_3000_step_result_as_4000(self):
        with self.assertRaisesRegex(ValueError, "do not relabel"):
            fill_pptx("unused", "unused", {}, [0, 750, 1000, 3000])

    def test_replace_shared_image_slots_preserves_native_loss_curve(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template, output = root / "template.pptx", root / "filled.pptx"
            curve = '<p:sp><p:nvSpPr><p:cNvPr id="42" name="SCHEMATIC_LOSS_FLOW"/></p:nvSpPr><p:spPr><a:custGeom><a:pathLst><a:path w="100" h="100"><a:moveTo><a:pt x="0" y="0"/></a:moveTo><a:lnTo><a:pt x="100" y="80"/></a:lnTo></a:path></a:pathLst></a:custGeom></p:spPr></p:sp>'
            pictures, panels = [], {}
            for index, stage in enumerate(("before", "priming")):
                pictures.append(
                    f'<p:pic><p:nvPicPr><p:cNvPr id="{index + 1}" descr="NTK_SLOT:backbone:{stage}"/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"/></p:blipFill></p:pic>'
                )
                path = root / f"{stage}.png"
                path.write_bytes(f"distinct-{stage}-bytes".encode())
                panels[("backbone", stage)] = path
            pending = (
                '<p:sp><p:nvSpPr><p:cNvPr id="43" name="NTK_PENDING_backbone_before"/></p:nvSpPr></p:sp>'
            )
            slide = f'<p:sld xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}"><p:cSld><p:spTree>{curve}{pending}{"".join(pictures)}</p:spTree></p:cSld></p:sld>'
            rels = f'<Relationships xmlns="{REL_NS}"><Relationship Id="rId1" Target="../media/shared.png" Type="{NS["r"]}/image"/></Relationships>'
            with zipfile.ZipFile(template, "w") as package:
                package.writestr("ppt/slides/slide1.xml", slide)
                package.writestr("ppt/slides/_rels/slide1.xml.rels", rels)
                package.writestr("ppt/media/shared.png", b"original-placeholder")
            original = template.read_bytes()
            self.assertEqual(fill_pptx(template, output, panels, [0, 750, 1000, 4000]), set(panels))
            self.assertEqual(template.read_bytes(), original)
            with zipfile.ZipFile(output) as package:
                result = ET.fromstring(package.read("ppt/slides/slide1.xml"))
                paths = result.findall(".//a:custGeom/a:pathLst/a:path", NS)
                self.assertEqual(len(paths), 1)
                self.assertEqual(paths[0].find("a:lnTo/a:pt", NS).attrib, {"x": "100", "y": "80"})
                self.assertFalse(
                    any(
                        node.get("name", "").startswith("NTK_PENDING_")
                        for node in result.findall(".//p:cNvPr", NS)
                    )
                )
                ids = [node.get(f"{{{NS['r']}}}embed") for node in result.findall(".//a:blip", NS)]
                self.assertEqual(len(set(ids)), 2)
                for (_, stage), path in panels.items():
                    self.assertEqual(package.read(f"ppt/media/ntk_backbone_{stage}.png"), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
