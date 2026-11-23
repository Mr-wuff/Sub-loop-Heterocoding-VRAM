import hashlib
import io
import json
import unittest
import zipfile
from pathlib import Path
from shv_preview.cli import load_evidence,verify_archive

class PreviewTest(unittest.TestCase):
    def test_evidence_rows(self):
        self.assertGreaterEqual(len(load_evidence()),10)
        self.assertTrue(any(row['round']=='R62' for row in load_evidence()))
    def test_checksum_and_crc(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p=Path(d)/'sample.zip';s=Path(d)/'sample.zip.sha256'
            with zipfile.ZipFile(p,'w') as z:z.writestr('a','abc')
            s.write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  sample.zip\n')
            self.assertTrue(verify_archive(str(p),str(s)))
            s.write_text('0'*64+'  sample.zip\n')
            self.assertFalse(verify_archive(str(p),str(s)))
    def test_manifest_schema(self):
        p=Path(__file__).resolve().parents[1]/'benchmarks/EVIDENCE_MANIFEST.json'
        j=json.loads(p.read_text())
        self.assertEqual(len(j['records']),7)
    def test_artifact_mirror(self):
        root=Path(__file__).resolve().parents[1]/'artifacts'
        j=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(j['artifacts']),7)
        for row in j['artifacts']:
            f=root/row['path']
            self.assertTrue(f.exists())
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),row['sha256'])
            with zipfile.ZipFile(f) as z:
                self.assertIsNone(z.testzip())
if __name__=='__main__':unittest.main()
