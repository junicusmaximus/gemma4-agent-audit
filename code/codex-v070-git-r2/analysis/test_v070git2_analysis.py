"""Pure analysis fixtures; no evaluation, PIN, result reads or model server."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_external_e4b_v070git2 as analysis


class AnalysisTests(unittest.TestCase):
    def test_frozen_thresholds(self):
        self.assertEqual(analysis.termination(.30, .80), 'NON-COLLAPSED')
        self.assertEqual(analysis.termination(.40, .70), 'INTERMEDIATE')
        self.assertEqual(analysis.termination(.41, .80), 'COLLAPSED')
        self.assertEqual(analysis.termination(.10, .69), 'COLLAPSED')
        self.assertEqual(analysis.termination(None, .80), 'UNDEFINED')

    def test_comparator_never_reads_old_results_or_bootstraps(self):
        with mock.patch.object(analysis, 'sealed_v068', side_effect=AssertionError('forbidden read')):
            got = analysis.comparator({}, [], [], [], object(), {})
        self.assertFalse(got['shown'])
        self.assertTrue(got['withheld'])
        self.assertNotIn('arms', got)
        self.assertNotIn('source', got)

    def test_direct_old_result_read_is_forbidden(self):
        with self.assertRaisesRegex(RuntimeError, 'forbidden'):
            analysis.sealed_v068({})

    def test_e1_primary_form_unchanged(self):
        labels = {'A_onpol_ts1': 'NON-COLLAPSED', 'A_onpol_ts2': 'NON-COLLAPSED', 'T_sub': 'COLLAPSED'}
        rates = {'A_onpol_ts1': (.1, .9), 'A_onpol_ts2': (.2, .85), 'T_sub': (.5, .6)}
        got = analysis.cell(labels, rates, 12)
        self.assertEqual(got['cell'], 'E1')
        self.assertEqual(got['tasks'], 12)

    def test_e2_primary_form_unchanged(self):
        labels = {'A_onpol_ts1': 'NON-COLLAPSED', 'A_onpol_ts2': 'NON-COLLAPSED', 'T_sub': 'INTERMEDIATE'}
        rates = {'A_onpol_ts1': (.1, .9), 'A_onpol_ts2': (.2, .85), 'T_sub': (.35, .75)}
        self.assertEqual(analysis.cell(labels, rates, 12)['cell'], 'E2')

    def test_e3_primary_form_unchanged(self):
        labels = {'A_onpol_ts1': 'INTERMEDIATE', 'A_onpol_ts2': 'NON-COLLAPSED', 'T_sub': 'COLLAPSED'}
        rates = {'A_onpol_ts1': (.35, .75), 'A_onpol_ts2': (.2, .85), 'T_sub': (.5, .6)}
        self.assertEqual(analysis.cell(labels, rates, 12)['cell'], 'E3')

    def test_partial_panel_cannot_get_cell(self):
        frozen = dict(tasks=[dict(instance_id='t1'), dict(instance_id='t2')], runs=[{}, {}, {}])
        rows = [dict(state='completed', instance_id='t1')]
        got = analysis.readings(frozen, rows, [], dict(stop=None), None, '2026-11-10T08:00:00Z')
        self.assertEqual(got['kind'], 'incomplete')
        self.assertIsNone(got['cell'])

    def test_late_panel_cannot_get_cell(self):
        frozen = dict(tasks=[dict(instance_id='t1')], runs=[{}])
        rows = [dict(state='completed', instance_id='t1')]
        with mock.patch.object(analysis, 'termination_table', return_value={}):
            got = analysis.readings(frozen, rows, ['t1'], dict(stop=None), {}, '2026-11-10T09:00:00Z')
        self.assertEqual(got['kind'], 'late')
        self.assertIsNone(got['cell'])

    def test_output_and_manifests_are_new(self):
        for path in [analysis.OUTPUT, analysis.MANIFEST, analysis.MANIFEST_DRAFT]:
            self.assertTrue(path.is_relative_to(analysis.runner.WORK))

    def test_original_primary_producer_is_unchanged(self):
        self.assertEqual(analysis.runner.sha(analysis.PRODUCER), analysis.PRODUCER_SHA)
        self.assertEqual(analysis.runner.sha(analysis.RESOLUTION), analysis.RESOLUTION_SHA)


if __name__ == '__main__':
    unittest.main()
