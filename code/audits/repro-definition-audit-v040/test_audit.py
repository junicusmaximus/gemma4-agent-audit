"""Focused CPU fixtures; never execute model or serialized tool-call contents."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit

ROOT = Path(__file__).resolve().parents[2]


def call(content, name="write_file"):
    return '<|tool_call>call:' + name + '{filepath:<|"|>repro.py<|"|>,content:<|"|>' + content + '<|"|>}<tool_call|>'


class HarmonizedDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = audit.ParserAdapter(ROOT)

    def test_early_reproduce_matches_both_definitions(self):
        text = call(audit.HEADER + '\n# reproduce the failure\n')
        self.assertTrue(audit.old_raw_flag(text))
        result = self.parser.parse(text)
        self.assertEqual(result['status'], 'parsed')
        self.assertTrue(result['any_repro'])

    def test_keyword_after_200_content_characters_is_raw_only(self):
        content = audit.HEADER + 'x' * (205 - len(audit.HEADER)) + 'reproduce'
        self.assertTrue(audit.old_raw_flag(call(content)))
        self.assertFalse(self.parser.parse(call(content))['any_repro'])

    def test_reproduction_stem_does_not_equal_reproduce(self):
        text = call(audit.HEADER + '\n# reproduction script')
        self.assertTrue(audit.old_raw_flag(text))
        self.assertFalse(self.parser.parse(text)['any_repro'])

    def test_keyword_outside_both_windows_is_negative(self):
        text = call(audit.HEADER + 'x' * 310 + 'reproduce')
        self.assertFalse(audit.old_raw_flag(text))
        self.assertFalse(self.parser.parse(text)['any_repro'])

    def test_nonwrite_file_is_not_repro_script(self):
        result = self.parser.parse(call(audit.HEADER + '\n# reproduce', name='read_file'))
        self.assertEqual(result['status'], 'parsed')
        self.assertFalse(result['any_repro'])

    def test_quoted_backslash_and_newline_content_is_preserved(self):
        content = audit.HEADER + '\n# reproduce "quoted" \\path\nprint("hi")'
        result = self.parser.parse(call(content))
        self.assertEqual(result['calls'][0]['arguments']['content'], content)
        self.assertTrue(result['any_repro'])

    def test_captured_escaped_json_matches_gemma_parsed_arguments(self):
        content = audit.HEADER + '\n# reproduce "quoted" \\path\n'
        expected = {'filepath': 'repro.py', 'content': content}
        message = {'tool_calls': [{'function': {'name': 'write_file', 'arguments': json.dumps(expected)}}]}
        captured = audit.captured_calls(message)
        self.assertEqual(captured, self.parser.parse(call(content))['calls'])

    def test_plain_answer_is_known_no_action(self):
        result = self.parser.parse('No code change is needed.<turn|>')
        self.assertEqual((result['status'], result['any_repro']), ('no_action', False))

    def test_unclosed_tool_call_is_explicitly_unknown(self):
        result = self.parser.parse('<|tool_call>call:write_file{content:<|"|>unfinished')
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['any_repro'])

    def test_tool_like_text_inside_reasoning_is_not_an_action(self):
        text = '<|channel>thought\nExample: ' + call(audit.HEADER + '\n# reproduce') + '<channel|>' + call('source.py', 'read_file')
        self.assertTrue(audit.old_raw_flag(text))
        result = self.parser.parse(text)
        self.assertEqual([c['function_name'] for c in result['calls']], ['read_file'])
        self.assertFalse(result['any_repro'])

    def test_prompt_side_open_reasoning_without_close_is_unknown(self):
        result = self.parser.parse('still reasoning', '<|turn>model\n<|channel>thought\n')
        self.assertEqual((result['status'], result['reason']), ('unknown', 'unclosed_reasoning'))

    def test_multiple_actions_keep_first_separate_from_anywhere(self):
        text = call('source.py', 'read_file') + call(audit.HEADER + '\n# reproduce')
        result = self.parser.parse(text)
        self.assertFalse(result['first_repro'])
        self.assertTrue(result['any_repro'])

    def test_parser_timeout_is_unknown_instead_of_negative(self):
        with mock.patch.object(self.parser, 'parse_arguments', side_effect=TimeoutError):
            result = self.parser.parse(call(audit.HEADER))
        self.assertEqual((result['status'], result['reason']), ('unknown', 'argument_parser_deadline'))
        self.assertIsNone(result['any_repro'])

    def test_unknown_before_known_action_keeps_first_action_unknown(self):
        results = [dict(status='unknown'), dict(status='parsed', first_repro=True)]
        self.assertIsNone(audit.first_parsed_action(results))

    def test_known_no_action_can_precede_first_parsed_action(self):
        results = [dict(status='no_action'), dict(status='parsed', first_repro=True)]
        self.assertTrue(audit.first_parsed_action(results))

    def test_unknown_values_are_not_zero_filled(self):
        result = audit.tally([True, False, None])
        self.assertEqual((result['positive'], result['negative'], result['unknown']), (1, 1, 1))
        self.assertEqual(result['known_denominator'], 2)
        self.assertEqual(result['share_among_known'], .5)

    def test_inference_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = 'results/v025_a1_e4b_s1/traces/trace_synthetic.json'
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text('{"steps": []}')
            record = dict(run='v025_a1_e4b_s1', trace_missing=False, instance_id='synthetic', opener='no_tool_call')
            with self.assertRaisesRegex(ValueError, 'Historical inference trace hash mismatch'):
                audit.verify_inference_labels(root, [record], self.parser.metrics, {relative: '0' * 64})

    def test_inference_label_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = 'results/v025_a1_e4b_s1/traces/trace_synthetic.json'
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text('{"steps": []}')
            record = dict(run='v025_a1_e4b_s1', trace_missing=False, instance_id='synthetic', opener='repro_script_opener')
            with self.assertRaisesRegex(ValueError, 'Stored inference opener label differs'):
                audit.verify_inference_labels(root, [record], self.parser.metrics, {relative: audit.sha(path)})


if __name__ == '__main__':
    unittest.main()
