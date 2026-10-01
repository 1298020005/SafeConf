"""The train/validation reader must select roles before expression access."""
import unittest
import numpy as np
from tools.scripts.build_safeconf_common_gene_biology import allowed_row_spans


class AllowedBiologyRows(unittest.TestCase):
    def test_interleaved_private_rows_are_never_in_a_span(self):
        mask = np.array([False, True, True, False, True, False, True, True, True, False])
        reads = list(allowed_row_spans(mask, 2))
        self.assertEqual(reads, [(1, 3), (4, 5), (6, 8), (8, 9)])
        materialized = np.concatenate([np.arange(a, b) for a, b in reads])
        np.testing.assert_array_equal(materialized, np.flatnonzero(mask))
        self.assertTrue(all(mask[a:b].all() for a, b in reads))

    def test_no_allowed_rows_produces_no_expression_request(self):
        self.assertEqual(list(allowed_row_spans(np.zeros(5, bool), 2)), [])

    def test_endpoint_and_chunk_boundaries_do_not_drop_cells(self):
        mask = np.array([True, True, True, False, True])
        self.assertEqual(list(allowed_row_spans(mask, 2)), [(0, 2), (2, 3), (4, 5)])


if __name__ == '__main__':
    unittest.main()
