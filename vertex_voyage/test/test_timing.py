import threading
import time
import unittest

from vertex_voyage.timing import TimeMetric


class TestTimeMetric(unittest.TestCase):
    def setUp(self):
        TimeMetric.set_enabled(True)
        TimeMetric.reset()

    def tearDown(self):
        TimeMetric.reset()
        TimeMetric.set_enabled(True)

    def test_requires_name(self):
        with self.assertRaises(ValueError):
            TimeMetric("")
        with self.assertRaises(ValueError):
            TimeMetric("a/b")

    def test_context_manager_records_once(self):
        with TimeMetric("part"):
            time.sleep(0.02)
        data = TimeMetric.dump()
        self.assertIn("part", data)
        self.assertEqual(data["part"]["count"], 1)
        self.assertGreaterEqual(data["part"]["total"], 0.015)

    def test_decorator_aggregates_across_calls(self):
        @TimeMetric("fn")
        def fn(x):
            time.sleep(0.005)
            return x * 2

        results = [fn(i) for i in range(5)]
        self.assertEqual(results, [0, 2, 4, 6, 8])
        rec = TimeMetric.dump()["fn"]
        self.assertEqual(rec["count"], 5)
        self.assertGreater(rec["total"], rec["mean"])
        self.assertGreaterEqual(rec["stddev"], 0.0)
        self.assertLessEqual(rec["min"], rec["max"])

    def test_manual_start_stop(self):
        tm = TimeMetric("manual")
        tm.start()
        time.sleep(0.01)
        tm.stop()
        self.assertEqual(TimeMetric.dump()["manual"]["count"], 1)

    def test_nesting_builds_dotted_paths_and_self_time(self):
        with TimeMetric("outer"):
            time.sleep(0.02)
            with TimeMetric("inner"):
                time.sleep(0.03)

        data = TimeMetric.dump()
        self.assertIn("outer", data)
        self.assertIn("outer/inner", data)
        self.assertGreaterEqual(data["outer/inner"]["total"], 0.025)
        # outer self-time excludes the nested inner span
        self.assertLess(data["outer"]["self_total"], data["outer"]["total"])
        self.assertGreaterEqual(data["outer"]["self_total"], 0.015)

    def test_exception_still_records(self):
        with self.assertRaises(RuntimeError):
            with TimeMetric("boom"):
                raise RuntimeError("x")
        self.assertEqual(TimeMetric.dump()["boom"]["count"], 1)

    def test_disabled_has_no_effect(self):
        TimeMetric.set_enabled(False)
        with TimeMetric("nope"):
            pass
        self.assertEqual(TimeMetric.dump(), {})

    def test_thread_safe_aggregation(self):
        def worker():
            for _ in range(20):
                with TimeMetric("shared"):
                    time.sleep(0.001)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(TimeMetric.dump()["shared"]["count"], 80)

    def test_threads_do_not_cross_nest(self):
        # A parent active on the main thread must not prefix a child on another thread.
        seen = {}

        def worker():
            with TimeMetric("child"):
                time.sleep(0.01)
            seen["paths"] = set(TimeMetric.dump())

        with TimeMetric("parent"):
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        self.assertIn("child", seen["paths"])
        self.assertNotIn("parent/child", TimeMetric.dump())

    def test_report_lines_smoke(self):
        with TimeMetric("x"):
            pass
        lines = TimeMetric.report_lines()
        self.assertTrue(any("x" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
