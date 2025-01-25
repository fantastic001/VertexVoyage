import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from io import StringIO
from vertex_voyage.storage.graph import ConsistentHashing, ShardMountManager, GraphStorageWithConsistentHashing

class TestConsistentHashing(unittest.TestCase):
    def setUp(self):
        self.shards = ["shard1", "shard2", "shard3"]
        self.hashing = ConsistentHashing(self.shards, replicas=3)

    def test_get_shard(self):
        shard = self.hashing.get_shard("graph1:v1")
        self.assertIn(shard, self.shards)

    def test_consistent_hashing_distribution(self):
        keys = [f"key{i}" for i in range(100)]
        shard_counts = {shard: 0 for shard in self.shards}

        for key in keys:
            shard = self.hashing.get_shard(key)
            shard_counts[shard] += 1

        # Check that keys are distributed across shards
        for shard, count in shard_counts.items():
            self.assertGreater(count, 0)


class TestShardMountManager(unittest.TestCase):
    def setUp(self):
        self.shard_mapping = {
            "shard1": {"hostname": "nfs1.example.com", "protocol": "nfs", "port": 2049, "remote_path": "/exports/shard1"},
            "shard2": {"hostname": "nfs2.example.com", "protocol": "nfs", "port": 2049, "remote_path": "/exports/shard2"},
            "shard3": {"hostname": "local", "protocol": "local", "remote_path": "/local/shard3"}
        }
        self.local_mount_dir = "test_mounts"
        self.manager = ShardMountManager(self.shard_mapping, self.local_mount_dir)

    @patch("os.makedirs")
    @patch("subprocess.run")
    def test_mount_remote_shard(self, mock_run, mock_makedirs):
        mock_run.return_value = MagicMock()
        self.manager.mount("shard1")
        mock_run.assert_called_once_with(
            ["mount", "-t", "nfs", "nfs1.example.com:/exports/shard1", os.path.join(self.local_mount_dir, "shard1")],
            check=True
        )

    @patch("os.makedirs")
    def test_mount_local_shard(self, mock_makedirs):
        with patch("os.path.exists", return_value=True):
            self.manager.mount("shard3")
            mock_makedirs.assert_called_once_with(os.path.join(self.local_mount_dir, "shard3"), exist_ok=True)

    @patch("os.path.ismount", return_value=True)
    def test_is_mounted(self, mock_ismount):
        result = self.manager.is_mounted("/path/to/mount")
        self.assertTrue(result)
        mock_ismount.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_is_not_mounted(self, mock_file):
        result = self.manager.is_mounted("/path/to/mount")
        self.assertFalse(result)


class TestGraphStorageWithConsistentHashing(unittest.TestCase):
    def setUp(self):
        self.shard_mapping = {
            "shard1": {"hostname": "nfs1.example.com", "protocol": "nfs", "port": 2049, "remote_path": "/exports/shard1"},
            "shard2": {"hostname": "nfs2.example.com", "protocol": "nfs", "port": 2049, "remote_path": "/exports/shard2"},
            "shard3": {"hostname": "local", "protocol": "local", "remote_path": "/local/shard3"}
        }
        self.shard_names = ["shard1", "shard2", "shard3"]
        self.storage = GraphStorageWithConsistentHashing(
            storage_dir="test_storage",
            shard_names=self.shard_names,
            shard_mapping=self.shard_mapping
        )

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({}))
    def test_add_vertex(self, mock_file, mock_makedirs, mock_exists):
        self.storage.add_vertex("graph1", "v1", {"name": "Vertex1"})

        # Ensure the graph path and file are created
        mock_makedirs.assert_called()
        mock_file.assert_called()

    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({}))
    def test_add_edge(self, mock_file, mock_makedirs, mock_exists):
        self.storage.add_edge("graph1", "v1", "v2", {"weight": 10})

        # Ensure the graph path and file are created
        mock_makedirs.assert_called()
        mock_file.assert_called()

    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"v1": {"name": "Vertex1"}}))
    def test_get_vertex(self, mock_file, mock_makedirs, mock_exists):
        vertex = self.storage.get_vertex("graph1", "v1")
        mock_file.assert_called()
        self.assertEqual(vertex, {"name": "Vertex1"})

    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"v1": [{"to": "v2", "data": {"weight": 10}}]}))
    def test_get_edges(self, mock_file, mock_makedirs, mock_exists):
        edges = self.storage.get_edges("graph1", "v1")
        self.assertEqual(edges, [{"to": "v2", "data": {"weight": 10}}])

    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"v1": {"name": "Vertex1"}}))
    def test_delete_vertex(self, mock_file, mock_makedirs, mock_exists):
        with patch("json.dump") as mock_json_dump:
            self.storage.delete_vertex("graph1", "v1")
            # Check that json.dump was called to update the files
            mock_json_dump.assert_called()


if __name__ == "__main__":
    unittest.main()
