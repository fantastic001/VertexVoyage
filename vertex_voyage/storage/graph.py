import os
import json
import hashlib
import bisect
import subprocess

from vertex_voyage.config import get_config_str

class ConsistentHashing:
    def __init__(self, shards, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        self._initialize_ring(shards)

    def _hash(self, key: str):
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def _initialize_ring(self, shards):
        """Initialize the consistent hashing ring with replicas."""
        for shard in shards:
            for replica in range(self.replicas):
                key = f"{shard}:{replica}"
                hashed_key = self._hash(key)
                self.ring[hashed_key] = shard
                self.sorted_keys.append(hashed_key)
        self.sorted_keys.sort()

    def get_shard(self, key):
        """Get the shard for a given key."""
        hashed_key = self._hash(key)
        index = bisect.bisect(self.sorted_keys, hashed_key)
        if index == len(self.sorted_keys):  # Wrap around the ring
            index = 0
        return self.ring[self.sorted_keys[index]]

    def add_shard(self, shard):
        for replica in range(self.replicas):
            key = f"{shard}:{replica}"
            hashed_key = self._hash(key)
            self.ring[hashed_key] = shard
            bisect.insort(self.sorted_keys, hashed_key)

    def remove_shard(self, shard):
        for replica in range(self.replicas):
            key = f"{shard}:{replica}"
            hashed_key = self._hash(key)
            self.ring.pop(hashed_key)
            self.sorted_keys.remove(hashed_key)
        self.sorted_keys.sort()


class ShardMountManager:
    """Manages mounting and access to shards stored on remote or local filesystems."""
    def __init__(self, shard_mapping, local_mount_dir):
        """
        shard_mapping: A dictionary mapping shard names to their remote details.
        local_mount_dir: The local base directory where shards will be mounted.
        """
        self.shard_mapping = shard_mapping
        self.local_mount_dir = local_mount_dir
        os.makedirs(local_mount_dir, exist_ok=True)

    def mount(self, shard_name):
        """Mount the shard if not already mounted."""
        if shard_name not in self.shard_mapping:
            raise ValueError(f"Shard {shard_name} is not defined in the mapping.")

        shard_info = self.shard_mapping[shard_name]
        local_mount_path = os.path.join(self.local_mount_dir, shard_name)

        # Ensure local mount directory exists
        os.makedirs(local_mount_path, exist_ok=True)

        if shard_info["protocol"] == "local":
            # Local shard, no need to mount
            if not os.path.exists(shard_info["remote_path"]):
                raise ValueError(f"Local path {shard_info['remote_path']} does not exist.")
            return

        # Check if already mounted
        if self.is_mounted(local_mount_path):
            return
        if shard_info["protocol"] == "nfs":    
            # Mount the remote shard
            remote_path = f"{shard_info['hostname']}:{shard_info['remote_path']}"
            mount_cmd = ["mount", "-t", "nfs", remote_path, local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {remote_path} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {remote_path} to {local_mount_path}: {e}")
        elif shard_info["protocol"] == "sshfs":
            # Mount the remote shard using sshfs
            mount_cmd = ["sshfs", f"{shard_info['hostname']}:{shard_info['remote_path']}", local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {shard_info['remote_path']} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {shard_info['remote_path']} to {local_mount_path}: {e}")
        elif shard_info["protocol"] == "smb":
            # Mount the remote shard using smbclient
            mount_cmd = ["smbclient", f"{shard_info['hostname']}:{shard_info['remote_path']}", local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {shard_info['remote_path']} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {shard_info['remote_path']} to {local_mount_path}: {e}")
        elif shard_info["protocol"] == "ftp":
            # Mount the remote shard using curlftpfs
            mount_cmd = ["curlftpfs", f"{shard_info['hostname']}:{shard_info['remote_path']}", local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {shard_info['remote_path']} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {shard_info['remote_path']} to {local_mount_path}: {e}")
        elif shard_info["protocol"] == "cifs":
            # Mount the remote shard using mount.cifs
            mount_cmd = ["mount.cifs", f"{shard_info['hostname']}:{shard_info['remote_path']}", local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {shard_info['remote_path']} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {shard_info['remote_path']} to {local_mount_path}: {e}")
        elif shard_info["protocol"] == "s3fs":
            # Mount the remote shard using s3fs
            mount_cmd = ["s3fs", f"{shard_info['hostname']}:{shard_info['remote_path']}", local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {shard_info['remote_path']} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {shard_info['remote_path']} to {local_mount_path}: {e}")
        elif shard_info["protocol"] == "hdfs":
            # Mount the remote shard using fuse-dfs
            mount_cmd = ["fuse-dfs", f"{shard_info['hostname']}:{shard_info['remote_path']}", local_mount_path]

            try:
                subprocess.run(mount_cmd, check=True)
                print(f"Mounted {shard_info['remote_path']} to {local_mount_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to mount {shard_info['remote_path']} to {local_mount_path}: {e}")

    def is_mounted(self, local_mount_path):
        return os.path.ismount(local_mount_path)

    def open(self, shard_name, relative_path, mode="r"):
        """
        Open a file within the shard's mount point.
        :param shard_name: The name of the shard.
        :param relative_path: The path of the file relative to the shard's root directory.
        :param mode: The mode in which to open the file (default is 'r').
        """
        if shard_name not in self.shard_mapping:
            raise ValueError(f"Shard {shard_name} is not defined in the mapping.")

        local_mount_path = os.path.join(self.local_mount_dir, shard_name)

        # Ensure the shard is mounted
        self.mount(shard_name)

        # Open the file
        file_path = os.path.join(local_mount_path, relative_path)
        return open(file_path, mode)

    def add_shard(self, shard_name, shard_info):
        self.shard_mapping[shard_name] = shard_info
        self.mount(shard_name)

    def remove_shard(self, shard_name):
        if shard_name not in self.shard_mapping:
            raise ValueError(f"Shard {shard_name} is not defined in the mapping.")

        local_mount_path = os.path.join(self.local_mount_dir, shard_name)
        if self.is_mounted(local_mount_path):
            subprocess.run(["umount", local_mount_path], check=True)
            print(f"Unmounted {local_mount_path}")
        self.shard_mapping.pop(shard_name)

class GraphStorageWithConsistentHashing:
    """Manages graph storage across shards using consistent hashing."""
    def __init__(self, storage_dir, shard_names, shard_mapping, replicas=3):
        self.mount_manager = ShardMountManager(shard_mapping, storage_dir)
        self.hashing = ConsistentHashing(shard_names, replicas)

    def _get_graph_path(self, graph_name, key):
        """Get the graph path for a given key."""
        shard = self.hashing.get_shard(key)
        shard_mount_path = os.path.join(self.mount_manager.local_mount_dir, shard)
        return os.path.join(shard_mount_path, graph_name)

    def _ensure_graph_files(self, graph_path):
        os.makedirs(graph_path, exist_ok=True)
        vertices_file = os.path.join(graph_path, "vertices.json")
        edges_file = os.path.join(graph_path, "edges.json")

        if not os.path.exists(vertices_file):
            with open(vertices_file, "w") as f:
                json.dump({}, f)
        if not os.path.exists(edges_file):
            with open(edges_file, "w") as f:
                json.dump({}, f)

    def add_vertex(self, graph_name, vertex_id, data):
        key = f"{graph_name}:{vertex_id}"
        graph_path = self._get_graph_path(graph_name, key)
        self._ensure_graph_files(graph_path)

        vertices_file = os.path.join(graph_path, "vertices.json")
        with open(vertices_file, "r") as f:
            vertices = json.load(f)
            vertices[vertex_id] = {
                "data": data,
                "graph": graph_name,
                "id": vertex_id,
                "key": key
            }
            json.dump(vertices, f)

    def add_edge(self, graph_name, from_vertex, to_vertex, data):
        key = f"{graph_name}:{from_vertex}"
        graph_path = self._get_graph_path(graph_name, key)
        self._ensure_graph_files(graph_path)

        edges_file = os.path.join(graph_path, "edges.json")
        with open(edges_file, "r") as f:
            edges = json.load(f)

        if from_vertex not in edges:
            edges[from_vertex] = []
        edges[from_vertex].append({"to": to_vertex, "data": data, "graph": graph_name, "from": from_vertex, "key": key})

        with open(edges_file, "w") as f:
            json.dump(edges, f)

    def get_vertex(self, graph_name, vertex_id):
        key = f"{graph_name}:{vertex_id}"
        graph_path = self._get_graph_path(graph_name, key)
        vertices_file = os.path.join(graph_path, "vertices.json")

        if not os.path.exists(vertices_file):
            return None

        with open(vertices_file, "r") as f:
            vertices = json.load(f)

        return vertices.get(vertex_id, None)

    def get_edges(self, graph_name, vertex_id):
        key = f"{graph_name}:{vertex_id}"
        graph_path = self._get_graph_path(graph_name, key)
        edges_file = os.path.join(graph_path, "edges.json")

        if not os.path.exists(edges_file):
            return []

        with open(edges_file, "r") as f:
            edges = json.load(f)

        return edges.get(vertex_id, [])

    def delete_vertex(self, graph_name, vertex_id):
        key = f"{graph_name}:{vertex_id}"
        graph_path = self._get_graph_path(graph_name, key)
        vertices_file = os.path.join(graph_path, "vertices.json")
        edges_file = os.path.join(graph_path, "edges.json")

        if not os.path.exists(vertices_file) or not os.path.exists(edges_file):
            return

        with open(vertices_file, "r") as f:
            vertices: dict = json.load(f)

        with open(edges_file, "r") as f:
            edges: dict = json.load(f)

        vertices.pop(vertex_id, None)
        edges.pop(vertex_id, None)

        with open(vertices_file, "w") as f:
            json.dump(vertices, f)

        with open(edges_file, "w") as f:
            json.dump(edges, f)
    
    def add_shard(self, shard_name, shard_info):
        self.mount_manager.add_shard(shard_name, shard_info)
        self.hashing.add_shard(shard_name)
        # move vertices from shard before this newly added shard to this shard if they belong here
        for root, dirs, files in os.walk(self.mount_manager.local_mount_dir):
            for file in files:
                if file == "vertices.json":
                    with open(os.path.join(root, file), "r") as f:
                        vertices = json.load(f)
                        for vertex_id in vertices:
                            if self.hashing.get_shard(vertices[vertex_id]["key"]) == shard_name:
                                self.add_vertex(vertices[vertex_id]["graph"], vertices[vertex_id]["id"], vertices[vertex_id]["data"])
                                del vertices[vertex_id]
                elif file == "edges.json":
                    with open(os.path.join(root, file), "r") as f:
                        edges = json.load(f)
                        for from_vertex in edges:
                            for edge in edges[from_vertex]:
                                if self.hashing.get_shard(edge["key"]) == shard_name:
                                    self.add_edge(edge["graph"], edge["from"], edge["to"], edge["data"])
                                    del edges[from_vertex]
    
    def remove_shard(self, shard_name):
        self.hashing.remove_shard(shard_name)
        # move data from this shard to other shards
        shard_root = os.path.join(self.mount_manager.local_mount_dir, shard_name)
        # read vertices 
        for root, dirs, files in os.walk(shard_root):
            for file in files:
                if file == "vertices.json":
                    with open(os.path.join(root, file), "r") as f:
                        vertices = json.load(f)
                        for vertex_id in vertices:
                            self.add_vertex(vertices[vertex_id]["graph"], vertices[vertex_id]["id"], vertices[vertex_id]["data"])
                elif file == "edges.json":
                    with open(os.path.join(root, file), "r") as f:
                        edges = json.load(f)
                        for from_vertex in edges:
                            for edge in edges[from_vertex]:
                                self.add_edge(edge["graph"], edge["from"], edge["to"], edge["data"])
        self.mount_manager.remove_shard(shard_name)

# Example Usage
if __name__ == "__main__":
    shard_mapping = {
        "shard1": {"hostname": "nfs1.example.com", "protocol": "nfs", "port": 2049, "remote_path": "/exports/shard1"},
        "shard2": {"hostname": "nfs2.example.com", "protocol": "nfs", "port": 2049, "remote_path": "/exports/shard2"},
        "shard3": {"hostname": "local", "protocol": "local", "remote_path": "/local/shard3"}
    }

    shard_names = ["shard1", "shard2", "shard3"]
    storage = GraphStorageWithConsistentHashing(
        storage_dir="mounted_storage",
        shard_names=shard_names,
        shard_mapping=shard_mapping
    )

    # Add vertices
    storage.add_vertex("graph1", "v1", {"name": "Vertex1"})
    storage.add_vertex("graph1", "v2", {"name": "Vertex2"})

    # Add edges
    storage.add_edge("graph1", "v1", "v2", {"weight": 10})

    # Retrieve data
    print(storage.get_vertex("graph1", "v1"))  # Output: {'name': 'Vertex1'}
    print(storage.get_edges("graph1", "v1"))  # Output: [{'to': 'v2', 'data': {'weight': 10}}]
