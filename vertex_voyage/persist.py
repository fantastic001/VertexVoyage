import pickle 
import json
import os
import logging
logger = logging.getLogger(__name__)

class PersistedRun:
    """
    A class to represent a persisted run of the vertex voyage algorithm. It contains all the necessary information to resume the run at a later time.

    In constructor, it takes the location of directory and parameters of the run. It checks if directory exists, if not it creates it. It also checks if there is a file named 'metadata.json' and compares the parameters with the ones in the file. If they are different, it raises an error. If they are the same, it loads can use the existing files to 
    resume the run. If there is no file named 'metadata.json', it creates one and saves the parameters in it.

    If directory is None or empty string, it will not persist anything and will only keep the data in memory. This can be useful for testing or for runs that do not need to be persisted.

    Example usage:
    ```python
    from vertex_voyage.persist import PersistedRun
    p = PersistedRun("./runs/run1", epochs=10, batch_size=32)

    if "n2v" in p:
        n2v = p["n2v"]
    else:
        n2v = p("n2v", Node2Vec, dimensions=128, walk_length=80, num_walks=10, workers=4)
        # train the model
        n2v.train()
        # save the model
        p["n2v"] = n2v
    
    # Alternatively, you can load saved model if it exists, otherwise create a new one and save it:
    n2v = p("n2v", Node2Vec, dimensions=128, walk_length=80, num_walks=10, workers=4)
    n2v.train()
    p["n2v"] = n2v

    # Or, use context manager to automatically save the model at the end of the block:
    with p("n2v", Node2Vec, dimensions=128, walk_length=80, num_walks=10, workers=4) as n2v:
        n2v.train()
    ```
    """
    def load(directory: str) -> "PersistedRun":
        """
        Loads a persisted run from the given directory. It checks if the directory exists and if there is a file named 'metadata.json'. If the directory does not exist or if there is no 'metadata.json' file, it raises an error.

        In case of success, it returns a PersistedRun object where parameters are loaded from the 'metadata.json' file and data is loaded from the files in the directory.

        Args:
            directory (str): The path to the directory where the persisted run is stored.
        """
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist.")
            raise ValueError(f"Directory {directory} does not exist.")
        metadata_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.error(f"No metadata.json file found in directory {directory}.")
            raise ValueError(f"No metadata.json file found in directory {directory}.")
        with open(metadata_path, "r") as f:
            params = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}: {params}")
        return PersistedRun(directory, **params)

    def __init__(self, directory, **params):
        logger.info(f"Initializing PersistedRun with directory: {directory} and params: {params}")
        self.directory = directory
        self.params = params
        self.data = {}
        # check if directory exists, if not create it
        if self.directory is None or self.directory == "":
            logger.info("No directory provided, using in-memory storage only.")
            return
        if not os.path.exists(directory):
            os.makedirs(directory)
        # check if there is a file named 'metadata.json' and compare the parameters with the ones in the file
        metadata_path = os.path.join(directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}: {metadata}")
            if metadata != params:
                logger.error(f"Parameters do not match with the ones in the metadata file. Metadata: {metadata}, Params: {params}")
                raise ValueError("Parameters do not match with the ones in the metadata file.")
        else:
            with open(metadata_path, "w") as f:
                logger.info(f"Creating metadata file at {metadata_path} with params: {params}")
                json.dump(params, f)
    def __getitem__(self, key):
        logger.info(f"Getting item with key: {key}")
        if key in self.data:
            logger.info(f"Key {key} found in memory, returning value.")
            return self.data[key]
        else:
            if self.directory is None or self.directory == "":
                logger.error(f"Key {key} not found in memory and no directory provided for persistence.")
                raise KeyError(f"{key} not found in the persisted run.")
            file_path = os.path.join(self.directory, f"{key}.pkl")
            logger.info(f"Key {key} not found in memory, checking for file at {file_path}.")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    logger.info(f"File {file_path} found, loading value.")
                    self.data[key] = pickle.load(f)
                return self.data[key]
            else:
                logger.error(f"Key {key} not found in the persisted run.")
                raise KeyError(f"{key} not found in the persisted run.")
    def __setitem__(self, key, value):
        logger.info(f"Setting item with key: {key}")
        self.data[key] = value
        if self.directory is None or self.directory == "":
            logger.info(f"No directory provided, not persisting key {key}.")
            return
        file_path = os.path.join(self.directory, f"{key}.pkl")
        with open(file_path, "wb") as f:
            logger.info(f"Saving value for key {key} to file {file_path}.")
            pickle.dump(value, f)
    def __call__(self, key, cls, *args, **kwargs):
        logger.info(f"Calling PersistedRun with key: {key}, class: {cls}, args: {args}, kwargs: {kwargs}")
        if key in self.data:
            logger.info(f"Key {key} found in memory, returning value.")
            return self.data[key]
        else:
            if self.directory is None or self.directory == "":
                logger.info(f"No directory provided, creating new instance of {cls} with args: {args} and kwargs: {kwargs}.")
                instance = cls(*args, **kwargs)
                self.data[key] = instance
                return instance
            file_path = os.path.join(self.directory, f"{key}.pkl")
            logger.info(f"Key {key} not found in memory, checking for file at {file_path}.")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    logger.info(f"File {file_path} found, loading value.")
                    self.data[key] = pickle.load(f)
                return self.data[key]
            else:
                logger.info(f"File {file_path} not found, creating new instance of {cls} with args: {args} and kwargs: {kwargs}.")
                instance = cls(*args, **kwargs)
                self.data[key] = instance
                with open(file_path, "wb") as f:
                    logger.info(f"File {file_path} not found, creating new instance and saving value.")
                    pickle.dump(instance, f)
                return instance
    def __enter__(self):
        logger.info("Entering context manager for PersistedRun.")
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Exiting context manager for PersistedRun.")
        if self.directory is None or self.directory == "":
            logger.info("No directory provided, not persisting any data on exit.")
            return
        for key, value in self.data.items():
            file_path = os.path.join(self.directory, f"{key}.pkl")
            logger.info(f"Saving value for key {key} to file {file_path} on exit.")
            with open(file_path, "wb") as f:
                pickle.dump(value, f)
    def __contains__(self, key):
        logger.info(f"Checking if key {key} exists.")
        if key in self.data:
            logger.info(f"Key {key} found in memory.")
            return True
        else:
            if self.directory is None or self.directory == "":
                logger.info(f"No directory provided, key {key} not found in memory.")
                return False
            file_path = os.path.join(self.directory, f"{key}.pkl")
            exists = os.path.exists(file_path)
            if exists:
                logger.info(f"Key {key} found in file {file_path}.")
            else:
                logger.info(f"Key {key} not found.")
            return exists

    def __repr__(self):
        return f"PersistedRun(directory={self.directory}, params={self.params}, data_keys={list(self.data.keys())})"
    def __str__(self):
        return self.__repr__()
    def __del__(self):
        logger.info("Deleting PersistedRun instance, saving all data to files.")
        if self.directory is None or self.directory == "":
            logger.info("No directory provided, not persisting any data on delete.")
            return
        for key, value in self.data.items():
            file_path = os.path.join(self.directory, f"{key}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(value, f)
    
    def clear(self):
        logger.info("Clearing all data from PersistedRun.")
        self.data = {}
        if self.directory is None or self.directory == "":
            logger.info("No directory provided, not clearing any files.")
            return
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                os.remove(os.path.join(self.directory, file))
    
    def delete(self, key):
        logger.info(f"Deleting key {key} from PersistedRun.")
        if key in self.data:
            del self.data[key]
        if self.directory is None or self.directory == "":
            logger.info("No directory provided, not deleting any files.")
            return
        file_path = os.path.join(self.directory, f"{key}.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)
    
