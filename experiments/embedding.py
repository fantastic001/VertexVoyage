
from vertex_voyage.grid_search import GridSearchPersistence
from vertex_voyage.command_executor import command_executor_main

GS_LOCATION = "gs_cache"

class Commands:
    def list_datasets(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        datasets = set()
        for params, result in gsp.load(threshold=0):
            datasets.add(params['dataset'])
        return list(datasets)
    
    def restore(self):
        gsp = GridSearchPersistence(GS_LOCATION)
        gsp.restore_backups()
    
    

if __name__ == "__main__":
    command_executor_main(Commands)