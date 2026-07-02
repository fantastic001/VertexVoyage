
from vertex_voyage.command_executor import * 


from vertex_voyage.cli import Commands, setup_logging
import vertex_voyage.cli 
import vertex_voyage.config as cfg


def main():
    setup_logging()
    executors = cfg.get_classes_inheriting(vertex_voyage.cli.CustomCLICommandExecutor)
    print("Executors:", executors)
    command_executor_main([Commands] + executors)
if __name__ == '__main__':
    main()