
from vertex_voyage.command_executor import * 


from vertex_voyage.cli import Commands, setup_logging
import vertex_voyage.config as cfg

def main():
    setup_logging()
    command_executor_main(Commands)
if __name__ == '__main__':
    main()