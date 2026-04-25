from vertex_voyage.command_executor import command_executor_main
from vertex_voyage.cli import Commands, setup_logging
if __name__ == "__main__":
    setup_logging()
    command_executor_main(Commands)