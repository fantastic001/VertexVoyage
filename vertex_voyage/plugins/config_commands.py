
import vertex_voyage.config as cfg

class ConfigurationCommands:
    def get_config_location(self):
        return cfg.get_config_location()
    
    def get_config(self, key: str):
        value = cfg.get_config(key, None, "")
        if value is None:
            return "" 
        else:
            return value
    
    def set_config(self, key: str, value: str):
        return cfg.set_config(key, value)


def register_commands(ctrl):
    ctrl.add_command_class(ConfigurationCommands)