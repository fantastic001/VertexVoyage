
import vertex_voyage.config as cfg

class PluginManagerCommands:
    def list_plugins(self):
        return cfg.list_plugins()
    def disable_plugin(self, name: str):
        my_plugins = self.list_plugins()
        if name not in my_plugins:
            return "Plugin not found"
        disabled_plugins = cfg.get_disabled_plugins()
        if name in disabled_plugins:
            return "Plugin already disabled"
        disabled_plugins.append(name)
        cfg.set_config('disabled_plugins', disabled_plugins)
        return "Plugin disabled"
    def enable_plugin(self, name: str):
        all_plugins = list(set(self.list_plugins()) | set(cfg.get_disabled_plugins()))
        if name not in all_plugins:
            return "Plugin not found"
        disabled_plugins = cfg.get_disabled_plugins()
        if name not in disabled_plugins:
            return "Plugin already enabled"
        disabled_plugins.remove(name)
        cfg.set_config('disabled_plugins', disabled_plugins)
        return "Plugin enabled"
    def install_plugin(self, name: str):
        all_plugins = self.list_plugins()
        if name in all_plugins:
            return "Plugin already installed"
        all_plugins.append(name)
        cfg.set_config('plugins', all_plugins)
        cfg.set_config('disabled_plugins', cfg.get_disabled_plugins())

def register_commands(ctrl):
    print("Registering plugin manager commands")
    ctrl.add_command_class(PluginManagerCommands)

if __name__ == '__main__':
    from vertex_voyage.cluster import do_rpc_client, get_binding_port
    import sys 
    import argparse
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest='command')
    command.add_parser('list')
    disable_parser = command.add_parser('disable').add_argument("name", type=str, help='Name of the plugin to disable')
    enable_parser = command.add_parser('enable').add_argument("name", type=str, help='Name of the plugin to enable')
    install_parser = command.add_parser('install').add_argument("name", type=str, help='Name of the plugin to install')
    parser.add_argument("--host", type=str, help='Host to connect to', default='localhost')
    parser.add_argument("--port", type=int, help='Port to connect to', default=get_binding_port())
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.command == 'list':
        plugins = do_rpc_client(args.host, "list_plugins")
        for plugin in plugins:
            print(plugin)
    elif args.command == 'disable':
        print(do_rpc_client(args.host, "disable_plugin", name=args.name))
    elif args.command == 'enable':
        print(do_rpc_client(args.host, "enable_plugin", name=args.name))
    elif args.command == 'install':
        print(do_rpc_client(args.host, "install_plugin", name=args.name))
    else:
        parser.print_help()
        sys.exit(1)