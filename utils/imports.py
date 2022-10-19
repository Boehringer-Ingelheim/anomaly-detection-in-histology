import importlib
import os
import pickle
import ast

def import_config_from_file(path):

    modul_name = os.path.splitext(os.path.split(path)[1])[0]
    spec = importlib.util.spec_from_file_location(modul_name, path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    return cfg


def show_configuration(cfg, file=None):

    if not file:
        print('---used configuration---')
    for att in dir(cfg):

        if not att.startswith('_'):
            attr_value = getattr(cfg, att)

            normal_print = True
            if isinstance(attr_value, (list, tuple, dict)) and len(str(attr_value)) > 100:
                for val in attr_value:
                    if isinstance(val, (list, tuple, dict)):
                        normal_print = False
                        break

            if normal_print:
                print('{:<30} {:>100}'.format(att, str(attr_value)), file=file)
            else:
                for i, current_attr in enumerate(attr_value):
                    print('{:<30} {:>100}'.format(att + '[' + str(i) + ']', str(current_attr)), file=file)
                

def save_configuration(cfg, path):

    with open(path, 'w') as f:
        show_configuration(cfg, file=f)


def pickle_configuraton_as_dictionary(cfg, path):

    dic = {}   
    for att in dir(cfg):

        if not att.startswith('_'):
            dic[att] = getattr(cfg, att)

    pickle.dump(dic, open(path, 'wb'))


def update_configuration(parser=None, cfg=None):
    """
    Adds or overwrites configuration with parameters provided in command line

    :param parser: parser from argparse
    :param cfg: configuration (will be changed to new configuration inplace)
    :return: configuration with overwritten/added parameters from command line

    cfg = get_configuration(parser) - will load configuration from file provided --config cfg_file.py, adds/overwrites
    parameters from command line with prefix P, e.g. -Pdevice='cuda:0' -Pseed=100 overwrites parameters 'device' and
    'seed' from cfg_file.py

    cfg = get_configuration(parser, cfg) - will return configuration cfg with overwritten parameters from command line
    with prefix P, e.g. -Pdevice='cuda:0' -Pseed=100 overwrites parameter 'device' ad 'seed' from cfg.
    Also works inplace: get_configuration(parser, cfg)

    cfg = get_configuration(cfg) - just returns configuration cfg (nothing is done)

    """

    if cfg is None and parser is None:
        logging.error("if configuration cfg is not given, parser must be provided to load configuration")
        raise

    if parser is not None:
        parser.add_argument('--config', type=str, help="config file", default=None)
        parser.add_argument('-P', action='append')
        args = parser.parse_args()

        if cfg is None:
            cfg = import_config_from_file(args.config)

        if parser.parse_args().P is not None:
            overwrite_parameters = {}
            for key, value in [s.split('=') for s in args.P]:
                try:
                    key = ast.literal_eval(key)
                except ValueError:
                    pass

                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass

                overwrite_parameters[key] = value

            for par in overwrite_parameters.keys():
                setattr(cfg, par, overwrite_parameters[par])

    return cfg


