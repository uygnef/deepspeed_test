import os
import sys
import time
import yaml
import json
import subprocess

from xml.dom import minidom
from string import Template


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green',
                 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def unicode_convert(value):
    if isinstance(value, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [unicode_convert(element) for element in value]
    elif type(value).__name__ == 'unicode':
        return value.encode('utf-8')
    else:
        return value


def xml2dict(xml_file):
    xml_config_dict = {}
    if xml_file is None:
        return xml_config_dict

    tree = minidom.parse(xml_file)
    root = tree.documentElement
    properties = root.getElementsByTagName('property')
    for prop in properties:
        dtype = prop.getAttribute('type')
        name = prop.getElementsByTagName('name')[0]
        name_key = name.childNodes[0].data
        value = prop.getElementsByTagName('value')[0]
        name_value = value.childNodes[0].data if len(
            value.childNodes) > 0 else ''
        if dtype in ['str', 'int', 'float']:
            name_value = eval(dtype)(name_value)
        elif dtype in ['list', 'dict', 'tuple', 'bool']:
            name_value = name_value.replace('\'', '\"')
            name_value = json.loads(name_value) # encoding is removed in py3.9
            name_value = unicode_convert(name_value)
        elif dtype == 'NoneType':
            name_value = None
        elif dtype in ['unicode', '']:
            name_value = str(name_value)
        else:
            print('name_key: %s type: %s, xml_type: %s, name_value: %s', name_key,
                  type(name_value).__name__, dtype, name_value)
            name_value = str(name_value)
        if name_key.startswith('args.') and '-' in name_key:
            print(set_color('replace xml param:', 'cyan'), end=' ')
            pstr = '%s with %s' % (name_key, name_key.replace('-', '_'))
            print(set_color(pstr, 'green'))
            name_key = name_key.replace('-', '_')
        xml_config_dict[name_key] = name_value
    return xml_config_dict


def yml2dict(file_list):
    yml_config_dict = {}
    if file_list is None:
        return yml_config_dict

    if isinstance(file_list, str):
        file_list = [file_list, ]
    for file in file_list:
        with open(file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        yml_config_dict.update(cfg)
    return yml_config_dict


def find_macro_in_yml(cfg_dict):
    macros = {}
    for key, value in cfg_dict.items():
        if key.isupper():
            macros[key] = value
            cfg_dict.pop(key)
    return macros


def recurrent_substitue(cfg_dict, macros):
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            recurrent_substitue(value, macros)
        if isinstance(value, str):
            value = Template(value).substitute(macros)
            cfg_dict[key] = value


def recurrent_dict(cfg_dict):
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            for key, value in recurrent_dict(value):
                yield key, value
        else:
            yield key, value


def save_config_to_json(xml_file):
    with open(xml_file) as f:
        xml_cfg = json.load(f)

    xml_cfg.update('')
    with open('./tmp.xml') as f:
        json.dump(xml_cfg)


def dict2xml(final_config):
    tree = minidom.Document()
    root = tree.createElement('configuration')
    tree.appendChild(root)
    keys = sorted(final_config.keys())

    # create preperty elements
    for key in keys:
        value = final_config.get(key)
        # create property cell
        prop = tree.createElement('property')
        # set data type attribute
        prop.setAttribute('type', type(value).__name__)
        # create name cell
        name_node = tree.createElement('name')
        name_text = tree.createTextNode(key)
        name_node.appendChild(name_text)
        prop.appendChild(name_node)
        # create value cell
        value_node = tree.createElement('value')
        if isinstance(value, str):
            assert ';' not in value, "can not use ; in dict str_value, because ; means shell command end"
            value_text = tree.createTextNode(value)
        elif value is None:
            value_text = tree.createTextNode('None')
        else:
            value_text = tree.createTextNode(
                json.dumps(value).replace('"', '\''))
        value_node.appendChild(value_text)
        prop.appendChild(value_node)

        root.appendChild(prop)
    return tree


def stitch_param_list(args):
    '''
    input: '--load_epoch=[', 'epoch-0', 'epoch-1]'
    return: '--load_epoch=[epoch-0, epoch-1]'
    '''
    res_args = []
    for arg in args:
        if arg.startswith('--'):
            res_args.append(arg)
        else:
            res_args[-1] += arg
    return res_args


def parse_update_elem(cfg, elem, logger):
    key, value = elem[2:].split('=', 1)  # remove '--'
    key = key if key in cfg else ('args.' + key)  # afo params
    if key not in cfg:
        logger.info('[arg_param] set %s = %s' % (key, value))
        cfg[key] = value
        return

    xml_value = cfg[key]
    dtype = type(xml_value).__name__
    if dtype in ['str', 'int', 'float']:
        update_value = eval(dtype)(value)
    elif dtype == 'bool':
        value = value.lower()
        assert value in ['true', 'false'], f'{key} is bool, must be true or false'
        update_value = True if value == 'true' else False
    elif dtype == 'NoneType':
        value = value.lower()
        update_value = None if value == 'none' else str(value)
    else:
        logger.info(f'[arg_param] {elem} dtype is {dtype}, use value in xml instead')
        return

    if xml_value != update_value:
        logger.info(f'[arg_param] {key} replace xml_value: {xml_value} with arg_value: {update_value}')
        cfg[key] = update_value


def hope_wait_status():
    hope_wait = os.environ.get('WAIT_FOR_HOPE_FINISH', 'false')
    return hope_wait


def create_hope_params(cfg):
    job_type = cfg.get('type')
    user_group = cfg.get('usergroup')
    require = '../../../src/conf/requirements.txt'
    hope_wait = hope_wait_status()

    # _train/_predict/_build/_eval
    # run_mode = cfg['args.run_mode']
    # if run_mode in ['train', 'train_and_evaluate']:
    #     mode = '_train'
    # elif run_mode == 'evaluate':
    #     mode = '_eval'
    # else:
    #     mode = '_predict'

    if cfg.get('args.run_mode') == 'workbench':
        return f'--usergroup={user_group} -Dtype={job_type}'
    elif job_type == 'ml-spark':
        return f'--usergroup={user_group} -Dtype={job_type}'

    return f'--usergroup={user_group} -Dtype={job_type} --requirements={require} -Dafo.engine.wait_for_job_finished={hope_wait}'


def create_git_info(conf_file):
    git_dir = '../../../.git'
    try:
        with open(os.path.join(git_dir, 'HEAD')) as f:
            branch = f.readline().split()[-1]
        with open(os.path.join(git_dir, branch)) as f:
            commit = f.readline().strip()
    except:
        return {'git_branch': "not find %s" % git_dir}
    home_url = 'https://dev.sankuai.com/code/repo-detail/wm_ai/starship_galaxy/file/'
    conf_path = conf_file[conf_file.index('src'):]
    remote_head = os.path.join(
        git_dir, branch.replace('heads', 'remotes/origin'))
    if os.path.exists(remote_head):
        git_url = home_url + ('detail?branch=%s&path=%s' %
                              (branch, conf_path)).replace('/', '%2F')
    else:
        git_url = home_url + 'list?'
    return {'git_url': git_url, 'git_commit': commit, 'git_branch': branch}


def set_device(cfg):
    ngpu = cfg.get('num_gpu')
    if not ngpu: return
    assert (ngpu < 8 and ngpu > 0) or ngpu % 8 == 0, 'gpu num must be < 8 or 8*N'
    cfg['workers'] = 1 if ngpu <= 8 else int(ngpu / 8)
    cfg['worker.vcore'] = int(20 * ngpu / cfg['workers'])
    cfg['worker.memory'] = int(115200 * ngpu / cfg['workers'])
    cfg['worker.gcores80g'] = int(ngpu / cfg['workers'])


def str_dict(cfg, color):
    color_func = set_color if color else lambda x, y: x
    infos = '\n'

    afo_info, arg_info = '', ''
    for key, value in sorted(cfg.items()):
        if key.startswith('args.'):
            arg_info += (color_func("{}", 'cyan') + " = " +
                         color_func("{}", 'yellow') + '\n').format(key, value)
        else:
            afo_info += (color_func("{}", 'cyan') + " = " +
                         color_func("{}", 'yellow') + '\n').format(key, value)

    infos += color_func('Afo Parameters: \n', 'pink') + afo_info
    infos += color_func('\nArg Parameters: \n', 'pink') + arg_info

    infos += '\n'

    return infos


def get_log_file_name(log_file, app_name=None):
    if log_file:
        log_dir = os.path.dirname(log_file)
    else:
        log_dir = os.path.join('../log', os.path.basename(os.getcwd()))
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = os.path.join(log_dir, f'{app_name}_{time_str}.log')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_file


def get_default_params():
    '''add default param, sort by letter order'''
    params = {
        # train_params
        "args.batch_size": 100,
        "args.epoch_num": 10,
        "args.load_epoch": None,
        "args.lr": 0.01,
        "args.lr_scheduler": "",
        "args.timeline_enable": False,
        "args.wd": 0.0,
        "args.use_bn_sync": False,
        "args.log_step": 50,
        "args.main_process_log": True,
        "args.save_after_epoch": True,
        "args.predict_after_epoch": True,
        "args.evaluate_after_epoch": True,
        "args.trainer": "runner.trainer.Trainer",
    }
    return params


class Configurator(object):
    """
    """

    def __init__(self, params={}, xml_file=None, yml_file=None):
        # priority order: params > yml_file > xml_file > default_params
        self.final_config = {}
        self.afo_params = {}
        self.parse_params = {}
        # self.conf_file = sys._getframe(1).f_code.co_filename
        self.yml_config_dict = yml2dict(yml_file)
        self.xml_config_dict = xml2dict(xml_file)
        self.final_config.update(get_default_params())
        self.final_config.update(self.xml_config_dict)
        self.final_config.update(self.yml_config_dict)
        self.final_config.update({k.replace('-', '_'): v for k, v in params.items()})        
        # self.final_config.update(create_git_info(self.conf_file))
        self.set_params()

    def set_train_params(self, params):
        self.train_params = params
        self.final_config.update(params)
        for key in params.keys():
            assert '-' not in key, "'-' in %s is not allowed, use '_' instead" % key

    def set_afo_params(self, params):
        self.afo_params = params
        self.final_config.update(params)

    def set_arg_params(self, args):
        '''
        set params from argparser
        '''
        kwargs = args._get_kwargs()
        for key, value in kwargs:
            self.parse_params["args." + key] = value
        self.final_config.update(self.parse_params)

    def set_list_params(self, elems, logger):
        '''
        set params which are not defined in argparser
        '''
        elems = stitch_param_list(elems)
        for elem in elems:
            parse_update_elem(self.final_config, elem, logger)

    def save_config_to_xml(self, xml_file):
        '''
        used when submit job via hope
        '''
        set_device(self.final_config)
        tree = dict2xml(self.final_config)
        # write conf dict to xml template
        with open(xml_file, 'w') as f:
            tree.writexml(f, indent='', addindent='\t',
                          newl='\n', encoding='utf-8')
        # test load xml_file
        assert xml2dict(
            xml_file) == self.final_config, 'file.xml is different from config.py'
        self.check_valid_params()

    def set_params(self):
        for key, value in self.final_config.items():
            # only set item starting with "args."
            if key.startswith('args.'):
                key = key[5:]
                setattr(self, key, value)

    def check_valid_params(self):
        cfg = self.final_config
        if cfg.get('args.receivers', 'RECEIVERS') != 'RECEIVERS':
            assert '@' in cfg.get(
                'args.receivers'), "args.receivers act like MIS_ID@meituan.com"
        if cfg.get('args.run_mode') in ['train']:
            assert isinstance(cfg.get('args.batch_size'),
                              int), 'args.batch_size must be a int value'
            assert isinstance(cfg.get('args.ckpt_dir'),
                              str), 'ckpt_dir must be set'
        return True

    def submit_job(self, xml_file, device=None, log_file=None):
        cfg = self.final_config
        is_spark = 'ml-spark' in cfg.get('type', '')
        app_name = cfg['appName'] if is_spark else cfg['afo.app.name']
        self.log_file = get_log_file_name(log_file, app_name)
        if is_spark:
            self.submit_spark_hope(cfg, xml_file)
        elif 'local' in cfg.get('args.run_mode', ''):
            self.submit_job_local(cfg, xml_file)
        else:
            self.submit_job_hope(cfg, xml_file)

    def submit_spark_hope(self, cfg, xml_file):
        hope_params = create_hope_params(cfg)
        self.exec_shell(
            f'hope run --xml={xml_file} --workdir=./ {hope_params}')

    def submit_job_hope(self, cfg, xml_file):
        model = '../../../src'
        hope_params = create_hope_params(cfg)
        if cfg.get('args.run_mode') == 'workbench':
            self.exec_shell(
                f"hope workbench --xml={xml_file} --workdir={model} --init --dual-gpus {hope_params}")
        else:
            self.exec_shell(
                f"hope run --xml={xml_file} --workdir={model}  {hope_params} --force")

    def submit_job_local(self, cfg, xml_file=None):
        nproc = cfg.get('worker.gcores80g')
        command = cfg.get('worker.script')
        # export NCCL_DEBUG=INFO   # self.exec_shell(
        print(f'execute cmd: {command}')
        os.system(
            f'cd ../../..; PYTHONPATH=./;'
            f'python -m torch.distributed.launch --nproc_per_node={nproc} --nnodes=1 --node_rank=0 '
            f'{command}')

    def exec_shell(self, command):
        print(f'execute cmd: {command}')
        with open(self.log_file, 'w') as f:
            f.write(f'{repr(self)}{command} \n')
            p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=sys.stderr)
            while True:
                ret_code = p.poll()
                if ret_code is None:
                    line = p.stdout.readline().decode()
                    print(f'{line}', end='')
                    f.write(line)
                else:
                    if ret_code != 0:
                        sys.exit(ret_code)
                    else:
                        break

    def remove_hdfs_file(self, rm, file_path):
        '''Delete hdfs file via hope or hadoop.

        Args:
          rm: whether to delete.
          file_path: path to be deleted.
        '''
        if not rm:
            return

        if sys.platform.lower() == 'darwin':
            ret = os.system(f'hope dfs -rm -r {file_path}')
        else:
            ret = os.system(f'hadoop fs -rmr {file_path}')

        if ret != 0:
            print(
                'please login with usergroup, eg. hope login --user=hadoop-hmart-waimai-rank')

    def get(self, key, value=None):
        if key in self.final_config:
            return self.final_config.get(key)
        else:
            return self.final_config.get('args.' + key, value)

    def _get_kwargs(self):
        return sorted(self.final_config.items())

    def __str__(self):
        color = hope_wait_status() == 'false'
        return str_dict(self.final_config, color=color)

    def __repr__(self):
        return str_dict(self.final_config, color=False)

    def __getitem__(self, key):
        return self.final_config[key]

    @staticmethod
    def get_default_script(xml_file, mode=''):
        '''
        set max_restarts=0 to disable torch elastic launch.
        '''
        if 'local' in mode.lower():
            return f'src/app/main.py --conf=./src/scheduler/{os.path.basename(os.getcwd())}/{xml_file}'
        else:
            return f'--max_restarts=0 src/app/main.py --conf=./src/scheduler/{os.path.basename(os.getcwd())}/{xml_file}'

    @staticmethod
    def get_pre_dir(log_dir):
        '''
        set board_log_dir to previous dir, so we can show all log events in the same tensorboard page.
        '''
        return os.path.abspath(os.path.join(log_dir, '../'))
