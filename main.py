import os
import argparse
from accelerate.logging import get_logger
from configurator import Configurator
from train_text_to_image import train

logger = get_logger("aaa")

def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False

def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def init_distribute_param():
    local_rank, global_rank, world_size = world_info_from_env()
    distributed = is_using_distributed()
    return {
        'args.local_rank': local_rank,
        'args.global_rank': global_rank,
        'args.world_size': world_size,
        'args.distributed': distributed,
        'args.dist_url': None,
        'args.rank': 0  # 废弃
    }

def cmd_parser():
    parser = argparse.ArgumentParser(description="FNN Train and Evaluate script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--conf', type=str, default="",
                        help='cmd config file.')

    '''
    params used in afo dataset, gpu and cpu
    '''
    parser.add_argument('--use_socket', type=bool, default=False, help='1')
    parser.add_argument('--data_port', type=int, default=0, help='')
    parser.add_argument('--shm_size', type=int, default=0, help='')
    parser.add_argument('--shm_name', type=str, default='', help='')
    parser.add_argument('--data_ports', type=str, default='', help='')
    parser.add_argument('--shm_names', type=str, default='', help='')

    '''
    params used in tensorflow
    '''
    parser.add_argument('--worker_hosts', type=str, default='',
                        help='woker host.')
    parser.add_argument('--task_index', type=int, default=1,
                        help='index of task within the job.')

    '''
    add distribute param
    '''
    dist_params = init_distribute_param()

    '''
    argument in front will coverage params in .xml
    '''
    args, unknowns = parser.parse_known_args()

    if args.conf.endswith('.xml'):
        conf = Configurator(dist_params, xml_file=args.conf)
        conf.set_arg_params(args)
        conf.set_list_params(unknowns, logger)
        conf.set_params()
        args = conf

    return args


def cmd_infos(args):
    l = args._get_kwargs()
    for e in l:
        logger.info('%s = %s' % (e[0], e[1]))


def main():
    # parse args and config params, init distribute params
    args = cmd_parser()

    # log args param infos
    cmd_infos(args)

    # run    
    train(args)
    
    logger.info('main finish')


if __name__ == '__main__':
    main()