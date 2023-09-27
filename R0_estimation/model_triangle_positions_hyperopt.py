import sys, os, argparse, json, yaml, subprocess, secrets, shutil
from datetime import datetime
from pathlib import Path
from functools import partial

from hyperopt import fmin, tpe, space_eval
from hyperopt.mongoexp import MongoTrials
from hyperopt import Trials

import hyperopt as hp

__homedir__ = str(Path(__file__).resolve().parents[1])
os.system(f"export PYTHONPATH={__homedir__}:$PYTHONPATH")  # note the export! Required so that mongod processes have same access
sys.path.append(__homedir__)

from helper_functions import parse_output_path, parse_input_path
from model_triangle_positions import model_triangle_positions


def generate_header(loss):
    out_txt = f"""
# TRIANGLE PARAMETERS HYPEROPT-GENERATED PARAMETER FILE
# timestamp: {datetime.now().strftime('%y-%m-%d_%H:%M:%S')}
# loss: {loss}

"""
    return out_txt

def objective(params, fret_dict, non_variable_args, out_dir):
    dir_id = secrets.token_hex(20)
    # dir_id = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
    cur_out_dir = parse_output_path(f'{out_dir}{dir_id}')
    for nv in non_variable_args:
        params[nv] = non_variable_args[nv]
    loss = model_triangle_positions(fret_dict, params, cur_out_dir, non_variable_args['dna_shape'], False)
    shutil.move(cur_out_dir, f'{out_dir}{loss}_{dir_id}')
    return {'loss': loss,
            # 'dir_id': dir_id,
            'status': hp.STATUS_OK}

def optimize(fret_dict, ranges_dict, out_dir, hyperopt_iters, parallel_jobs):

    start_time = datetime.now()
    # make paths
    out_dir_iters = parse_output_path(f'{out_dir}iters', clean=True)
    mongodb_pth = parse_output_path(f'{out_dir}mongodb', clean=True)

    # define search space
    space = {}
    for var in ranges_dict['variable']:
        cd = ranges_dict['variable'][var]
        space[var] = hp.hp.quniform(var, cd['min'], cd['max'], cd['step'])

    # formulate objective function
    fmin_objective = partial(objective, fret_dict=fret_dict, non_variable_args=ranges_dict['nonvariable'],
                             out_dir=out_dir_iters)
    if parallel_jobs > 1:

        # Start mongod process
        subprocess.run(["mongod",
                        "--dbpath", mongodb_pth,
                        "--port", "1234",
                        "--directoryperdb",
                        "--fork",
                        "--journal",
                        "--logpath", f"{out_dir}mongodb_log.log"
                        ])

        # start worker processses
        # worker_cmd_list = [f"PYTHONPATH={__homedir__}", "hyperopt-mongo-worker", "--mongo=localhost:1234/db", f'--workdir={out_dir}']
        worker_cmd_list = ["hyperopt-mongo-worker", "--mongo=localhost:1234/db"]
        # worker_list = []
        worker_list = [subprocess.Popen(worker_cmd_list,
                                        stdout=open(os.devnull, 'wb'),
                                        stderr=open(os.devnull, 'wb')
                       )
                       for _ in range(parallel_jobs)]
        trials = MongoTrials('mongo://localhost:1234/db/jobs',
                             exp_key=datetime.now().strftime('%Y%m%d%H%M%S')
                             )
    else:
        trials = Trials()

    # Minimize objective
    out_param_dict = hp.fmin(fmin_objective,
                             space=space,
                             algo=hp.atpe.suggest,
                             trials=trials,
                             max_evals=hyperopt_iters,
                             max_queue_len=10)

    if parallel_jobs > 1:
        for worker in worker_list:
            worker.terminate()
        subprocess.run(['mongod', '--shutdown', '--dbpath', mongodb_pth])

    print(out_param_dict)
    for p in ranges_dict['nonvariable']:
        out_param_dict[p] = ranges_dict['nonvariable'][p]

    # store hyperparams as new parameter file
    out_param_dict['HEADER'] = generate_header(min(trials.losses()))
    with open(f'{out_dir}parameterFile.json', 'w') as fh:
        json.dump(out_param_dict, fh)

    # Run with new parameter file
    out_dir_final = parse_output_path(f'{out_dir}best_run')
    model_triangle_positions(fret_dict, out_param_dict, out_dir_final, out_param_dict['dna_shape'], False)

    wall_time = datetime.now() - start_time
    print(f'Done! Run took {wall_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find best solution to triangle-fitting problem')
    parser.add_argument('--fret-json', type=str, required=True)
    parser.add_argument('--ranges', type=str, default=f'{__homedir__}/triangle_reconstruction/dna_property_params/parameter_ranges.yaml')
    parser.add_argument('--hyperopt-iters', type=int, default=1000)
    parser.add_argument('--parallel-jobs', type=int, default=4)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    with open(args.fret_json, 'r') as fh:
        fret_dict = json.load(fh)
    with open(args.ranges, 'r') as fh:
        ranges_dict = yaml.load(fh, yaml.FullLoader)
    out_dir = parse_output_path(args.out_dir, clean=True)
    optimize(fret_dict, ranges_dict, out_dir, args.hyperopt_iters, args.parallel_jobs)
