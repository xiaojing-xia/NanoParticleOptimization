#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''simulation-optimization loop'''

import argparse
from contextlib import ExitStack
from datetime import datetime
import logging
import math
import numpy as np
import os
import pandas as pd
import time

from fireworks import LaunchPad
from jobflow import JobStore
from jobflow.managers.fireworks import flow_to_workflow
from maggma.stores.mongolike import MongoStore
from NanoParticleTools.flows.flows import get_npmc_flow
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
import uuid

import torch
from torch.quasirandom import SobolEngine

import gpytorch
import gpytorch.settings as gpts
import pykeops
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
pykeops.test_torch_bindings()

from common import seed_generator
from common import utils
from common import configs

seed_generator = seed_generator.SeedGenerator()

# credential
LP_FILE = '/Users/xiaojing/my_launchpad.yaml'
LP = LaunchPad.from_file(LP_FILE)

DOCS_STORE = MongoStore.from_launchpad_file(LP_FILE, 'test_fws_npmc')
DATA_STORE = MongoStore.from_launchpad_file(LP_FILE, 'test_docs_npmc')
LAUNCH_STORE = MongoStore.from_launchpad_file(LP_FILE, 'launches')

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

RANGES = {'UV': [300, 450],
          'VIS': [400, 700],
          'BLUE': [450, 500],
          'GREEN': [500, 590],
          'RED': [610, 700],
          'TOTAL': [200, 900],
          'ABSORPTION': [950, 1030],
         }

def encode_inputs(x_arr, x_max = 34):
    '''encode simulation input to botorch'''
    for i, arr in enumerate(x_arr):
        x_arr[i, 0] = arr[0] + arr[1]
        if arr[0] + arr[1] == 0:
            x_arr[i, 1] = 0.5
        else:
            x_arr[i, 1] = arr[0] / (arr[0] + arr[1])
        x_arr[i, 2] = arr[2] + arr[3]
        if arr[2] + arr[3] == 0:
            x_arr[i, 3] = 0.5
        else:
            x_arr[i, 3] = arr[2] / (arr[2] + arr[3])
        x_arr[i, 4] = arr[4] / x_max


def decode_candidates(x_arr, x_max = 34):
    '''decode botorch recommendation candidates for simulation'''
    for i, arr in enumerate(x_arr):
        x_arr[i, 0], x_arr[i, 1] = arr[0] * arr[1], arr[0] * (1 - arr[1])
        x_arr[i, 2], x_arr[i, 3] = arr[2] * arr[3], arr[2] * (1 - arr[3])
        x_arr[i, 4] = arr[4] * x_max


def average_simulation_csv(simulation_file, dest_file):
    '''average previous done simulation'''
    df = pd.read_csv(simulation_file, index_col=False).sort_values(by = ['yb_1', 'er_1', 'yb_2', 'er_2', 'radius'])
    df = df.reset_index(drop=True)
    df2 = pd.DataFrame(data=None, columns=df.columns)
    start_index = 0
    for index, row in df.iterrows():
        if index == 0:
            continue
        current_row = list(df.iloc[index, :5])
        previous_row = list(df.iloc[index-1, :5])
        if current_row == previous_row:
            continue
        else:
            df2 = df2.append(df.loc[start_index:index-1].mean().round(6), ignore_index=True)
            start_index = index
    df2.to_csv(dest_file, index=False)

    
def convert_time(time_str):
    '''convert time stamp string from fw to unix timestamp'''
    dt_utc = datetime.strptime(time_str, DATETIME_FORMAT)
    unix_timestamp = time.mktime(dt_utc.timetuple())
    
    return unix_timestamp


def get_data_botorch(data_file):
    my_data = np.loadtxt(data_file, delimiter=',', skiprows=1)

    # features
    train_x = torch.from_numpy(my_data[:, :5])
    # labels
    train_y = torch.from_numpy(my_data[:, 5]).unsqueeze(-1)
    # best observation
    best_y = train_y.max().item()
    
    return train_x, train_y, best_y


def get_job_uuid(fw_id):
    DOCS_STORE.connect()
    
    fws = DOCS_STORE.query({'metadata.fw_id': fw_id})
    fws = list(fws)
    if len(fws) != 1:
        raise RuntimeError(f'found duplicated fw_id: {fw_id}')
    
    return fws[0]['uuid']


def submit_jobs(candidates):
    '''
    submit a recipe for simulation
    '''
    store = JobStore(DOCS_STORE, additional_stores={'trajectories': DATA_STORE})
    submitted_id = []
    candidates = candidates.tolist()
    
    for candidate in candidates:
        # iterate candidates
        dopant_specifications = []
        for spec in configs.cfg['dopant_specifications']:
            spec = list(spec)
            if 'Surface6' in spec:
                dopant_specifications.append(tuple(spec))
                break
            elif spec[1] == -1:
                spec[1] = candidate.pop(0)
                dopant_specifications.append(tuple(spec))
            else:
                dopant_specifications.append(tuple(spec))
        
        constraints = []
        for radii in configs.cfg['radius']:
            if radii == -1:
                constraints.append(SphericalConstraint(candidate.pop(0)))
            else:
                constraints.append(SphericalConstraint(radii))
                
        npmc_args = {'npmc_command': configs.cfg['npmc_command'], #NPMC
                     'num_sims': configs.cfg['num_sims'], #4
                     # 'base_seed': seed_generator.genereate(), #1000
                     'base_seed': 1000, 
                     'thread_count': configs.cfg['ncpu'],
                     #'simulation_length': 100000 #
                     'simulation_time': configs.cfg['simulation_time'] #0.01s
                     }
        spectral_kinetics_args = {'excitation_wavelength': configs.cfg['excitation_wavelength'],
                                  'excitation_power': configs.cfg['excitation_power']}

        initial_state_db_args = {'interaction_radius_bound': configs.cfg['interaction_radius_bound']} #3

        # is this being used?
        np_uuid = str(uuid.uuid4())

        for doping_seed in range(configs.cfg['num_dopant_mc']): #4
            flow = get_npmc_flow(constraints=constraints,
                                 dopant_specifications=dopant_specifications,
                                 # doping_seed=seed_generator.generate(),
                                 doping_seed = doping_seed, 
                                 spectral_kinetics_args=spectral_kinetics_args,
                                 initial_state_db_args=initial_state_db_args,
                                 npmc_args=npmc_args,
                                 output_dir=configs.cfg['output_dir']
                                 )

            wf = flow_to_workflow(flow, store=store)
            mapping = LP.add_wf(wf)
            submitted_id.append(list(mapping.values())[0])
            
        print(f"Initialized {configs.cfg['num_dopant_mc']} jobs. Submitted {len(submitted_id)}.")

    return submitted_id


def run_jobs(fw_ids):
    pass


def monitor(fw_ids):
    '''monitor status of submitted jobs by looking up the fw_id'''
    LAUNCH_STORE.connect()
    all_done = [False] * len(fw_ids)
    runtimes = [-1] * len(fw_ids) 
    
    for i, fw_id in enumerate(fw_ids):
        done = all_done[i]
        ready_count = 0
        running_count = 0
        while not done:
            # publisher has a 60s sleep. give it 120s to be safe...
            for j in range(120):
                launch = LAUNCH_STORE.query({'fw_id': fw_id})
                launch = list(launch)
                if len(launch) == 0:
                    time.sleep(1)
                else:
                    if running_count == 0:
                        print(f"waited {j} seconds for query {fw_id}...")
                    break
                
            launch = launch[-1]
            if launch['state'] == 'COMPLETED':
                done = True
                all_done[i] = True
                time_start = convert_time(launch['time_start'])
                time_end = convert_time(launch['time_end'])
                runtime = time_end - time_start
                # NOTE(xxia): the time at fw is UTC, which is different from local clock
                print(f"job {fw_id} ended at {launch['time_end']}. took {runtime} seconds.")
                runtimes[i] = runtime
                
            elif launch['state'] == 'READY':
                print(f"time: {datetime.datetime.fromtimestamp(time.time())}: {fw_id} ready! Waiting to start. "
                      f"Been {ready_count} minutes!")
                ready_count += 1
                # check for 1hr
                if ready_count > 60:
                    raise RuntimeError(f"{fw_id} failed to start in {ready_count + 1} minutes!")
                time.sleep(60)
                
            elif launch['state'] == 'RUNNING':
                print(f"time: {datetime.fromtimestamp(time.time())}: {fw_id} still running! Been {running_count + 1} minutes!")
                running_count += 1
                # check for 2hrs
                if running_count > 120:
                    print(f"{fw_id} failed to complete in {running_count + 1} minutes!")
                    break # break the while loop
                time.sleep(60)
                
            elif launch['state'] == 'FIZZLED':
                print(f"something went wrong in {fw_id}! Fizzled.") 
                break
                
            else:
                raise RuntimeError(f"unknown state: {launch['state']}") 

    return all_done, runtimes


def get_results(all_done, fw_ids):
    '''get finished simulation result and log it (averaged) to log file'''
    DATA_STORE.connect()
    DATA_DEST = configs.cfg["data_file"]
    
    log = pd.read_csv(DATA_DEST)
    df = pd.DataFrame()
    for done, fw_id in zip(all_done, fw_ids):
        if done:
            uuid = get_job_uuid(fw_id)
            docs = DATA_STORE.query({'job_uuid': uuid})
            docs = list(docs)
            for i, doc in enumerate(docs):
                data = {}
                data['yb_1'] = 0
                data['er_1'] = 0
                data['yb_2'] = 0
                data['er_2'] = 0
                for dopant in doc['data']['input']['dopant_specifications']:
                    if dopant[0] == 0 and dopant[2] == 'Yb':
                        data['yb_1'] = dopant[1]
                    if dopant[0] == 0 and dopant[2] == 'Er':
                        data['er_1'] = dopant[1]
                    if dopant[0] == 1 and dopant[2] == 'Yb':
                        data['yb_2'] = dopant[1]
                    if dopant[0] == 1 and dopant[2] == 'Er':
                        data['er_2'] = dopant[1]

                data['radius'] = doc['data']['input']['constraints'][0]['radius']

                for spec, spec_range in RANGES.items():
                    data[spec] = utils.get_int(doc, spec_range)

                data['qe'] = utils.get_qe(doc, RANGES['TOTAL'], RANGES['ABSORPTION'])
                
                df = df.append(data, ignore_index=True)
                print(f"job {fw_id} result {i} collected.")
        else:
            print(f"job {fw_id} results abandoned.")
    
    if sum(all_done) * 4 != len(df):
        raise ValueError(f"number of completed simulation: {sum(all_done) * 4} != number of queried results {len(df)}")
    
    if df.empty:
        print("all jobs in this loop failed. check configuration.")
        restart = input("Do want to continue loop? Press \"y\" to continue or press any key to end ")
        if restart != "y" and restart != "Y":
            print("shutting down")
            sys.exit()
            
    log = log.append(df.mean().round(6), ignore_index=True)
    log.to_csv(DATA_DEST, index=False)
    print("loop results sucessfully appended!")
    print("current total number of results:", len(log))
    print("current best UV:", log["UV"].max())
    print("current best structure:")
    print(log.iloc[log["UV"].idxmax(), :5].to_string(index=True))

                         
def append_data_to_csv(doc):
    data = {}
    data['yb_1'] = 0
    data['er_1'] = 0
    data['yb_2'] = 0
    data['er_2'] = 0
    for dopant in doc['data']['input']['dopant_specifications']:
        if dopant[0] == 0 and dopant[2] == 'Yb':
            data['yb_1'] = dopant[1]
        if dopant[0] == 0 and dopant[2] == 'Er':
            data['er_1'] = dopant[1]
        if dopant[0] == 1 and dopant[2] == 'Yb':
            data['yb_2'] = dopant[1]
        if dopant[0] == 1 and dopant[2] == 'Er':
            data['er_2'] = dopant[1]

    data['radius'] = doc['data']['input']['constraints'][0]['radius']

    for spec, spec_range in RANGES.items():
        data[spec] = get_int(doc, spec_range)

    data['qe'] = get_qe(doc, RANGES['TOTAL'], RANGES['ABSORPTION'])
    
    my_df = my_df.append(data, ignore_index=True)

    things = [EXCITATION_WAVELENGTH, EXCITATION_POWER, NUMBER_LAYERS, THIRD_LAYER_RADIUS, SECOND_LAYER_RADIUS]
    file_name = '_'.join(str(int(s)) for s in things)
    file_dest = FILE_DEST + file_name + '.csv'
    my_df.to_csv(file_dest, index=False)                 

    
def recommend(train_x, train_y, best_y, bounds, n_trails = 1):
    if isinstance(bounds, list):
        bounds = torch.tensor(bounds)
    elif torch.is_tensor(bounds):
        pass
    else:
        raise TypeError(f"expect bounds in a list or tensor. was given {type(bounds)}")
    
    single_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_model(mll)
    
    EI = qExpectedImprovement(model = single_model, best_f = best_y)
    
    # hyperparameters are super sensitive here
    candidates, _ = optimize_acqf(acq_function = EI,
                                 bounds = bounds, 
                                 q = 1, 
                                 num_restarts = 200, 
                                 raw_samples = 412, 
                                 options = {'batch_limit': 5, "maxiter": 200}
                                 )
    
    return candidates


def thompson_sampling(X, Y, batch_size, n_candidates, sampler="cholesky",  # "cholesky", "ciq", "rff"
    use_keops=False,):

    assert sampler in ("cholesky", "ciq", "rff", "lanczos")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # NOTE: We probably want to pass in the default priors in SingleTaskGP here later
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": X.shape[-1]}
    if sampler == "rff":
        base_kernel = RFFKernel(**kernel_kwargs, num_samples=1024)
    else:
        base_kernel = (
            KMaternKernel(**kernel_kwargs) if use_keops else MaternKernel(**kernel_kwargs)
        )
    covar_module = ScaleKernel(base_kernel)

    # Fit a GP model
    train_Y = (Y - Y.mean()) / Y.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    model = SingleTaskGP(X, train_Y, likelihood=likelihood, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Draw samples on a Sobol sequence
    sobol = SobolEngine(X.shape[-1], scramble=True)
    X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)

    # Thompson sample
    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(gpts.minres_tolerance(2e-3))  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def run_study():
    max_loops = configs.cfg["max_loops"]
    bounds = torch.tensor(configs.cfg["bounds"])
    DATA_DEST = configs.cfg["data_file"]
    for i in range(max_loops):
        train_x, train_y, best_y = get_data_botorch(DATA_DEST)
        
        encode_inputs(train_x)
        #candidates = recommend(train_x, train_y, best_y, bounds)
        candidates = thompson_sampling(train_x, train_y, 1, 1)
        print(f"recommending: {candidates}")
        decode_candidates(candidates)
        print(f"actual recommended recipe: {candidates}")
        
        fw_ids = submit_jobs(candidates)
        
        # run_jobs(fw_ids)
        
        # sample data for a quick test
        # fw_ids = [2385, 2386]
        
        all_done, runtimes = monitor(fw_ids)
        print(f"submitted {len(fw_ids)} jobs. sucessfully completed {sum(all_done)}.")
        for done, fw_id in zip(all_done, fw_ids):
            if done:
                print(f"{fw_id} done")
            else:
                print(f"{fw_id} failed")

        get_results(all_done, fw_ids)
        
    print("all loops done!")

    
def main():
    parser = argparse.ArgumentParser(description = "optimize using simulation")
    parser.add_argument("-n", "--ncpu", type=int, dest="ncpu",
                       default=4,
                       help="number of cpus")
    
    parser.add_argument("-c", "--configs", dest="configs",
                       default="common/defaults.cfg",
                       help="Configuration file")
    
    parser.add_argument("-v", "--verbose", dest="verbose",
                       action="store_true",
                       help="more verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    # most runtime info already handled by flow_to_workflow. created one-time time stamp to make things unique.
    cmdline = {
        "verbose": args.verbose,
        "config_file": args.configs,
        "ncpu": args.ncpu,
        "timestamp": int(time.time() * 1000)
    }
    
    # after this call, the config code becomes global which 
    # tries to treat the configs.cfg as a frozen dictionary to avoid accidents.
    configs.cfg = configs.Configuration(fname=args.configs,
                                        defaults=cmdline)
        
    run_study()

    
if __name__ == '__main__':
    main()

