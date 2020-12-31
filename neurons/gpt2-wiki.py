#!/bin/python3
"""GPT2 Language Modelling miner

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python neurons/gpt2-wiki.py

"""
import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
import traceback
import time
import bittensor

from termcolor import colored
from munch import Munch
from datasets import load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from bittensor import Session
from bittensor.utils.logging import log_all
from bittensor.config import Config
from bittensor.synapses.gpt2 import GPT2LMSynapse, nextbatch
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--neuron.datapath', default='data', type=str,help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
    parser.add_argument('--neuron.batch_size_train', default=4, type=int, help='Training batch size.')
    parser.add_argument('--neuron.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
    parser.add_argument('--neuron.log_interval', default=10, type=int, help='Batches before we log session info.')
    parser.add_argument('--neuron.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
    parser.add_argument('--neuron.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--neuron.name', default='gpt-wiki', type=str, help='Trials for this neuron go in neuron.datapath / neuron.name')
    parser.add_argument('--neuron.trial_id', default=str(time.time()).split('.')[0], type=str, help='Saved models go in neuron.datapath / neuron.name / neuron.trial_id')
    parser.add_argument('--neuron.record_log', default=True, help='Record all logs when running this session')
    GPT2LMSynapse.add_args(parser)

def check_config(config: Munch):
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
    trial_path = '{}/{}/{}'.format(config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
    config.neuron.trial_path = trial_path
    if not os.path.exists(config.neuron.trial_path):
        os.makedirs(config.neuron.trial_path)
    GPT2LMSynapse.check_config(config)

# Neuron main.
def main(config: Munch, session: Session):

    #  # ---- Model ----
    model = GPT2LMSynapse(config, session)

    # ---- Serve ----
    session.serve( model ) # Serves model to Axon RPC endpoint for network access.

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr = config.neuron.learning_rate, momentum=config.neuron.momentum)
    #scheduler = WarmupLinearSchedule(optimizer,5,1000)
    scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, 50, 300)

    # ---- Dataset ----
    # 74 million sentences pulled from books.
    dataset = load_dataset('bookcorpus')['train']

    tensorboard = SummaryWriter(log_dir = config.neuron.trial_path)

    if config.neuron.record_log:
        current_time = time.strftime("%H_%M_%S", time.localtime())
        logger.add("{}_{}.log".format(config.neuron.name, current_time),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

    # ---- Train Epoch ----
    def train(epoch: int, global_step: int):
        # ----- Init training state ---
        model.train()
        session.metagraph.sync() # Sync with the chain.
        row_weights = session.metagraph.W[ 0, :] # My weights on the chain-state (zeros initially).
        history = []
        local_step = 0
        local_epochs = 10
        output = None
        net = torch.nn.DataParallel(model)

        while local_step < local_epochs:
            try:
                inputs = nextbatch(dataset, config.neuron.batch_size_train, bittensor.__tokenizer__())

                output = net(
                    inputs,
                    training = True,
                    remote = True # WITH rpc-queries made to the network
                )

                # ---- Backward pass ----
                output.loss.backward() # Accumulates gradients on the model.
                optimizer.step() # Applies accumulated gradients.
                optimizer.zero_grad() # Zeros out gradients for next accummulation

                history.append(output) # Save for later analysis/logs.
                logger.info('GS: {} LS: {} Epoch: {} \t Local Target Loss: {}\tRemote Target Loss: {}\tDistillation Loss: {}\t Dendrite: {}\t Axon: {}',
                        colored('{}'.format(global_step), 'red'),
                        colored('{}'.format(local_step), 'blue'),
                        colored('{}'.format(epoch), 'green'),
                        colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                        colored('{:.4f}'.format(output.remote_target_loss.item()), 'blue'),
                        colored('{:.4f}'.format(output.distillation_loss.item()), 'red'),
                        session.dendrite,
                        session.axon)

                if (global_step+1) % config.neuron.log_interval == 0:	
                    log_all(session, history); history = [] # Log batch history.

                tensorboard.add_scalar('Rloss', output.remote_target_loss.item(), global_step)
                tensorboard.add_scalar('Lloss', output.local_target_loss.item(), global_step)
                tensorboard.add_scalar('Dloss', output.distillation_loss.item(), global_step)

                # ---- Update State ----
                batch_weights = torch.mean(output.weights, axis = 0) # Average over batch.
                row_weights = (1 - 0.03) * row_weights + 0.03 * batch_weights # Moving avg update.
                row_weights = F.normalize(row_weights, p = 1, dim = 0) # Ensure normalization.

                if (global_step + 1) % config.neuron.sync_interval == 0:
                    # ---- Sync Metagraph State ----
                    logger.info('Emitting with weights {}', row_weights.tolist())
                    session.metagraph.emit( row_weights, wait_for_inclusion = True ) # Sets my row-weights on the chain.
                    session.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                    row_weights = session.metagraph.W[ 0, :]

                    # ---- Update Axon Priority ----
                    col_weights = session.metagraph.W[:,0]
                    session.axon.set_priority( session.metagraph.neurons, col_weights ) # Sets the nucleus-backend request priority.

                local_step += 1
                global_step += 1
                torch.cuda.empty_cache()

            # --- Catch Errors during training ----
            except Exception as e:
                logger.error('Exception in training script with error: {}', e)
                logger.info(traceback.print_exc())
                logger.info('Continuing to train.')

        return output, global_step

    epoch = -1
    global_step = 0
    best_train_loss = math.inf
    while True:
        epoch += 1

        # ---- Train Model ----
        output, global_step = train(epoch, global_step)
        scheduler.step()

        # Save best loss and model
        if output:
            train_loss = output.local_target_loss

            if train_loss < best_train_loss:
                best_train_loss = train_loss # update best train loss
                logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/model.torch'.format(epoch, best_train_loss, config.neuron.trial_path))
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': best_train_loss},"{}/model.torch".format(config.neuron.trial_path))
                tensorboard.add_scalar('Train loss', train_loss, global_step)


if __name__ == "__main__":
    # ---- Load config ----
    parser = argparse.ArgumentParser(); add_args(parser)
    config = Config.load(parser); check_config(config)
    logger.info(Config.toString(config))

    # ---- Build Session ----
    session = bittensor.init(config)

    # ---- Start Neuron ----
    with session:
        main(config, session)

