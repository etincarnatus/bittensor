# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

""" Template server.

Example:
    $ import neurons
    $ neurons.text.core_server.neuron().run()
"""

import bittensor
import os
from time import sleep
import traceback

from .nucleus_impl import server
from .run import serve
import copy
from torch import multiprocessing

class Process(multiprocessing.Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception

class neuron:
    r"""
    Creates a bittensor neuron that specializes in the serving. The template server miner
    serves a NLP model from huggingface on the bittensor network. By default, the model does 
    not train itself and thus requires less memory to run. 

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            lasthidden (:obj:bool, `optional`):
                lasthidden synapse control
            causallm (:obj:bool, `optional`):
                causallm synapse control
            causallmnext (:obj:bool, `optional`):
                causallmnext synapse control
            seq2seq (:obj:bittensor.metagraph, `optional`):
                seq2seq synapse control
            synapse_list (:obj:list of int, `optional`):
                

    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> server = bittensor.neuron.text.core_server.neuron(subtensor=subtensor)
            >>> server.run()
    """
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,
        lasthidden = None,
        causallm = None,
        causallmnext = None,
        seq2seq = None,
        synapse_list = None,
    ):
        if config == None: config = server.config()
        config = config; 

        if synapse_list != None:
            config.neuron.lasthidden = False
            config.neuron.causallm = False
            config.neuron.causallmnext = False
            config.neuron.seq2seq = False

            if bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE in synapse_list:
                config.neuron.lasthidden = True
            
            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM in synapse_list:
                config.neuron.causallm = True

            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT in synapse_list:
                config.neuron.causallmnext = True

            if bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ in synapse_list:
                config.neuron.seq2seq = True

        config.neuron.lasthidden = lasthidden if lasthidden != None else config.neuron.lasthidden
        config.neuron.causallm = causallm if causallm != None else config.neuron.causallm
        config.neuron.causallmnext = causallmnext if causallmnext is not None else config.neuron.causallmnext
        config.neuron.seq2seq = seq2seq if seq2seq != None else config.neuron.seq2seq

        self.check_config( config )
        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )

        self.model = server(config = config)
        self.config = config
        self.shared_subconfigs = self.create_shared_subconfigs(config)

        self.subtensor = subtensor
        self.wallet = wallet
        self.axon = axon
        self.metagraph = metagraph

    def run(self):
        if len(self.shared_subconfigs)==1:
            serve(
                self.config,
                self.model,
                subtensor = self.subtensor,
                wallet = self.wallet,
                axon = self.axon,
                metagraph = self.metagraph
                )
        else:
            multiprocessing.set_start_method('spawn', force=True)
            self.model.to(self.config.neuron.device)
            self.model.share_memory()

            processes = {}
            n = 0
            for shared_subconfig in self.shared_subconfigs:
                p = Process(target=serve, args=(shared_subconfig,
                                                self.model,
                                                self.subtensor,
                                                self.wallet,
                                                self.axon,
                                                self.metagraph
                                                )
                              )
                p.start()
                processes[n] = (p, shared_subconfig)
                n+=1
            
            while len(processes)>0:
                for n in list(processes):
                    (p, shared_subconfig) = processes[n]
                    sleep(1)
                    if not p.is_alive():
                        if p.exitcode is None:
                            for p_, task in processes.values(): # Terminate all child processes without waiting
                                p_.terminate()
                            raise ChildProcessError("Not started and not exited!")
                        elif p.exception: # Exception in child process
                            error, trace = p.exception
                            for p_, task in processes.values(): # Terminate all child processes without waiting
                                p_.terminate()
                            raise ChildProcessError(trace)
                        elif p.exitcode==0:
                            p.join()
                            del processes[n]

    @classmethod
    def config(cls):
        return server.config()

    @staticmethod
    def create_shared_subconfigs(config: 'bittensor.Config'):
        if config.wallet.shared_keys is not None:
            keylist = config.wallet.shared_keys.split(",")
            portlist = map(int,config.axon.shared_ports.split(","))
            shared_subconfigs = []
            for key,port in zip(keylist,portlist):
                coldkey,hotkey = key.split(":")
                subconfig = copy.deepcopy(config)
                subconfig.wallet.name = coldkey
                subconfig.wallet.hotkey = hotkey
                subconfig.axon.port = port
                shared_subconfigs.append(subconfig)
        else:
            shared_subconfigs = [config]
        return shared_subconfigs

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name), config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)
