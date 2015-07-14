# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""

import multiprocessing
#import time

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


## This Task object is an example of a task that can be handed over to
## the function runMultiCore
#class Task(object):
#    def __init__(self, a, b):
#        self.a = a
#        self.b = b
#    def __call__(self):
#        time.sleep(0.1) # pretend to take some time to do the work
#        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)
#    def __str__(self):
#        return '%s * %s' % (self.a, self.b)


def runMultiCore(jobs, numberProcessors=multiprocessing.cpu_count()):
    """
    Run the tasks in jobs on numberProcessors cores.
    
    The tasks need to be an object with implemented __call__() method.
    
    For more details see: http://pymotw.com/2/multiprocessing/basics.html
    """
    
    assert( isinstance(jobs, list) )
    
    # Establish communication queues
    tasks        = multiprocessing.JoinableQueue()
    resultsQueue = multiprocessing.Queue()
    
    # Start consumers
#    num_consumers = multiprocessing.cpu_count()
    num_consumers = numberProcessors

    consumers = [ Consumer(tasks, resultsQueue) for i in range(num_consumers) ]
    
    for w in consumers:
        w.start()
    
    # Enqueue jobs
    for job in jobs:
        tasks.put(job)
    
    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()
    
    # collect all results and put them in a list
    results = list()
    while not resultsQueue.empty():
        results.append(resultsQueue.get())

    return results
