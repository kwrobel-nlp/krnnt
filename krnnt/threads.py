import csv
import multiprocessing
import sys
import threading
from timeit import default_timer as timer


class LogTime(multiprocessing.Process):
    def __init__(self, queue_log):
        super(LogTime, self).__init__()
        self.queue_log = queue_log
        self.f=open('log2cores2workersWsort.csv','w', newline='')
        self.csv_writer = csv.writer(self.f)

    def run(self):
        while True:
            item = self.queue_log.get()
            if item is None:
                self.queue_log.task_done()
                break
            self.csv_writer.writerow(item)
            self.f.flush()
            self.queue_log.task_done()


class StdInThread(threading.Thread):
    def __init__(self, queue, queue_log=None):
        super(StdInThread, self).__init__()
        self.queue=queue
        self.queue_log = queue_log

    def log(self, desc):
        if self.queue_log:
            # print(self.name, timer(), desc, file=sys.stderr)
            self.queue_log.put([self.name, timer(), desc])

    def run(self):
        self.log('START')
        ss = []
        for i,line in enumerate(sys.stdin):
            # self.queue.put(line.strip())
            ss.append((i,line.strip()))


        #ss = sorted(ss, key=lambda sentence: sentence[1].count(' '))
        for s in ss:
            self.queue.put(s)

        self.queue.put(1)
        self.log('STOP')


class BatcherThread(threading.Thread):
    def __init__(self, input_queue, output_queue, batch_size, number_of_consumers, queue_log=None):
        super(BatcherThread, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.number_of_consumers=number_of_consumers
        self.queue_log = queue_log

    def log(self, desc):
        if self.queue_log:
            # print(self.name, timer(), desc)
            self.queue_log.put([self.name, timer(), desc])

    def run(self):
        # print(self.name, 'RUN', self.input_queue.qsize())
        self.log('START')
        batch = []
        while True:
            self.log('WORKING')
            item = self.input_queue.get()
            self.log('WAIT')
            if isinstance( item, int ):
                if batch:
                    pass
                    self.output_queue.put(batch)
                if item>1:
                    self.input_queue.put(item-1)
                else:
                    pass
                    self.output_queue.put(self.number_of_consumers)

                self.input_queue.task_done()
                break

            batch.append(item)

            if len(batch) == self.batch_size:
                self.log('PUT0')
                self.output_queue.put(batch)
                self.log('PUT1')
                batch = []
            self.input_queue.task_done()


        self.log('STOP')
        # print('batcher stop')