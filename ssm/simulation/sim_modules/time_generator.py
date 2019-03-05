from ssm.core.pchain import ProcessingModule
from ssm.core.pchain import ProcessingModule
import datetime

class TimeGenerator(ProcessingModule):
    def __init__(self,start_time,end_time,time_step=datetime.timedelta(seconds=.1)):
        super().__init__('Time')
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        self.n_calls = 0
    def configure(self,config):
        config['start_time'] = self.start_time
        config['end_time'] = self.end_time
        config['time_step'] = self.time_step
        config['run_duration'] = self.end_time - self.start_time
        config['n_frames'] = int(config['run_duration'].to_datetime().total_seconds()/self.time_step.total_seconds())
        self.n_calls = 0
    def run(self,frame):
        frame['time'] = self.start_time + self.time_step*self.n_calls
        frame['timestamp'] = (frame['time']-self.start_time).to_datetime().total_seconds()*1e9
        self.n_calls +=1
        return frame