import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def print_ycho(msg):
    print(f"안녕하세요? 조영주입니다. {msg}")

class Analyzer:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.optimize import curve_fit
    def __init__(self, name):
        # name of this class. normally set with run number, e.g. 'run1'
        self.name = name 
        
        ## Description about variables
        # 1. Signal datas
        # shape of signal datas will be (number of detuned, number of iterations, number of shots)
        # variables starts with abs will be datas about inCell absorption
        # variables starts with pfc will be datas about power fluctuation compensation
        # variables starts with flu will be datas about florescence at MOT chamber
        
        # self.parameter : Import parameter file

        # self.timedata : contains time data. The unit is µs.
        # self.abs_rawdata : contains rawdata from inCell absorption probe beam photodiode
        # self.pfc_rawdata : contains rawdata from inCell absorption probe beam power fluctuation monitoring photodiode
        # self.flu_rawdata : contains rawdata from fluorescence PMT
        
        # self.freq_set : contains frequencies set by system
        # self.freq_meter : contains  frequencies measured by wavemeter
        
        # self.iteration : number of repetition for run
        # self.shot_num : number of shots for each frequency & iteration
        # self.base_lim : data limit for baseline determine
        
        # # the variables below will be used for setting x limit and y limit for graphs
        # self.time_min : in second
        # self.time_max : in second
        # self.abs_maxamp : in a.u.
        # self.abs_minamp : in a.u.
        
    # Data loading function
    def load_data(self, filename, filepath, resultpath, freq_ref, freq_diff_check = False, iteration = 0, shot_num = 1, freq_double = False, flu_correction = 0, abs_which_shot = 0, flu_which_shot = 0):
        ## Description about input variables
        # filename : name of the file
        # filepath : path that will be used when loading datas
        # resultspath : path that will be used to save figures
        # freq_ref : reference frequency for the x axis in figures shape is [name of certain transition line, absolute frequency in THz]
        # freq_diff_check : plot the difference between the set frequency and the frequency wavemeter measured
        # iteration : number of iteration
        # shot_num : number of shots for each frequency in each iteration
        # freq_double : choose whether doubling the frequencies in data (for changing IR freq to UV freq)
        # flu_correction : this is correction of frequency calculated by spectrum of fluorescence signal
        # abs_which_shot : choose absorption signal to use for calculation. default value is 0 to disable the feature
        # flu_which_shot : choose fluorescence signal to use for calculation. default value is 0 to disable the feature
        
        # setting variables
        run_info = pd.read_csv(f'{filepath}{filename}_info.csv').values[0]
        if iteration == 0:
            self.iteration = int(run_info[3])
        else:
            self.iteration = iteration
        if shot_num == 1:
            self.shot_num = int(run_info[1])
        else:
            self.shot_num = shot_num
        print(f'iteration number : {self.iteration} & reptition number : {self.shot_num}')
        self.filepath = filepath
        self.resultpath = resultpath
        self.fname = filename
        self.freq_ref = freq_ref
        
        #Loading frequency data
        print(f'{self.fname} : Loading wavemeter data', end = '\r')
        temp_wav_dat = pd.read_csv(f'{filepath}{filename}_0_wavemeter_data.csv')
        self.freq_set = np.array(temp_wav_dat.iloc[:,0])

        print(f'{self.fname} : Loading detuning datas from {self.freq_set[0]:.6f}, to {self.freq_set[-1]:.6f}')
        
        # Setting array structure..
        temp_csv = pd.read_csv(f'{filepath}{filename}_0_{self.freq_set[0]:.6f}.csv', skiprows = [0])
        sig_len = len(temp_csv.iloc[:, 0].values)
        self.timedata = np.empty([self.iteration, len(self.freq_set), self.shot_num, sig_len]) # unit : µs
        self.abs_rawdata = np.empty([self.iteration, len(self.freq_set), self.shot_num, sig_len]) # unit : V
        self.pfc_rawdata = np.empty([self.iteration, len(self.freq_set), self.shot_num, sig_len]) # unit : V
        self.flu_rawdata = np.empty([self.iteration, len(self.freq_set), self.shot_num, sig_len]) # unit : V
        self.freq_meter = np.empty([self.iteration, len(self.freq_set), self.shot_num]) # unit : THz
        
        ## Loading signal data
        for i in range(self.iteration) :
            print(f'{self.fname} : Loading the data of {i}th iteration')
            temp_wav_dat = pd.read_csv(f'{filepath}{filename}_{i}_wavemeter_data.csv')
            self.freq_meter[i][:][:] = np.moveaxis(np.array(temp_wav_dat.iloc[:,1:self.shot_num+1]).T,0,1)
            for j in range(len(self.freq_set)) :
                freq = f'{self.freq_set[j]:.6f}' # To fix the precision as 6
                print(f'{self.fname} : Loading {freq} MHz data.........', end = '\r')
                temp_csv = pd.read_csv(f'{filepath}{filename}_{i}_{freq}.csv', skiprows = [0])
                for s in range(self.shot_num):
                    # Add to variables
                    self.timedata[i][j][s][:] = np.array(temp_csv.iloc[:, 2 * s]) * 1e6 # unit : µs
                    self.abs_rawdata[i][j][s][:] = np.array(temp_csv.iloc[:,2 * s + 1]) # unit : V
                    self.pfc_rawdata[i][j][s][:] = np.array(temp_csv.iloc[:, 4 * self.shot_num + 2 * s + 1]) # unit : V
                    self.flu_rawdata[i][j][s][:] = np.array(temp_csv.iloc[:, 2 * self.shot_num + 2 * s + 1]) # unit : V
                print(f'{self.fname} : Loading {i} MHz data finish!', end = '\r')
        
        if freq_double == True:
            self.freq_set *= 2
            self.freq_meter = [
                [
                    i * 2 for i in j
                ] for j in self.freq_meter
            ]
        ## Add parameter datas
        print(f'{self.fname} : Adding parameter datas..', end = '\r')
        self.parameter = pd.read_csv(f'{filepath}{filename}_parameter.csv', header = None).dropna(axis = 1)
        temp = self.parameter.values
        ## parameter in texts..
        self.parameter_text = [
            temp[0][i] + ' : ' + str(temp[1][i])
            for i in range(len(temp[0]))
            ]

        ## making detunelists
        print(f'{self.fname} : Making detunelists..', end = '\r')
        self.com_detunelist = (np.array(self.freq_set.copy()) - self.freq_ref[1])*1e6
        self.com_detunelist = np.rint(self.com_detunelist)
        self.met_detunes = (np.array(self.freq_meter.copy()) - self.freq_ref[1])*1e6
        self.met_detunes = np.rint(self.met_detunes)
        
        ## Draw difference bewteen set frequencies & actual frequencies
        if freq_diff_check == True:
            self.freq_fig, self.freq_ax = plt.subplots(figsize = (12,6))
            for i in range(len(self.freq_meter)):
                freq = self.freq_meter[i]
                for j in range(self.shot_num):
                    self.freq_ax.plot(self.com_detunelist, (self.freq_set - np.moveaxis(freq, 0, 1)[j]) * 1e6, label = f'{i}th iteration {j}th shot')
            self.freq_ax.set_xlabel(f'Detune from {self.freq_ref[0]} (MHz)', size = 14)
            self.freq_ax.set_ylabel(f'Detune from set frequency (MHz)', size = 14)
            self.freq_ax.grid()
            self.freq_ax.legend()
            self.freq_fig.suptitle(f'Difference between set frequency & actual frequency measured by WS8', size = 18)
            self.freq_fig.tight_layout()
            self.freq_fig.savefig(f'{self.resultpath}{self.name}_Frequencies.jpg', dpi = 300)
            plt.close()

        
        ## making detunelists by using fluorescence signal correction
        self.flu_correction = flu_correction
        self.flu_detunes = np.rint(self.met_detunes.copy() + self.flu_correction)
        self.flu_pr = [0, self.flu_correction] # this will be used in guessing parameters when fitting spectrum data
        
        print(f'{self.fname} : Making some constans for figures...', end = '\r')
        self.time_min = round(self.timedata[0][0][0].min()*1e-3)*1e3 # in µs
        self.time_interval = self.timedata[0][0][0][1] - self.timedata[0][0][0][0] # in µs
        self.time_max = round(self.timedata[0][0][0].max()*1e-3)*1e3 # in µs
        self.abs_maxamp = np.array([[np.array([np.array(i).max() for i in j]).max() for j in k] for k in self.abs_rawdata]).max()
        self.abs_minamp = np.array([[np.array([np.array(i).min() for i in j]).min() for j in k] for k in self.abs_rawdata]).min()
        self.flu_maxamp = np.array([[np.array([np.array(i).max() for i in j]).max() for j in k] for k in self.flu_rawdata]).max()
        self.flu_minamp = np.array([[np.array([np.array(i).min() for i in j]).min() for j in k] for k in self.flu_rawdata]).min()
        
        print(f'{self.fname} : Finished loading datas!!!')

    def calc_data(self, base_lims, sum_lims, PFC = False):
        # Setting input variables...
        self.abs_base_lim = base_lims[0]
        self.flu_base_lim = base_lims[1]
        self.abs_sumlim = sum_lims[0]
        self.flu_sumlim = sum_lims[1]
        self.tot_shot_num = self.abs_rawdata.shape[0] * self.abs_rawdata.shape[1] * self.abs_rawdata.shape[2]

        #calculating datas
        print(f'{self.fname} : Caculating Frequency datas')
        self.met_detunes_mean = np.moveaxis(self.met_detunes, 0, 1) \
            .reshape(self.met_detunes.shape[1], self.met_detunes.shape[0] * self.met_detunes.shape[2]) \
                .mean(axis = 1)
        self.met_detunes_std = np.moveaxis(self.met_detunes, 0, 1) \
            .reshape(self.met_detunes.shape[1], self.met_detunes.shape[0] * self.met_detunes.shape[2]) \
                .std(axis = 1)
        self.met_detunes_ste = self.met_detunes_std.copy() / np.sqrt(self.met_detunes.shape[0] * self.met_detunes.shape[2])
        print(f'{self.fname} : Caculating absorption datas')
        #baseline calculation and subtraction
        print(f'Absorption signal background signal. \n{self.parameter.values[0][6]} is {self.parameter.values[1][6]}. \nThis message is just for check.')

        # removing background noise
        self.abs_bgreducted = self.abs_rawdata.copy() - np.ones(self.abs_rawdata.shape) * float(self.parameter.iloc[1,5]) * 1e-3
        # calculating baseline datas
        self.abs_base_list = self.abs_bgreducted[:,:,:,0:self.abs_base_lim].mean(axis=3, keepdims = True)
        if PFC == True:
            print(f'Power fluctuation compensation. \n{self.parameter.values[0][7]} is {self.parameter.values[1][7]}. \nThis message is just for check.')
            self.pfc_bgreducted = self.pfc_rawdata.copy() - np.ones(self.pfc_rawdata.shape) * float(self.parameter.iloc[1,7]) * 1e-3
            self.pfc_base_list = self.pfc_bgreducted[:,:,:,0:self.abs_base_lim].mean(axis=3, keepdims = True)
            ## scale to match with absorption signal
            self.pfc_scaled = self.pfc_bgreducted / self.pfc_base_list * self.abs_base_list
            # change to positive data
            self.abs_pos_list = (- self.abs_bgreducted + self.pfc_scaled) / self.abs_base_list
        else:
            # change to positive data
            self.abs_pos_list = 1 - self.abs_bgreducted  / self.abs_base_list
        ## Change to absorption vs detuning
        # calculated absorption signal data
        self.abs_amount_list = self.abs_pos_list[:,:,:,self.abs_sumlim[0]:self.abs_sumlim[1]].sum(axis=3)
        # self.abs_amount_list = self.abs_pos_list[:,:,:,self.abs_sumlim].sum(axis=3)
        self.abs_spectrum_data = np.moveaxis(self.abs_amount_list, 0, 1) \
            .reshape(self.abs_amount_list.shape[1], self.abs_amount_list.shape[0] * self.abs_amount_list.shape[2]) \
                .mean(axis = 1)
        self.abs_std = np.moveaxis(self.abs_amount_list, 0, 1) \
            .reshape(self.abs_amount_list.shape[1], self.abs_amount_list.shape[0] * self.abs_amount_list.shape[2]) \
                .std(axis = 1)
        self.abs_ste = self.abs_std.copy() / np.sqrt(self.abs_amount_list.shape[0] * self.abs_amount_list.shape[2])

        print(f'{self.fname} : Caculating fluorescence datas', end = '\r', flush = True)
        #data analysis
        #baseline calculation and subtraction
        print(f'fluorescence signal background signal. \n{self.parameter.values[0][6]} is {self.parameter.values[1][6]}. \nThis message is just for check.')

        # Normalization
        # calculating baseline datas
        self.flu_base_list = self.flu_rawdata[:,:,:,self.flu_base_lim:].mean(axis=3, keepdims = True) 
        # change to positive data
        self.flu_pos_list = self.flu_rawdata.copy()  / self.flu_base_list - 1
        # self.flu_pos_list = self.flu_rawdata.copy() - self.flu_base_list
        ## Change to fluorescence vs detuning
        # calculated fluorescence signal data
        self.flu_amount_list = self.flu_pos_list[:,:,:,self.flu_sumlim[0]:self.flu_sumlim[1]].sum(axis=3)
        self.flu_spectrum_data = np.moveaxis(self.flu_amount_list, 0, 1) \
            .reshape(self.flu_amount_list.shape[1], self.flu_amount_list.shape[0] * self.flu_amount_list.shape[2]) \
                .mean(axis = 1)
        self.flu_std = np.moveaxis(self.flu_amount_list, 0, 1) \
            .reshape(self.flu_amount_list.shape[1], self.flu_amount_list.shape[0] * self.flu_amount_list.shape[2]) \
                .std(axis = 1)
        self.flu_ste = self.flu_std.copy() / np.sqrt(self.flu_amount_list.shape[0] * self.flu_amount_list.shape[2])

        print(f'{self.fname} : Caculating fluorescence datas', end = '\r', flush = True)
        
        print(f'Data calculation end!               ')

    def baseline_test(self, for_all = False):
        timedat = self.timedata[0][0][0]
        # Generate random 10 idx
        randarr = np.random.randint(0,self.tot_shot_num,10)
        if for_all == True:
            randarr = np.array(range(self.tot_shot_num), dtype = int)
        print(f'{self.fname} : Drawing baseline graphs...', end = '\r')
        fig, ax = plt.subplots(2, 2, figsize = (24, 12))
        flat_abs = self.abs_rawdata.reshape(self.tot_shot_num, self.abs_rawdata.shape[3])
        flat_flu = self.flu_rawdata.reshape(self.tot_shot_num, self.flu_rawdata.shape[3])
        flat_flu_pos_list = self.flu_pos_list.reshape(self.tot_shot_num, self.flu_rawdata.shape[3])
        for idx in randarr :
            ax[0,0].plot(timedat[0:self.abs_base_lim], flat_abs[idx][0:self.abs_base_lim])
            ax[1,0].plot(timedat[0:self.abs_base_lim+1], flat_abs[idx][0:self.abs_base_lim+1])
        for idx in randarr :
            ax[0,1].plot(timedat, flat_flu[idx])
            ax[1,1].plot(timedat, flat_flu_pos_list[idx])
        for i in range(2):
            ax[0,0].set_xlim(0,self.abs_base_lim * self.time_interval)
            ax[0,0].set_title(f'To {self.abs_base_lim  * self.time_interval} µs', size = 15)
            ax[1,0].set_xlim(0,self.abs_base_lim * self.time_interval + 1)
            ax[1,0].set_title(f'To {(self.abs_base_lim + 1)  * self.time_interval} µs', size = 15)
            ax[0,1].set_xlim(0,timedat.max())
            ax[0,1].set_title(f'Raw data', size = 15)
            ax[1,1].set_xlim(0,timedat.max())
            ax[1,1].set_title(f'Divided by baseline', size = 15)
        for j in range(2):
            for i in range(2):
                ax[i,j].set_xlabel('Time (µs)')
                ax[i,j].set_ylabel('Amplitude (V)')
                ax[i,j].grid()

        fig.suptitle("Validation for baseline range absorption / fluorescence", size = 25)
        fig.tight_layout()
        fig.savefig(f'{self.resultpath}{self.name}_Validation for baseline range.jpg', dpi = 300)
        plt.close('all')
        print(f'{self.fname} : Finished drawing baseline!')
    
    def sumline_test(self, for_all = True):
        timedat = self.timedata[0][0][0]
        # Generate random 10 idx
        randarr = np.random.randint(0,self.tot_shot_num,10)
        if for_all == True:
            randarr = np.array(range(self.tot_shot_num), dtype = int)
        print(f'{self.fname} : Drawing sumline graphs...', end = '\r')
        fig, ax = plt.subplots(2, 2, figsize = (24, 12))
        flat_abs = self.abs_pos_list.reshape(self.tot_shot_num, self.abs_pos_list.shape[3])
        flat_flu = self.flu_pos_list.reshape(self.tot_shot_num, self.flu_pos_list.shape[3])
        for idx in randarr :
            ax[0,0].plot(timedat[self.abs_sumlim[0] - 10 : self.abs_sumlim[1]], flat_abs[idx][self.abs_sumlim[0] - 10 : self.abs_sumlim[1]])
            ax[1,0].plot(timedat[self.abs_sumlim[0] : self.abs_sumlim[1]], flat_abs[idx][self.abs_sumlim[0] : self.abs_sumlim[1]])
        for idx in randarr :
            ax[0,1].plot(timedat[self.flu_sumlim[0] - 10 : self.flu_sumlim[1]], flat_flu[idx][self.flu_sumlim[0] - 10 : self.flu_sumlim[1]])
            ax[1,1].plot(timedat[self.flu_sumlim[0] : self.flu_sumlim[1]], flat_flu[idx][self.flu_sumlim[0] : self.flu_sumlim[1]])
        for i in range(2):
            # ax[0,i].set_xlim((self.abs_sumlim[0]- 1) * self.time_interval)
            # ax[1,i].set_xlim(self.abs_sumlim[0] * self.time_interval)
            ax[0,i].set_title(f'From {(self.abs_sumlim[0] - 10)  * self.time_interval} µs', size = 15)
            ax[1,i].set_title(f'From {self.abs_sumlim[0]  * self.time_interval} µs', size = 15)
        for j in range(2):
            for i in range(2):
                ax[i,j].set_xlabel('Time (µs)')
                ax[i,j].set_ylabel('Amplitude (V)')
                ax[i,j].grid()

        fig.suptitle("Validation for summation range absorption / fluorescence", size = 25)
        fig.tight_layout()
        fig.savefig(f'{self.resultpath}{self.name}_Validation for summation range.jpg', dpi = 300)
        plt.close('all')
        print(f'{self.fname} : Finished drawing sumline test!')
        
    def pfc_test(self, for_all = True):
        randarr = np.random.randint(0, self.tot_shot_num, 10)
        if for_all == True:
            randarr = np.array(range(self.tot_shot_num), dtype = int)
        print(f'{self.fname} : Drawing pfc graphs...', end = '\r')
        flat_dat = self.abs_pos_list.reshape(self.tot_shot_num, self.abs_pos_list.shape[3])
        fig, ax = plt.subplots(figsize = (12,6))
        for idx in randarr:
            ax.plot(self.timedata[0][0][0] * 1e-3, flat_dat[idx])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage')
        ax.grid()
        fig.suptitle(f'Validation for Power Fluctuation Compensation')
        fig.savefig(f'{self.resultpath}{self.name}_Validation for compensation.jpg', dpi = 300)
        plt.close('all')
        print(f'{self.fname} : Finished drawing pfc test!')
        
    def draw_intermediate(self):
        fig_intm, ax_intm = plt.subplots(1, 2, figsize = (12, 6))
        for i in range(len(self.abs_rawdata)):
            for j in range(len(self.abs_rawdata[i])):
                for k in self.abs_rawdata[i][j] : ax_intm[0].plot(self.timedata[0][0][0], k)
                for k in self.abs_pos_list[i][j] : ax_intm[1].plot(self.timedata[0][0][0], k)
        
        for i in range(2) :
            ax_intm[i].set_xlabel('Time (µs)', size = 14)
            ax_intm[i].set_xlim(self.time_min, self.time_max)
            ax_intm[i].set_ylabel('Amplitude (V)', size = 14)
        
        fig_intm.suptitle('Intermediate graphs', size = 18)
        ax_intm[0].set_title('original absorption signals', size = 15)
        ax_intm[1].set_title('absorption / baseline signal', size = 15)
        fig_intm.tight_layout()
        fig_intm.savefig(f'{self.resultpath}{self.name}_Intermediate graphs(absorption).jpg', dpi = 300)
        
    def draw_graph(self, detune_selection = 'COM', which = 'absorption'):
        #x axis detune from given frequency reference when loading data
        if detune_selection == 'COM':
            darr = self.met_detunes_mean
        elif detune_selection == 'absorption': # only can used when inCell absorption specturm fitting was finished
            darr = self.abs_detunes
        elif detune_selection == 'fluorescence':
            darr = self.flu_detunes
        else:
            print('Try "COM" or "absorption" or "fluorescence')

        if which == 'absorption':
            print(f'{self.fname} : Drawing Absorption spectrum', end = '\r', flush = True)
            #drawing graphs
            self.abs_fig, self.abs_ax = plt.subplots(figsize = (6,6))
            print(self.parameter_text[0][2:])
            # for i in self.parameter_text[0][2:]:
                # self.abs_ax.plot(darr[0],self.flu_spectrum_data.min(),'w--', label = i)
            
            self.abs_ax.errorbar(darr, self.abs_spectrum_data, xerr = self.met_detunes_ste, yerr = self.abs_ste, elinewidth = 0.7, capsize = 2, capthick = 0.7)
            self.abs_ax.set_xlim(darr[0], darr[-1])
            self.abs_ax.set_xlabel(f'detuning from {self.freq_ref[0]} (MHz)')
            self.abs_ax.set_ylabel('Absorption (a.u.)')
            self.abs_ax.legend()
            self.abs_fig.suptitle('Absorption spectrum', size = 15)
            self.abs_fig.tight_layout()
            self.abs_fig.savefig(f'{self.resultpath}{self.name}_Absoprtion spectrum.jpg', dpi = 300)
            plt.close('all')
            print(f'{self.fname} : Finished drawing absorption spectrum graph!!')
        elif which == 'fluorescence':
            print(f'{self.fname} : Drawing Fluorescence spectrum', end = '\r', flush = True)
            #drawing graphs
            self.flu_fig, self.flu_ax = plt.subplots(figsize = (6,6))
            print(self.parameter_text[0][2:])
            # for i in self.parameter_text[0][2:]:
                # self.flu_ax.plot(darr[0],self.flu_spectrum_data.min(),'w--', label = i)
            
            self.flu_ax.errorbar(darr, self.flu_spectrum_data, xerr = self.met_detunes_ste, yerr = self.flu_ste, elinewidth = 0.7, capsize = 2, capthick = 0.7)
            self.flu_ax.set_xlim(darr[0], darr[-1])
            self.flu_ax.set_xlabel(f'detuning from {self.freq_ref[0]} (MHz)')
            self.flu_ax.set_ylabel('fluorescence (a.u.)')
            self.flu_ax.legend()
            self.flu_fig.suptitle('fluorescence spectrum', size = 15)
            self.flu_fig.tight_layout()
            self.flu_fig.savefig(f'{self.resultpath}{self.name}_Fluorescence spectrum.jpg', dpi = 300)
            plt.close('all')
            print(f'{self.fname} : Finished drawing fluorescence spectrum!!')
        elif which == 'both':
            print(f'{self.fname} : Drawing Abs & Fl spectrum')
            #drawing graphs
            self.absflu_fig, self.absflu_ax = plt.subplots(figsize = (6,6))
            color = 'tab:blue'
            self.absflu_ax.errorbar(darr, self.abs_spectrum_data, self.abs_ste, elinewidth = 0.7, capsize = 2, capthick = 0.7, color = color)
            self.absflu_ax.tick_params(axis='y', labelcolor=color)
            self.absflu_ax.set_ylabel('Absorption (a.u.)', size = 14, color = color)
            
            self.absflu_sec_ax = self.absflu_ax.twinx()  # instantiate a secon   axes that shares the same x-axis
            color = 'tab:red'
            self.absflu_sec_ax.set_ylabel('Main Cell temperature (K)', color=color)  # we already handled the x-label with ax1
            self.absflu_sec_ax.errorbar(darr, self.flu_spectrum_data, self.flu_ste , elinewidth = 0.7, capsize = 2, capthick = 0.7, color = color)
            self.absflu_sec_ax.tick_params(axis='y', labelcolor=color)
            self.absflu_sec_ax.set_ylabel('Fluorescence (a.u.)', size = 14, color = color)

            self.absflu_ax.set_xlim(darr[0], darr[-1])
            self.absflu_ax.set_xlabel(f'detuning from {self.freq_ref[0]} (MHz)')
            
            self.absflu_fig.suptitle('Abs & Fl Spectrum spectrum', size = 15)
            self.absflu_fig.tight_layout()
            self.absflu_fig.savefig(f'{self.resultpath}{self.name}_Whole Spectrum.jpg', dpi = 300)
            plt.close('all')
            print(f'{self.fname} : Finished drawing absorption / fluorescence spectrum!!')
        else:
            print('Try "absorption", "fluorescence" or "both"')

    def timetrace_map(self):
        #preparing datas into 2D
        abs_2D = self.abs_pos_list.mean(axis = 0).mean(axis = 1)
        # flu_2D = self.flu_pos_list.mean(axis = 0).mean(axis = 1).reshape(int(self.tot_shot_num / self.shot_num), self.flu_pos_list.shape[3])
        flu_2D = self.flu_pos_list.mean(axis = 0).mean(axis = 1)
        
        #Drawing 2D map
        self.abs_fl_map_fig, self.abs_fl_map_ax = plt.subplots(1,2, figsize=(12,6))
        self.abs_map = self.abs_fl_map_ax[0].imshow(abs_2D \
            , cmap = 'RdYlBu', extent = (self.timedata[0][0][0][0]*1e-3 , self.timedata[0][0][0][-1]*1e-3, self.met_detunes_mean[0], self.met_detunes_mean[-1]) \
            , interpolation='nearest', aspect='auto')
        self.flu_map = self.abs_fl_map_ax[1].imshow(flu_2D \
            , cmap = 'RdYlBu', extent = (self.timedata[0][0][0][0]*1e-3 , self.timedata[0][0][0][-1]*1e-3, self.met_detunes_mean[0], self.met_detunes_mean[-1]) \
            , interpolation='nearest', aspect='auto')
        self.abs_fl_map_fig.colorbar(self.abs_map, ax = self.abs_fl_map_ax[0])
        self.abs_fl_map_fig.colorbar(self.flu_map, ax = self.abs_fl_map_ax[1])
        self.abs_fl_map_fig.suptitle('Absorption & Fluorescence 2D map', fontsize = 18)
        for i in range(2):
            self.abs_fl_map_ax[i].set_xlabel("Time (ms)", size = 14)
            self.abs_fl_map_ax[i].set_ylabel(f"Detune from {self.freq_ref[0]}", size = 14)
            self.abs_fl_map_ax[i].set_xlim(self.timedata[0][0][0][1]*1e-3, self.timedata[0][0][0][-1]*1e-3)
            self.abs_fl_map_ax[i].set_ylim(self.met_detunes_mean[0], self.met_detunes_mean[-1])
        self.abs_fl_map_fig.tight_layout()
        self.abs_fl_map_fig.savefig(self.resultpath + f'{self.name}_2d map.jpg', dpi = 300)


    #make individual graph
    def draw_individual_graph(self, whichdata, which_iteration, show_params = True):
        filepath = self.filepath
        time = self.timedata[0][0][0] # unit : µs
        
        for i in range(len(self.abs_rawdata)):
            detune = self.met_detunes
            print(f'{self.fname} : making {detune} MHz graph', end = '\r')
            #loading data
            if whichdata == 'Absorption':
                volt_data = self.abs_rawdata[i] # unit : V
            elif whichdata == 'Fluorescence':
                volt_data = self.flu_rawdata[i]

            params = self.parameter[i]
            params_text = [i + ' : ' + str(params[i].iloc[0]) for i in params]
            
            #setting figures
            fig, ax = plt.subplots(figsize = (6, 6))
            #whole graph drawing
            for i in range(len(volt_data[which_iteration])):
                ax.plot(time, volt_data[which_iteration][i], label = f'{which_iteration}th data, {i}th shot')
            ax.set_xlabel('Time (µs)', size = 14)
            ax.set_ylabel('Amplitude (V)', size = 14)
            ax.set_xlim(self.time_min, self.time_max)
            ax.set_ylim(self.abs_minamp, self.abs_maxamp)
            ax.legend()
            #title setting
            fig.suptitle(f'$\Delta$ = {round(detune, 1)} MHz data graph', size = 18)
            fig.tight_layout()
            fig.savefig(f'{self.resultpath}/allgraph/{self.name}_Absorption signal ($\Delta$ = {round((detune-8098.7)*1e2, 1)} MHz, iteration = {which_iteration}).jpg', dpi = 300)
            plt.close('all')
        print(f'{self.fname} : Finished drawing graphs for single detune!!')
        
    #Do fit
    def fit(self, target_info, fit_func, p0, which = 'absorption', detune_selection = 'absorption'):
        # target info is.. ['atomic mass of target', 'transition line frequency of first gaussian func']
        self.target_info = target_info
        detune = self.met_detunes_mean.copy()
        detune_cont = np.linspace(detune[0], detune[-1],1000)
        def temp_round(x):
            if x > 0.1 :
                return str(round(x,1))
            elif x > 1e-4 :
                return str(round(x,4)*1e3) + 'e-3'
            elif x > 1e-7 :
                return str(round(x,7)*1e6) + 'e-6'
            else :
                return str(round(x,10)*1e9) + 'e-9'
        if which == 'absorption':
            #데이터 준비
            abs_dat = self.abs_spectrum_data.copy()

            #ffffiting
            self.abs_pr, self.abs_cv, fit_detune, self.abs_fit_data = fit_func(detune, abs_dat, p0)
            u = 299792458 * self.abs_pr[0] / (self.freq_ref[1] * 1e6)
            self.abs_temp = (300 * self.target_info[1]) * (u / 2230)**2
            print(f'Fitting factors! \nTemp : {self.abs_temp} K')
            self.abs_detunes = detune.copy() - self.abs_pr[1]

            #text loc decision
            text_abs_yloc = np.array(abs_dat).max() - abs(np.array(abs_dat).max() - np.array(abs_dat).min()) / 8
            #Drawing graphs
            if detune_selection == 'COM':
                darr = self.met_detunes_mean
            elif detune_selection == 'absorption':
                darr = self.abs_detunes
            elif detune_selection == 'fluorescence':
                darr = self.flu_detunes
            else:
                print('Try "COM" or "absorption" or "fluorescence')
            darr_cont = np.linspace(darr[0], darr[-1], 1000)
            text_xloc = darr[-1] - abs(darr[0] - darr[-1])/4

            self.abs_fig, self.abs_ax = plt.subplots(figsize = (6,6))
            self.abs_ax.errorbar(darr, abs_dat, self.abs_ste, fmt = 'o')
            self.abs_ax.plot(darr_cont, self.abs_fit_data, 'r-')
            self.abs_ax.text(text_xloc, text_abs_yloc, f'T = {temp_round(self.abs_temp)} K', size = 15, bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5))
            self.abs_ax.set_xlabel(f'detuning from {self.freq_ref[0]} (MHz)', size = 14)
            self.abs_ax.set_ylabel('Relative Absorption (a.u.)', size = 14)
            self.abs_fig.suptitle(f'{self.target_info[0]} absorption spectrum w/ d-G fit', size = 18)
            self.abs_fig.tight_layout()
            self.abs_fig.savefig(self.resultpath + f'{self.name}_{self.target_info[0]} Absorption Spectrum Fitting.jpg', dpi = 300)
        
        elif which == 'fluorescence':
            #데이터 준비
            flu_dat = self.flu_spectrum_data.copy()

            #ffffiting
            self.flu_pr, self.flu_cv, fit_detune, self.flu_fit_data = fit_func(detune, flu_dat, p0)
            u = 299792458 * self.flu_pr[0] / (self.freq_ref[1] * 1e6)
            self.flu_temp = (300 * self.target_info[1]) * (u / 2230)**2
            print(f'Fitting factors! \nTemp : {self.flu_temp} K')
            self.flu_detunes = detune.copy() - self.flu_pr[1]

            #text loc decision
            text_flu_yloc = flu_dat.max() - abs(flu_dat.max() - flu_dat.min()) / 8
            #Drawing graphs
            if detune_selection == 'COM':
                darr = self.met_detunes_mean
            elif detune_selection == 'absorption':
                darr = self.flu_detunes
            elif detune_selection == 'fluorescence':
                darr = self.flu_detunes
            else:
                print('Try "COM" or "absorption" or "fluorescence')
            darr_cont = np.linspace(darr[0], darr[-1], 1000)
            text_xloc = darr[-1] - abs(darr[0] - darr[-1])/4
            self.flu_fig, self.flu_ax = plt.subplots(figsize = (6,6))
            self.flu_ax.errorbar(darr, flu_dat, self.flu_ste, fmt = 'o')
            self.flu_ax.plot(darr_cont, self.flu_fit_data, 'r-')
            self.flu_ax.text(text_xloc, text_flu_yloc, f'T = {temp_round(self.flu_temp)} K', size = 15, bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5))
            self.flu_ax.set_xlabel(f'detuning from {self.freq_ref[0]} (MHz)', size = 14)
            self.flu_ax.set_ylabel('Relative Fluorescence (a.u.)', size = 14)
            self.flu_fig.suptitle(f'{self.target_info[0]} Fluorescence spectrum w/ d-G fit', size = 18)
            # self.flu_fig.tight_layout()
            # self.flu_fig.savefig(self.resultpath + f'{self.name}_{self.target_info[0]} Fluorescence Spectrum Fitting.jpg', dpi = 300)

        elif which == 'both':
            #데이터 준비
            abs_dat = self.abs_spectrum_data.copy()
            abs_dat = abs_dat - np.array(abs_dat).min()
            flu_dat = self.flu_spectrum_data.copy() - 1

            #ffffiting
            from scipy.optimize import curve_fit
            if fit_func == one_peak:
                self.abs_pr, self.abs_cv = curve_fit(fit_func, detune, abs_dat, p0, epsfcn = 1e-16)
                self.abs_temp = (300 * 7) * (299792458 * self.abs_pr[0] * self.abs_pr[2] / (2230 * (446.80987e12 - self.abs_pr[1]*1e6) ) )**2
                print(f'"Absorption" Temp : {self.abs_temp} K, w0 : {self.abs_pr[1]} MHz')
                print(f'Amplification 1 : {self.abs_pr[2]*1e-10}e10')
                self.flu_pr, self.flu_cv = curve_fit(fit_func, detune, flu_dat, p0)
                self.flu_temp = (300 * 7) * (299792458 * self.flu_pr[0] * self.flu_pr[2] / (2230 * (446.80987e12 - self.flu_pr[1]*1e6) ) )**2
                print(f'"Fluorescence" Temp : {self.flu_temp} K, w0 : {self.flu_pr[1]} MHz')
                print(f'Amplification 1 : {self.flu_pr[2]*1e-10}e10')
            elif fit_func == two_peak:
                self.abs_pr, self.abs_cv = curve_fit(fit_func, detune, abs_dat, p0)
                self.abs_temp = (300 * 7) * (299792458 * self.abs_pr[0] * self.abs_pr[2] / (2230 * (446.80987e12 - self.abs_pr[1]*1e6) ) )**2
                print(f'"Absorption" Temp : {self.abs_temp} K, w0 : {self.abs_pr[1]} MHz, w1 : {self.abs_pr[3] / 1e12} THz')
                print(f'Amplification 1 : {self.abs_pr[2]*1e-10}e10, Amplification 2 : {self.abs_pr[4]*1e-10}e10')
                self.flu_pr, self.flu_cv = curve_fit(fit_func, detune, flu_dat, p0)
                self.flu_temp = (300 * 7) * (299792458 * self.flu_pr[0] * self.flu_pr[2] / (2230 * (446.80987e12 - self.flu_pr[1]*1e6) ) )**2
                print(f'"Fluorescence" Temp : {self.flu_temp} K, w0 : {self.flu_pr[1]} MHz, w1 : {self.flu_pr[3] / 1e12} THz')
                print(f'Amplification 1 : {self.flu_pr[2]*1e-10}e10, Amplification 2 : {self.flu_pr[4]*1e-10}e10')
            elif fit_func == two_peak_offset:
                self.abs_pr, self.abs_cv = curve_fit(fit_func, detune, abs_dat, p0)
                self.abs_temp = (300 * 7) * (299792458 * self.abs_pr[0] * self.abs_pr[2] / (2230 * (446.80987e12 - self.abs_pr[1]*1e6) ) )**2
                print(f'"Absorption" Temp : {self.abs_temp} K, w0 : {self.abs_pr[1]} MHz, w1 : {self.abs_pr[3] / 1e12} THz')
                print(f'Amplification 1 : {self.abs_pr[2]*1e-10}e10, Amplification 2 : {self.abs_pr[4]*1e-10}e10')
                print(f'Offset : {self.abs_pr[5]}')
                self.flu_pr, self.flu_cv = curve_fit(fit_func, detune, flu_dat, p0)
                self.flu_temp = (300 * 7) * (299792458 * self.flu_pr[0] * self.flu_pr[2] / (2230 * (446.80987e12 - self.flu_pr[1]*1e6) ) )**2
                print(f'"Fluorescence" Temp : {self.flu_temp} K, w0 : {self.flu_pr[1]} MHz, w1 : {self.flu_pr[3] / 1e12} THz')
                print(f'Amplification 1 : {self.flu_pr[2]*1e-10}e10, Amplification 2 : {self.flu_pr[4]*1e-10}e10')
                print(f'Offset : {self.flu_pr[5]}')
            else:
                print('Select fitting func "one_peak" or "two_peak" or "two_peak_offset"!')
            self.abs_detunes = detune.copy() - self.abs_pr[1]
            self.flu_detunes = detune.copy() - self.flu_pr[1]

            #text loc decision
            text_abs_yloc = np.array(abs_dat).max() - abs(np.array(abs_dat).max() - np.array(abs_dat).min()) / 8
            #Drawing graphs
            offset = self.abs_spectrum_data.min() - self.flu_spectrum_data.min()
            undervalue = self.flu_spectrum_data.min()
            ratio = (self.abs_spectrum_data.max() - self.abs_spectrum_data.min()) / (self.flu_spectrum_data.max() - self.flu_spectrum_data.min())
            def conversion(x):
                return (x - undervalue) * ratio + offset
            def sec_ax_func(x):
                return (x - offset) / ratio + undervalue
            
            if detune_selection == 'COM':
                darr = np.array([ iteration.mean(axis = 0) for iteration in met_detunes])
            elif detune_selection == 'absorption':
                darr = self.abs_detunes
            elif detune_selection == 'fluorescence':
                darr = self.flu_detunes
            else:
                print('Try "COM" or "absorption" or "fluorescence')
            darr_cont = np.linspace(darr[0], darr[-1],1000)
            text_xloc = darr[-1] - abs(darr[0] - darr[-1])/4

            absflu_fig, absflu_ax = plt.subplots(figsize = (6,6))
            absflu_ax.errorbar(darr, abs_dat, self.abs_ste, fmt = 'o')
            absflu_ax.errorbar(darr, conversion(flu_dat), self.flu_ste * ratio, fmt = 'o', elinewidth = 0.7, capsize = 2, capthick = 0.7)
            if fit_func == one_peak:
                absflu_ax.plot(darr_cont, fit_func(detune_cont, self.abs_pr[0],self.abs_pr[1],self.abs_pr[2]), 'r-')
                absflu_ax.plot(darr_cont, conversion(fit_func(detune_cont, self.flu_pr[0],self.flu_pr[1],self.flu_pr[2])), 'r-')
            elif fit_func == two_peak:
                absflu_ax.plot(darr_cont, fit_func(detune_cont, self.abs_pr[0],self.abs_pr[1],self.abs_pr[2], self.abs_pr[3], self.abs_pr[4]), 'r-')
                absflu_ax.plot(darr_cont, conversion(fit_func(detune_cont, self.flu_pr[0],self.flu_pr[1],self.flu_pr[2], self.flu_pr[3], self.flu_pr[4])), 'r-')
            elif fit_func == two_peak_offset:
                absflu_ax.plot(darr_cont, fit_func(detune_cont, self.abs_pr[0],self.abs_pr[1],self.abs_pr[2], self.abs_pr[3], self.abs_pr[4], self.abs_pr[5]), 'r-')
                absflu_ax.plot(darr_cont, conversion(fit_func(detune_cont, self.flu_pr[0],self.flu_pr[1],self.flu_pr[2], self.flu_pr[3], self.flu_pr[4], self.flu_pr[5])), 'r-')
            else:
                print("there's nothing to show")
            
            absflu_ax.plot(darr[0], abs_dat[0], 'white', label = f'abs : T = {temp_round(self.abs_temp)} K, $\Delta$ = {int(self.abs_pr[1])} MHz \n fl : T = {temp_round(self.flu_temp)} K,  $\Delta$ = {int(self.flu_pr[1])} MHz')
            absflu_ax.set_xlabel(f'detuning from {self.freq_ref[0]} (MHz)', size = 14)
            absflu_ax.set_ylabel('Relative Absorption (a.u.)', size = 14)
            absflu_ax.legend()
            sec_ax = absflu_ax.secondary_yaxis('right', functions=(sec_ax_func,sec_ax_func))
            sec_ax.set_ylabel('Relative Fluorescence (a.u.)', size = 14)
            
            absflu_fig.suptitle(f'7Li D2 line absorption spectrum w/ d-G fit', size = 18)
            absflu_fig.tight_layout()
            absflu_fig.savefig(self.resultpath + f'{self.name}_Absorption and Fluorescence D2 line Fitting.jpg', dpi = 300)
        else:
            print('Try "Absorption" or "fluorescence" or "both"')
    
    ## Draw Forward velocity
    def forward_vel_cal(self, which = 'COM'):
        #To calculate the speed, I used w' = w + kv, v = detune / k and used F = 2 D2 transition line for k
        if which == 'COM':
            self.forward_vel_arr = -self.met_detunes.copy() * 299792458 / freq_ref[1]
            self.high_forward_vel = self.forward_vel_arr[0]
            self.low_forward_vel = self.forward_vel_arr[-1]
        if which == 'fluorescence':
            self.forward_vel_arr = -(self.met_detunes.copy() - self.flu_pr[1]) * 299792458 / (freq_ref[1] - self.flu_pr[1])
            self.high_forward_vel = self.forward_vel_arr[0]
            self.low_forward_vel = self.forward_vel_arr[-1]
        if which == 'absorption':
            self.forward_vel_arr = -(self.abs_detunes.copy() - self.abs_pr[1])* 299792458 / (freq_ref[1] - self.abs_pr[1])
            self.high_forward_vel = self.forward_vel_arr[0]
            self.low_forward_vel = self.forward_vel_arr[-1]
        else:
            print('Try "COM" or "fluorescence" or "absorption"')

    def forward_vel(self, correction_factor, limits = [0,0,0,0], tof_len = 0.343, tof_line = False):
        #raw array preparation & some calculations
        def flu_mean(i):
            temp = i[0].copy()
            for j in range(len(i)-1):
                temp += i[j+1]
            temp /= len(i)
            return temp
        self.forward_raw_arr = np.array([flu_mean(i) for i in self.flu_rawdata])[:,self.abs_base_lim:]
        self.high_forward_vel_cor = self.high_forward_vel * correction_factor
        self.low_forward_vel_cor = self.low_forward_vel * correction_factor

        # self.forward_s_time = (self.base_lim)*self.time_interval*1e-3
        self.forward_s_time = 0
        self.forward_e_time = round(self.time_max*1e-3)-self.abs_base_lim*self.time_interval*1e-3
        # self.forward_e_time = round(self.time_max*1e-3)
        # self.forward_cont_time_arr = np.linspace(self.forward_s_time,self.forward_e_time, 1000)

        #Draw graph
        self.forward_vel_fig, self.forward_vel_ax = plt.subplots(figsize=(6,6))
        self.vel_map = self.forward_vel_ax.imshow(self.forward_raw_arr \
            , cmap = 'RdYlBu', extent = (self.forward_s_time, self.forward_e_time , self.low_forward_vel_cor, self.high_forward_vel_cor) \
            , interpolation='nearest', aspect='auto')
        self.forward_vel_fig.colorbar(self.vel_map, ax = self.forward_vel_ax)
        if tof_line == True:
            self.forward_tof_vel_arr = np.linspace(0, self.high_forward_vel_cor, 1000)
            self.forward_vel_ax.plot(tof_len / self.forward_tof_vel_arr * 1e3, self.forward_tof_vel_arr, '--')
        self.forward_vel_ax.set_xlabel("Time (ms)", size = 14)
        self.forward_vel_ax.set_ylabel("Forward velocity (m/s)", size = 14)
        self.forward_vel_fig.suptitle('Forward Velocity of 7Li w/ no slowing laser', fontsize = 18)
        if limits == [0,0,0,0]:
            self.forward_vel_ax.set_xlim(self.forward_s_time, self.forward_e_time)
            self.forward_vel_ax.set_ylim(self.low_forward_vel_cor, self.high_forward_vel_cor)
        else:
            self.forward_vel_ax.set_xlim(limits[0],limits[1])
            self.forward_vel_ax.set_ylim(limits[2],limits[3])
        
        self.forward_vel_fig.tight_layout()
        if tof_line == True:
            self.forward_vel_fig.savefig(self.resultpath + f'{self.name}_Forward velocity_tof.jpg', dpi = 300)
        else:
            self.forward_vel_fig.savefig(self.resultpath + f'{self.name}_Forward velocity.jpg', dpi = 300)

#fitting to the doppler broadening
# def one_peak (w, T, w0, amp):
#     #original cons :  c / (u * w0 * np.sqrt(np.pi))
#     u = 2230 * np.sqrt(T / (300 * 7))
#     c = 299792458
#     ff = amp * (c / (u * w0 * np.sqrt(np.pi))) * np.exp(- c**2 *((w - w0) / w0)**2 / u**2)
#     return ff


def one_peak (w, C, w0, amp):
    #original cons :  c / (u * w0 * np.sqrt(np.pi))
    ff = amp / (C * np.sqrt(np.pi)) * np.exp(- (w - w0)**2 / C**2)
    return ff
def fit_one_peak(detune, data, p0):
    # Define fit function
    def fit_func (w, C, w0, amp):
        return one_peak(w, C, w0, amp)
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, p0[0], p0[1], p0[2])
    return pr, cv, fit_detune, fit_data

def fit_two_peak_offset(detune, data, p0):
    def fit_func(w, C, w0, amp0, w1, amp1, offset):
        f0 = one_peak(w, C, w0, amp0)
        f1 = one_peak(w, C, w1, amp1)
        return f0 + f1 + offset
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5])
    return pr, cv, fit_detune, fit_data

def fit_P_1_fixed_freq(detune, data, p0):
    def fit_func(w, C, freq_offset, amp0, amp1, amp2, offset):
        f0 = one_peak(w, C, -129 + freq_offset, amp0)
        f1 = one_peak(w, C, 0 + freq_offset, amp1)
        f2 = one_peak(w, C, 110 + freq_offset, amp2)
        return f0 + f1 + f2 + offset
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, pr[0], pr[1], pr[2], pr[3], pr[4], pr[5])
    return pr, cv, fit_detune, fit_data
def fit_P_1_fixed_freq_EOM(detune, data, p0):
    def fit_func(w, C, freq_offset, amp0, amp1, amp2, amp3, amp4, offset):
        f0 = one_peak(w, C, -249 + freq_offset, amp0)
        f1 = one_peak(w, C, -129 + freq_offset, amp1)
        f2 = one_peak(w, C, 0 + freq_offset, amp2)
        f3 = one_peak(w, C, 110 + freq_offset, amp3)
        f4 = one_peak(w, C, 230 + freq_offset, amp4)
        return f0 + f1 + f2 + f3 + f4 + offset
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7])
    return pr, cv, fit_detune, fit_data

def fit_Q_11_R_12_fixed_freq(detune, data, p0):
    def fit_func(w, C, freq_offset, amp0, amp1, amp2, amp3, offset):
        f0 = one_peak(w, C, 46588 + freq_offset, amp0)
        f1 = one_peak(w, C, 46719 + freq_offset, amp1)
        f2 = one_peak(w, C, 46834 + freq_offset, amp1)
        f3 = one_peak(w, C, 46487 + freq_offset, amp1)
        return f0 + f1 + f2 + f3 + offset
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6])
    return pr, cv, fit_detune, fit_data
def fit_N2_lines_fixed_freq(detune, data, p0):
    def fit_func(w, C, freq_offset, amp0, amp1, amp2, amp3, amp4, amp5, offset):
        f0 = one_peak(w, C, -15503 + freq_offset, amp0)
        f1 = one_peak(w, C, -15485 + freq_offset, amp1)
        f2 = one_peak(w, C, -15447 + freq_offset, amp2)
        f3 = one_peak(w, C, -15359 + freq_offset, amp3)
        f4 = one_peak(w, C, -15318 + freq_offset, amp4)
        f5 = one_peak(w, C, -15228 + freq_offset, amp5)
        return f0 + f1 + f2 + f3 + f4 + f5 + offset
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8])
    return pr, cv, fit_detune, fit_data
def three_peak(w, C, w0, amp0, w1, amp1, w2, amp2):
    f0 = one_peak(w, C, w0, amp0)
    f1 = one_peak(w, C, w1, amp1)
    f2 = one_peak(w, C, w2, amp2)
    return f0 + f1 + f2
def fit_three_peak_offset(detune, data, p0):
    def fit_func(w, C, w0, amp0, w1, amp1, w2, amp2, offset):
        f0 = one_peak(w, C, w0, amp0)
        f1 = one_peak(w, C, w1, amp1)
        f2 = one_peak(w, C, w2, amp2)
        return f0 + f1 + f2 + offset
    pr, cv = curve_fit(fit_func, detune, data, p0)
    fit_detune = np.linspace(detune[0], detune[-1], 1000)
    fit_data = fit_func(fit_detune, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7])
    return pr, cv, fit_detune, fit_data
def isotope_25(w, C, freq_offset, amp0, amp1, offset):
    f0 = one_peak(w, C, 0 + freq_offset, amp0)
    f1 = one_peak(w, C, 1000 + freq_offset, amp1)
    return f0 + f1 + offset
