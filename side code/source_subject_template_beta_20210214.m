%%% Calculate the SSVEP template from the source subjects 
% Edwin Wong (chiman465@gmail.com)
% 2021-2-14
% clc;
clear all;
close all;

str_dir='..\data\dataset2\';
Fs=250;
% ch_used=[1:64];
ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2

latencyDelay = 0.13;                % latency
% butterworth filter
% 0.5~60Hz
bandpass=[7 100];
[b1,a1]=butter(4,[bandpass(1)/(Fs/2) bandpass(2)/(Fs/2)]);

%notch filter
Fo = 50;
Q = 35;
BW = (Fo/(Fs/2))/Q;

[notchB,notchA] = iircomb(Fs/Fo,BW,'notch');

pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
    0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
sti_f=[8.6:0.2:15.8,8.0 8.2 8.4];
n_sti=length(sti_f);                     % number of stimulus frequencies
[~,target_order]=sort(sti_f);
sti_f=sti_f(target_order);
pha_val=pha_val(target_order);

tic
for sn=1:70
    load([str_dir '\S' num2str(sn) '.mat']);
    eegdata=data.EEG;
    data1(:,:,:,:) = permute(eegdata,[1 2 4 3]);
    
    %  pre-stimulus period: 0.5 sec
    %  latency period: 0.13 sec
    eeg=data1(ch_used,floor(0.5*Fs+latencyDelay*Fs)+1:floor(0.5*Fs+latencyDelay*Fs)+2*Fs,:,:);
    
    [d1_,d2_,d3_,d4_]=size(eeg);
    d1=d3_;d2=d4_;d3=d1_;d4=d2_;
    no_of_class=d1;
    % d1: num of stimuli
    % d2: num of trials
    % d3: num of channels % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
    % d4: num of sampling points
    
    for i=1:1:d1
        for j=1:1:d2
            y0=reshape(eeg(:,:,i,j),d3,d4);
            y0 = filtfilt(notchB, notchA, y0.'); %notch
            y0 = y0.';
            for ch_no=1:d3
                % CAR
                y0(ch_no,:)=y0(ch_no,:)-mean(y0([1:ch_no-1,ch_no+1:end],:));
                y(ch_no,:)=filtfilt(b1,a1,y0(ch_no,:));
            end    
            SSVEPdata(:,:,j)=reshape(y,d3,d4,1);
        end
        mu_ssvep=mean(SSVEPdata,3);
%         mu_ssvep=mu_ssvep-mean(mu_ssvep,2)*ones(1,length(mu_ssvep));
%         mu_ssvep=mu_ssvep./(std(mu_ssvep')'*ones(1,length(mu_ssvep)));
        subj(sn).ssvep_template(:,:,i)=mu_ssvep;
        
        % ----- LST -----
        subj(sn).SSVEPdata(:, :, :, i)=SSVEPdata;
        % ----- LST -----
    end
    subj(sn).ssvep_template=subj(sn).ssvep_template(:,:,target_order);
    
    % ----- LST -----
    subj(sn).SSVEPdata = subj(sn).SSVEPdata(:,:,:,target_order);
    % ----- LST -----
    
    clear eeg data1    
    toc
end
filename=mfilename('fullpath');
save_name=['beta_subj_ssvep_template_' num2str(length(ch_used)) '.mat'];
save(save_name,'subj','bandpass','filename','ch_used');

