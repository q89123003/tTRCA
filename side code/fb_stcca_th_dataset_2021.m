%%% subject-transfer in SSVEP-BCI %%%
% Evaluate the stcca performance on Tsinghua dataset

% Edwin Wong (chiman465@gmail.com)
% 2021-2-16

clear all;
close all;

% addpath('..\mytoolbox');
str_dir='..\data\dataset1\';
% str_dir='/data/2016_Tsinghua_SSVEP_database/';

Fs=250;
ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
num_of_trials=5;                     % number of calibration trials per stimulus from target subject   
num_of_harmonics=5;                  % for all cca
num_of_signal_templates=12 ;         % for multi-stimulus cca
num_of_signal_templates2=12;         % for multi-stimulus trca
num_of_signal_templates3=40;         % for subject transfer
f_idx=[1:40];
dataset_no=1;                        % 1:learning from the same dataset, 2:learning from another dataset

num_of_subbands=5;                   % for filter bank analysis
t_length0=0.2;
delta_t=0.1;
t_length=1.0;                         % Window length ([0.2:0.1:1.0])
temp_len=floor(t_length0*Fs);
enable_bit=[1 0 0 0 0];
% butterworth filter
% 0.5~60Hz
bandpass=[7 90];
[b1,a1]=butter(4,[bandpass(1)/(Fs/2) bandpass(2)/(Fs/2)]);
seed = RandStream('mt19937ar','Seed','shuffle');

pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
    0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
sti_f=[8:0.2:15.8];                 % stimulus frequency
n_sti=length(sti_f);                % number of stimulus frequencies
temp=reshape([1:40],8,5);
temp=temp';
target_order=temp(:)';

for k=1:num_of_subbands
    bandpass1(1)=8*k;
    bandpass1(2)=90;
    [b2(k,:),a2(k,:)] = cheby1(4,1,[bandpass1(1)/(Fs/2) bandpass1(2)/(Fs/2)],'bandpass');
end
FB_coef0=[1:num_of_subbands].^(-1.25)+0.25;     % Filter bank coefficients

% Load the source subjects' templates
if length(ch_used)==9
    if dataset_no==1
        load th_subj_ssvep_template_9.mat
        num_of_ssub=35;
    elseif dataset_no==2
        load beta_subj_ssvep_template_9.mat
        num_of_ssub=70;
    else
    end
elseif length(ch_used)==64
    if dataset_no==1
        load th_subj_ssvep_template_64.mat
        num_of_ssub=35;
    elseif dataset_no==2
        load beta_subj_ssvep_template_64.mat
        num_of_ssub=70;
    else
    end
else
end
source_data_len=size(subj(1).ssvep_template,2);
tic
for k=1:num_of_subbands
    for sn=1:num_of_ssub
        temp=[];
        ref=[];
        for m=1:40
            tmp=subj(sn).ssvep_template(:,:,m);
            
            % for LST
            tmp_ssvepdata = subj(sn).SSVEPdata(:, :, :, m);
            
            for ch_no=1:length(ch_used)
                tmp_sb(ch_no,:)=filtfilt(b2(k,:),a2(k,:),tmp(ch_no,:));
                
                % for LST
                tmp_sbssvepdata(ch_no,:, :)=filtfilt(b2(k,:),a2(k,:),squeeze(tmp_ssvepdata(ch_no, :, :)));
            end
            subj(sn).subband(k).ssvep_template(:,:,m)=tmp_sb;
            
            % for LST
           subj(sn).subband(k).SSVEPdata(:, :, :, m) = tmp_sbssvepdata;
            
            temp=[temp tmp_sb];
            ref0=ck_signal_nh(sti_f(m),Fs,pha_val(m),source_data_len,num_of_harmonics);
            ref=[ref ref0];
        end
        [W_x,W_y,r]=canoncorr(temp',ref');
        subj(sn).subband(k).sf=W_x(:,1);
        for m=1:40
            ssvep_temp=subj(sn).subband(k).ssvep_template(:,:,m);
            subj(sn).subband(k).filtered_ssvep_template(m,:)=W_x(:,1)'*ssvep_temp;      % source subject's spatially filtered SSVEP templates 
        end
    end
end
toc
sub_idx=[1:num_of_ssub];
for sn=1:35
    tic
    load(strcat(str_dir,'S',num2str(sn),'.mat'));
    
    eeg=data(ch_used,floor(0.5*Fs+0.14*Fs):floor(0.5*Fs+0.14*Fs)+4*Fs-1,:,:);
    
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
            for ch_no=1:d3
                % CAR
                y0(ch_no,:)=y0(ch_no,:)-mean(y0([1:ch_no-1,ch_no+1:end],:));
                y(ch_no,:)=filtfilt(b1,a1,y0(ch_no,:));
            end
            for sub_band=1:num_of_subbands
                for ch_no=1:d3
                    y_sb(ch_no,:)=filtfilt(b2(sub_band,:),a2(sub_band,:),y(ch_no,:));
                end
                subband_signal(sub_band).SSVEPdata(:,:,j,i)=reshape(y_sb,d3,d4,1,1);
            end
        end
    end
    
    clear eeg
    
    TW=t_length0:delta_t:t_length;           % Data length (or TW)
    TW_p=round(TW*Fs);
    n_run=d2;                                % number of used runs
    for sub_band=1:num_of_subbands
        subband_signal(sub_band).SSVEPdata=subband_signal(sub_band).SSVEPdata(:,:,:,target_order);
    end
    
    FB_coef=FB_coef0'*ones(1,n_sti);
    n_correct=zeros(length(TW),10);
    
    seq_0=zeros(d2,num_of_trials);
    for run=1:d2
        %  leave-one-run-out cross-validation
        if (num_of_trials==1)
            seq1=run;
        elseif (num_of_trials==d2-1)
            seq1=[1:n_run];
            seq1(run)=[];
        else
            % leave-one-run-out cross-validation
            isOK=0;
            while (isOK==0)
                seq=randperm(seed,d2);
                seq1=seq(1:num_of_trials);
                seq1=sort(seq1);
                if isempty(find(sum((seq1'*ones(1,d2)-seq_0').^2)==0))
                    isOK=1;
                end
            end
            
        end
        idx_traindata=seq1;
        idx_testdata=1:n_run;
        idx_testdata(seq1)=[];
        
        for i=1:no_of_class
            for k=1:num_of_subbands
                if length(idx_traindata)>1
                    subband_signal(k).signal_template(i,:,:)=mean(subband_signal(k).SSVEPdata(:,:,idx_traindata,i),3);
                else
                    subband_signal(k).signal_template(i,:,:)=subband_signal(k).SSVEPdata(:,:,idx_traindata,i);
                end
            end
        end
        
        % ----- LST ----- 
        source_idx=sub_idx;
        if dataset_no==1
            source_idx(sn)=[];
        end
        
        LST_templates = cell(length(TW_p), 1);
        LST_spatial_filters = cell(length(TW_p), 1);
        
         
        for i_tw = 1 : length(TW_p)
            sig_len=TW_p(i_tw);
            
            LST_templates{i_tw} = zeros(num_of_subbands, no_of_class, d3, sig_len);
            LST_spatial_filters{i_tw} = zeros(num_of_subbands, no_of_class, d3);
            
            for i_fb = 1 : num_of_subbands
                for i_class = 1 : no_of_class
                    tmp_target_trials = subband_signal(i_fb).SSVEPdata(:, 1:sig_len, idx_traindata, i_class);
                    tmp_target_mean = squeeze(mean(tmp_target_trials, 3));
                    Y = tmp_target_mean;

                    transferred_trials = zeros(d3, sig_len, 0);
                    source_trial_count = 0;
                    for i_source = 1 : length(source_idx)
                        tmp_source_trials = squeeze(subj(source_idx(i_source)).subband(i_fb).SSVEPdata(:,1:sig_len,:,i_class));
                        tmp_source_mean = squeeze(mean(tmp_source_trials, 3));

                        X = [ones(1, size(Y, 2)); tmp_source_mean];
                        b = Y * X.' / (X * X.');

                        for i_trial = 1 : size(tmp_source_trials, 3)
                            source_trial_count = source_trial_count + 1;
                            single_trial_eeg_tmp = squeeze(tmp_source_trials(:, :, i_trial));
                            X_trial = [ones(1, size(Y, 2)); single_trial_eeg_tmp];
                            transferred_trials(:, :, source_trial_count) = (b * X_trial);
                        end
                    end

                    transferred_trials = cat(3, tmp_target_trials, transferred_trials);
                    [w_tmp, ~] = trca(transferred_trials);
                    LST_templates{i_tw}(i_fb, i_class, :, :) = mean(transferred_trials, 3);
                    LST_spatial_filters{i_tw}(i_fb, i_class, :) = w_tmp(:,1);
                end
            end
        end
        % ----- LST -----
        
        for run_test=1:length(idx_testdata)
            
            for tw_length=1:length(TW)
                clear train_rawData test_rawData
                train_Y=[];clsY=[];
                sig_len=TW_p(tw_length);
                
                fprintf('eCCA and IT-CCA Processing TW %fs, No.crossvalidation %d \n',TW(tw_length),idx_testdata(run_test));
                
 
                for i=1:no_of_class
                                        
                    for sub_band=1:num_of_subbands
                        test_signal=subband_signal(sub_band).SSVEPdata(:,1:TW_p(tw_length),idx_testdata(run_test),i);
                        %                     test_signal=test_signal-mean(test_signal,2)*ones(1,length(test_signal));
                        %                     test_signal=test_signal./(std(test_signal')'*ones(1,length(test_signal)));
                        ssvep_template1=[];
                        ref_template1=[];
                        
            
                        for j=1:no_of_class
                            %% ms-etrca spatial filter
                            if ((i==1) && (j==1))
                                source_idx=sub_idx;
                                if dataset_no==1
                                    source_idx(sn)=[];
                                end
                                % individual spatial filter
                                target_ssvep=[];
                                target_ref=[];
                                for fn=1:length(f_idx)
                                    tmp1=reshape(subband_signal(sub_band).signal_template(f_idx(fn),:,1:sig_len),d3,sig_len);                                    ;
                                    ref1=ck_signal_nh(sti_f(f_idx(fn)),Fs,pha_val(f_idx(fn)),sig_len,num_of_harmonics);
                                    target_ssvep=[target_ssvep, tmp1];                                    
                                    target_ref=[target_ref ref1];
                                end
                                [W_x,W_y,r]=canoncorr(target_ssvep',target_ref');
                                subband_signal(sub_band).ind_Wx=W_x(:,1);
                                subband_signal(sub_band).ind_Wy=W_y(:,1);
                                
                                W_msTRCA(sub_band).val=[];
                                for my_j=1:no_of_class
                                    d0=floor(num_of_signal_templates2/2);
                                    if my_j<=d0
                                        template_st=1;
                                        template_ed=num_of_signal_templates2;
                                    elseif ((my_j>d0) && my_j<(d1-d0+1))
                                        template_st=my_j-d0;
                                        template_ed=my_j+(num_of_signal_templates2-d0-1);
                                    else
                                        template_st=(d1-num_of_signal_templates2+1);
                                        template_ed=d1;
                                    end
                                    template_seq=[template_st:template_ed];
                                    mstrca_X1=[];
                                    mstrca_X2=[];
                                    
                                    for n_temp=1:num_of_signal_templates2
                                        jj=template_seq(n_temp);
                                        trca_X2=[];
                                        trca_X1=zeros(d3,sig_len);
                                        template2=zeros(d3,sig_len);
                                        
                                        for tr=1:num_of_trials
                                            X0=reshape(subband_signal(sub_band).SSVEPdata(:,1:sig_len,idx_traindata(tr),jj),d3,sig_len);                                            
                                            X0=X0-mean(X0,2)*ones(1,size(X0,2));
                                           
                                            trca_X2=[trca_X2;X0'];
                                            trca_X1=trca_X1+X0;
                                        end
                                        mstrca_X1=[mstrca_X1 trca_X1];
                                        mstrca_X2=[mstrca_X2 trca_X2'];
                                    end
                                    S=mstrca_X1*mstrca_X1'-mstrca_X2*mstrca_X2';
                                    Q=mstrca_X2*mstrca_X2';
                                    [eig_v1,eig_d1]=eig(Q\S);
                                    [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
                                    eig_vec=eig_v1(:,sort_idx);
                                    W_msTRCA(sub_band).val=[W_msTRCA(sub_band).val; eig_vec(:,1)'];
                                    
                                    % template transfer
                                    d0=floor(num_of_signal_templates3/2);
                                    d1=n_sti;
                                    if my_j<=d0
                                        template_st=1;
                                        template_ed=num_of_signal_templates3;
                                    elseif ((my_j>d0) && my_j<(d1-d0+1))
                                        template_st=my_j-d0;
                                        template_ed=my_j+(num_of_signal_templates3-d0-1);
                                    else
                                        template_st=(d1-num_of_signal_templates3+1);
                                        template_ed=d1;
                                    end
                                    template_idx=[template_st:template_ed];
                                    source_ssvep_temp0=zeros(length(source_idx),sig_len*length(template_idx));
                                    for ssn=1:length(source_idx)
                                        stmp=[];
                                        
                                        for fn=1:length(template_idx)
                                            
                                            tmp2=subj(source_idx(ssn)).subband(sub_band).filtered_ssvep_template(template_idx(fn),1:sig_len);
                                            stmp=[stmp tmp2];
                                        end
                                        source_ssvep_temp0(ssn,:)=stmp;
                                    end
                                    
                                    Y=[];
                                    for fn=1:length(template_idx)
                                        tmp1=reshape(subband_signal(sub_band).signal_template(template_idx(fn),:,1:sig_len),d3,sig_len);
                                        
                                        Y=[Y (subband_signal(sub_band).ind_Wx'*tmp1)];
                                    end
                                    Y=Y';
                                    
                                    X=source_ssvep_temp0';
                                    
                                    W0=inv(X'*X)*X'*Y;
                                    W_template1=W0(:,1);
                                    
                                    if sum(abs(W_template1))==0
                                        W_template1=ones(1,length(source_idx)-1);
                                    end
                                    source_ssvep_temp=zeros(1,source_data_len);
                                    for ssn=1:length(source_idx)
                                        source_ssvep_temp=source_ssvep_temp+(W_template1(ssn))*subj(source_idx(ssn)).subband(sub_band).filtered_ssvep_template(my_j,1:source_data_len);
                                    end
                                    source_ssvep_temp=source_ssvep_temp/sum(abs(W_template1));
                                    subband_signal(sub_band).source_subject_filtered_template(my_j,:)=source_ssvep_temp;
                                end
                                
                            end
                            
                            %% ms-ecca spatial filter
                            if i==1
                                mscca_X=[];
                                mscca_Y=[];
                                d0=floor(num_of_signal_templates/2);
                                d1=n_sti;
                                if j<=d0
                                    template_st=1;
                                    template_ed=num_of_signal_templates;
                                elseif (j>d0 && j<(d1-d0+1))
                                    template_st=j-d0;
                                    template_ed=j+(num_of_signal_templates-d0-1);
                                else
                                    template_st=(d1-num_of_signal_templates+1);
                                    template_ed=d1;
                                end
                                template_idx=[template_st:template_ed];
                                for m=1:num_of_signal_templates
                                    mm=(template_idx(m));
                                    tmp=reshape(subband_signal(sub_band).signal_template(mm,:,1:sig_len),d3,sig_len);
                                    tmp=tmp-mean(tmp,2)*ones(1,size(tmp,2));                                    
                                    ref1=ck_signal_nh(sti_f(mm),Fs,pha_val(mm),sig_len,num_of_harmonics);
                                    mscca_X=[mscca_X,tmp];
                                    mscca_Y=[mscca_Y,ref1];
                                end
                                [A,B] = canoncorr(mscca_X',mscca_Y');
                                subband_signal(sub_band).mscca_Wx{j}=A(:,1);
                                subband_signal(sub_band).mscca_Wy{j}=B(:,1);
                            end                           
                            
                            
                            template2=reshape(subband_signal(sub_band).signal_template(j,:,1:sig_len),d3,sig_len);
                            template1=subband_signal(sub_band).source_subject_filtered_template(j,1:sig_len);
                            ref=ck_signal_nh(sti_f(j),Fs,pha_val(j),sig_len,num_of_harmonics);
                            
                            if (enable_bit(1)==1)
                                
                                CR(sub_band,j)=1;
                                CCAR(sub_band,j)=1;
                                
                                r1=corrcoef(subband_signal(sub_band).mscca_Wx{j}'*test_signal,subband_signal(sub_band).mscca_Wy{j}'*ref);
                                
                                r2=corrcoef(subband_signal(sub_band).ind_Wx'*test_signal,template1);
                                r3=corrcoef(subband_signal(sub_band).mscca_Wx{j}'*test_signal,subband_signal(sub_band).mscca_Wx{j}'*template2);
                                r4=corrcoef(W_msTRCA(sub_band).val*test_signal,W_msTRCA(sub_band).val*template2);
                                
                                itR(sub_band,j)=sign(r1(1,2))*r1(1,2)^2+...                                    
                                    sign(r4(1,2))*r4(1,2)^2;
                                ittR(sub_band,j)=sign(r1(1,2))*r1(1,2)^2+...                                    
                                    sign(r2(1,2))*r2(1,2)^2+...
                                    sign(r4(1,2))*r4(1,2)^2;
                                etrcaR(sub_band,j)=sign(r1(1,2))*r1(1,2)^2+...                                    
                                    sign(r3(1,2))*r3(1,2)^2;
                                etrca2R(sub_band,j)=sign(r1(1,2))*r1(1,2)^2+...                                    
                                    sign(r2(1,2))*r2(1,2)^2+...
                                    sign(r3(1,2))*r3(1,2)^2;
                                
                                
                            else
                                CR(sub_band,j)=0;
                                itR(sub_band,j)=0;
                                ittR(sub_band,j)=0;
                                CCAR(sub_band,j)=0;
                                etrcaR(sub_band,j)=0;
                                etrca2R(sub_band,j)=0;
                            end                          
                            
                            % ----- LST -----            
                            r_tmp = corrcoef(test_signal.'*squeeze(LST_spatial_filters{tw_length}(sub_band, :, :)).', ...
                               squeeze(LST_templates{tw_length}(sub_band, j, :, :)).'*squeeze(LST_spatial_filters{tw_length}(sub_band, :, :)).');
                            LST_r(sub_band,j) = sign(r_tmp(1, 2))*r_tmp(1, 2)^2 + sign(r1(1,2))*r1(1,2)^2;
                            % ----- LST -----
                        end  
                    end
                    CCAR1=sum((CCAR).*FB_coef,1);           
                    CR1=sum((CR).*FB_coef,1);
                    itR1=sum((itR).*FB_coef,1);             % ms-eCCA+ms-eTRCA
                    ittR1=sum((ittR).*FB_coef,1);           % ms-eCCA+ms-eTRCA (with subject ransfer)
                    etrcaR1=sum((etrcaR).*FB_coef,1);       % ms-eCCA
                    etrca2R1=sum((etrca2R).*FB_coef,1);     % ms-eCCA (with subject ransfer)
                    
                    [~,idx]=max(CCAR1);
                    if idx==i
                        n_correct(tw_length,1)=n_correct(tw_length,1)+1;
                    end
                    [~,idx]=max(CR1);
                    if idx==i
                        n_correct(tw_length,2)=n_correct(tw_length,2)+1;
                    end
                    [~,idx]=max(itR1);
                    if idx==i
                        n_correct(tw_length,3)=n_correct(tw_length,3)+1;
                    end
                    [~,idx]=max(ittR1);
                    if idx==i
                        n_correct(tw_length,4)=n_correct(tw_length,4)+1;
                    end
                    [~,idx]=max(etrcaR1);
                    if idx==i
                        n_correct(tw_length,5)=n_correct(tw_length,5)+1;
                    end
                    [~,idx]=max(etrca2R1);
                    if idx==i
                        n_correct(tw_length,6)=n_correct(tw_length,6)+1;
                    end
                    
                    % ----- LST -----
                    rho = FB_coef0*LST_r;
                    [~, tau] = max(rho);
                    if tau == i
                        n_correct(tw_length,7)=n_correct(tw_length,7)+1;
                    end
                    % ----- LST -----
                    
                end
                
            end
        end
        seq_0(run,:)=seq1;
    end
    toc
    
    accuracy=100*n_correct/n_sti/n_run/length(idx_testdata)
    save_acc(sn).accuracy=accuracy;
    filename=mfilename('fullpath');
    save_name=['th_stcca_acc_0216a_' num2str(num_of_signal_templates) '_' ...
        num2str(num_of_signal_templates2) '_' ...
        num2str(num_of_signal_templates3) '_' ...
        num2str(num_of_subbands) '_' ...
        num2str(length(ch_used)) '_' ...
        num2str(dataset_no) '_' ...
        num2str(num_of_trials) '.mat'];
    save(save_name,'num_of_signal_templates','num_of_signal_templates2',...
        'num_of_signal_templates3','num_of_subbands','ch_used',...
        'dataset_no','save_acc','filename',...
        'num_of_trials');
    
    disp(sn)
end


