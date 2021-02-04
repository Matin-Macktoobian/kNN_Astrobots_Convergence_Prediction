clear all
close all
clc

%% Load the data from a folder
% This first section of the script is needed to load the data in a .csv
% format and transform them into a .mat format

% Insert the path where the data are located
data_path='C:\Users\Mauro\Desktop\Data 500 Positioners\CSV';
addpath(data_path);

% Count the number of data (the number of simulated configurations)
d = dir([data_path, '\*.csv']);
n_of_simulations=length(d);

%Loading the data and transform them into .mat format
for i=1:n_of_simulations
   
   Data_name = ['Data' num2str(i) '.csv'];
   T=readtable(Data_name, 'HeaderLines',1);
   T.Var5=[];
   T.Var6=[];
   
   %Transpose the data
   Datas_matrix(:,:,i)=table2array(T).';
   
   %The data are transformed from a representation in polar coordinates
   %into a representation in cartesian coordinates
   x_pos=Datas_matrix(1,:,i).*sind(Datas_matrix(2,:,i));
   y_pos=Datas_matrix(1,:,i).*cosd(Datas_matrix(2,:,i));
   x_tar=Datas_matrix(3,:,i).*sind(Datas_matrix(4,:,i));
   y_tar=Datas_matrix(3,:,i).*cosd(Datas_matrix(4,:,i));
   
   %Final format of the data which will be used in the algoritm.
   %Each simulated configuration is a matrix whose first two rows are the
   %coordinates of the centroids of the astrobots. The third and fourth
   %rows are represents the coordinates of the targets of each astrobots.
   %the final rows is the label (1 if the astrobot converges to its target; 
   %0 if the astrobot does not converge).
   Datas_matrix_xy(:,:,i)=[x_pos; y_pos; x_tar; y_tar; Datas_matrix(5,:,i)];
   
   
end

%% Load the data already in a .mat format
% Since some data used in the prediction are already available in a .mat
% format, it is possible to directly load them:
% The configurations available are: 
% - Focal plane with 30 Astrobots  ----> file 'Data_30_pos.mat' 
% - Focal plane with 116 Astrobots ----> file 'Data_116_pos.mat'
% - Focal plane with 487 Astrobots ----> file 'Data_487_pos.mat'

%Comment the following line of code if data are loaded from a folder in
%.csv format
load Data_487_pos.mat


%% Info about the convergence
% This little section provides information about the number of astrobots
% which converge and the number of focal plane configuration which have a
% complete coordination of the astrobots.

%Datas_matrix_xy=remove_partial_conv(Datas_matrix_xy,0.5);

%Count the total number of astrobots which converge to their target position and the
%total number of astrobots which don't converge to their target position
[astr1, astr0]=count_zeros_and_ones(Datas_matrix_xy);

% Count the number of configurations with full convergence (All the
% astrobots reach their targets) and the number of configurations without
% full convergence
[info_conv, index_not_conv]=count_conv(Datas_matrix_xy);

% the next part is needed to compute the percentage of convergence of the
% entire focal plane and of the single astrobots in the focal plane

n_of_astrobots = size(Datas_matrix_xy,2);  %Number of astrobots in the focal plane configuration

%Initialize the percentage of convergence vector
percentage_of_convergence_vec=[];

%This vector has length equal to the number of astrobots in the focal
%plane. Its elements reptresent the number of times a single astrobot
%reaches its target position
Convergence_single_astrobots=zeros(1,size(Datas_matrix_xy,2));

for t=1:size(Datas_matrix_xy,3)
    percentage_of_convergence_vec=[percentage_of_convergence_vec; mean(Datas_matrix_xy(5,:,t))];
    Convergence_single_astrobots = Convergence_single_astrobots + Datas_matrix_xy(5,:,t);
end

%Average percentage of convergence in the entire focal plane
mean_percentage_of_convergence = mean(percentage_of_convergence_vec);
fprintf('The average percentage of convergence for the astrobots in the focal plane is %.02f %%',  100*mean_percentage_of_convergence)

%Find how many neighbours has each astrobot
n_of_neigh=[];
    for k=1:n_of_astrobots
        n_of_neigh(1,k)=length(find_neighbour(Datas_matrix_xy,k));  %Find the number of neighbours of each astrobots
    end

%This vector has length equal to the number of astrobots in the focal
%plane. Its elements represent the number of times a single astrobot
%DOES NOT reache its target position
Not_convergence_single_astrobots=size(Datas_matrix_xy,3) - Convergence_single_astrobots;

%Transform the number of times in a percentage
Convergence_single_astrobots=100 * Convergence_single_astrobots/n_of_simulations;

%Plot the percentage of convergence of the single astrobots in the focal
%plane as a bar plot. The red bars are for the astrobots in a full
%neighborhood configuration. The yellow bars are for astrobots with a
%number of neighbors comprised between 3 and 5. The blue bars are for
%astrobots with less than 3 neighbors.
figure('Position', [10 10 2000 900]), grid on, hold on
for i=1:length(Convergence_single_astrobots)
    h=bar(i,Convergence_single_astrobots(i));
    if n_of_neigh(i)==6
        set(h,'FaceColor','r');
    elseif n_of_neigh(i) > 2 && n_of_neigh(i)<=5
        set(h,'FaceColor','y');
    else
        set(h,'FaceColor','b');
    end
end
ylabel('Percentage of convergence [%]', 'FontSize', 15)
xlabel('Astrobot', 'FontSize',15)
title('Percentage of convergence of the single astrobots in the focal plane','FontSize',20)

%% KNN PREDICTOR
%This part is the core of the algorithm

%PARAMETERS SET BY THE USER

%Number of iterations of the Cross Validation process
K=1;

%number of closest train configurations to take into account for the
%computation of the prediction
K_nearest_configurations = 51;

%Corrector coefficients for the weight vector
alpha=1;
beta=1;

%Ratio Test_dataset/Full_dataset
d_dataset = 0.0001;

%Treshold for the transformation of the vector of probability into a binary
%output
treshold=0.5;



%START OF THE ALGORITHM

%Initialize the vectors whose elements the number of TP, TN, FP, FN at each
%iteration of the cross validation process.
TP_k=[]; TN_k=[]; FP_k=[]; FN_k=[];

tic
%OUTER LOOP - Cross validation process
for u=1:K

    fprintf('\n iteration number: %d', u)

    %Split the full dataset into train set and test set
    [Train_set, Test_set, indtre, indtes]=split_train_test(Datas_matrix_xy, d_dataset);

    % The following lines of code are needed to find the vector of weights
    % w. In the script the vector of weights is denoted as weight_local   
    Convergence_single_astrobots=zeros(1,size(Train_set,2));
    for t=1:size(Train_set,3)
        Convergence_single_astrobots = Convergence_single_astrobots + Train_set(5,:,t);
    end
    Not_convergence_single_astrobots=size(Train_set,3) - Convergence_single_astrobots;
    weight_local=Convergence_single_astrobots./Not_convergence_single_astrobots;
    
    %Adjust the vector of weights according to the corrector coefficients
    %alpha and beta
    weight_local=weight_local.*(alpha*(n_of_neigh==6) + beta*(n_of_neigh~=6));

    %Initialization of the relevant quantities for analysing the results
    %at the end 
    TP=0; TN=0; FP=0; FN=0;
    TP_vec=zeros(size(Test_set,3),n_of_astrobots);
    TN_vec=zeros(size(Test_set,3),n_of_astrobots);
    FP_vec=zeros(size(Test_set,3),n_of_astrobots);
    FN_vec=zeros(size(Test_set,3),n_of_astrobots);
    clear acc_vec acc_single_astrobots

    %INNER LOOP - Predictor
    for prove=1:size(Test_set,3)
        
        %initialize the auxiliary vector of probability
        prob=[];
        
        %The normalization_factor vector is needed to compute the vector
        %eta
        normalization_factor = zeros(n_of_astrobots,n_of_astrobots);

        %LOCAL ANALYSIS
        for k=1:n_of_astrobots    %Analysing the neighbourgh of each astrobots

            %Find the neighbours of each astrobots
            neigh=find_neighbour(Datas_matrix_xy,k);
            
            %The vector astrobots collects the indices of the neighborhood
            %under analysis
            astrobots=[k,neigh];
            
            %Taking into account which astrobots have appeared in the
            %neighborhood
            normalization_factor(k,astrobots)=1;

            %Distance metric:
            %Return the index of the K minimum distance considering only the neighbour of the considered astrobots
            [mindist,index]=min_dist(Train_set(:,astrobots,:),Test_set(:,astrobots,prove),K_nearest_configurations);  
            
           
            Neighbour_data=Train_set(:,:,index);  %"Neighbour" datas considering only the neighbour of the astrobots
            
            %auxiliary probability vector for the specific neighborhood
            %under analysis
            prob=[prob; prediction_probability_weighted_tot(Neighbour_data,weight_local,astrobots)];


        end
        
        %Computing the eta vector
        eta=sum(normalization_factor,1);
        
        %Final probability vector
        prob_2=sum(prob,1)./eta;
        
        %Transforming the probability vector into a binary output vector
        binary_output2=(prob_2>treshold);  
        
        %Ground truth vector of the test configuration under analysis
        real_value=Test_set(end,:,prove);
        
        acc_single_astrobots(prove,:)=(binary_output2==real_value);
        
        %Calculating the number of TP, TN, FP, FN of the test configuration
        %under analysis
        
        for t=1:n_of_astrobots

            if binary_output2(t)==1 && real_value(t)==1
                TP = TP+1;
                TP_vec(prove,t)=1;          
            elseif binary_output2(t)==1 && real_value(t)==0
                FP = FP+1;
                FP_vec(prove,t)=1;
            elseif binary_output2(t)==0 && real_value(t)==1
                FN = FN + 1;
                FN_vec(prove,t)=1;
            else
                TN = TN + 1;
                TN_vec(prove,t)=1;
            end        
        end
    

    end
    
    %Saving the results in the vectors for the further analysis of the
    %results
    
    acc_single_astrobots_k(u,:)=(sum(acc_single_astrobots,1)/size(acc_single_astrobots,1))*100;
    TP_k=[TP_k,TP];
    FP_k=[FP_k,FP];
    TN_k=[TN_k,TN];
    FN_k=[FN_k,FN];
    TP_vec_k(u,:)=sum(TP_vec,1);
    TN_vec_k(u,:)=sum(TN_vec,1);
    FP_vec_k(u,:)=sum(FP_vec,1);
    FN_vec_k(u,:)=sum(FN_vec,1);    

end
toc

%% Analysis of the results

%Calculating the average TP, FP, TN, FN over the iterations of cross
%validation
TP_mean=sum(TP_k)/K;
FP_mean=sum(FP_k)/K;
TN_mean=sum(TN_k)/K;
FN_mean=sum(FN_k)/K;

%Confusion matrix and accuracy
 C=round([[TN_mean,FN_mean];[FP_mean, TP_mean]]);
 plotConfMat(C)


%Plotting the average number of true positives for each astrobot
disp('True positive for each astrobots - mean over folds')
sum(TP_vec_k,1)/K;
figure, grid on
bar(sum(TP_vec_k,1)/K)
title('True positive for each astrobot - mean over folds')

%Plotting the average number of false positives for each astrobot
disp('False positive for each astrobots - mean over folds')
sum(FP_vec_k,1)/K;
figure, grid on
bar(sum(FP_vec_k,1)/K)
title('False positive for each astrobot - mean over folds')

%Plotting the average number of true negatives for each astrobot
disp('True negative for each astrobots - mean over folds')
sum(TN_vec_k,1)/K;
figure, grid on
bar(sum(TN_vec_k,1)/K)
title('True negative for each astrobot - mean over folds')

%Plotting the average number of false negatives for each astrobot
disp('False negative for each astrobots - mean over folds')
sum(FN_vec_k,1)/K;
figure, grid on
bar(sum(FN_vec_k,1)/K)
title('False negative for each astrobot - mean over folds')

TNR = 100*((sum(TN_vec_k,1)/K)./(sum(TN_vec_k,1)/K + sum(FP_vec_k,1)/K));
TPR = 100*((sum(TP_vec_k,1)/K)./(sum(TP_vec_k,1)/K + sum(FN_vec_k,1)/K));

BalAcc = (TNR + TPR)/2;

%Plot the TPR, TNR and Balanced accuracy for each type of neighborhood
plot_acc_astrobot_neighbors(TPR, TNR, BalAcc, n_of_neigh)

%Plotting the average balanced accuracy for each astrobot
disp('Balanced accuracy - mean over folds')
BalAcc;
figure, grid on
bar(BalAcc)
title('Balanced accuracy - mean over folds')

