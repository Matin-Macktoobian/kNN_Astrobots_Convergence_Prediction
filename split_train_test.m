function [Train_set, Test_set, index_train, index_test] = split_train_test(Data,percentage)
%This function splits in a random fashion the full dataset into training dataset and testing
%dataset according to a specified ratio

M=size(Data,3);
N_test = round(M*percentage);

index_test = datasample([1:M],N_test,'Replace',false);
aux=ones(1,M);
aux(index_test)=0;
index_train = find(aux);

Train_set = Data(:,:,index_train);
Test_set = Data(:,:,index_test);




end

